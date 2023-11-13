#include "../gpu_utils.cuh"
#include "k_fixed_point.cuh"
#include "k_logsumexp.cuh"
#include <assert.h>

namespace timemachine {

// When we are considering exchange we want to treat Nan probabilities as inf
// Allows us to go from a clashy state to a non-clashy state. And no nan poisoning
template <typename RealType> RealType __host__ __device__ convert_nan_to_inf(const RealType input) {
    if (isnan(input)) {
        return INFINITY;
    }
    return input;
}

void __global__ k_setup_sample_atoms(
    const int num_samples,
    const int sample_atoms,          // number of atoms in each sample
    const int *__restrict__ samples, // [num_samples]
    const int *__restrict__ target_atoms,
    const int *__restrict__ mol_offsets,
    int *__restrict__ output_atom_idxs,
    int *__restrict__ output_mol_offsets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < num_samples) {
        int mol_idx = samples[idx];
        int mol_start = mol_offsets[mol_idx];
        int mol_end = mol_offsets[mol_idx + 1];
        output_mol_offsets[mol_idx] = target_atoms[mol_start];
        output_mol_offsets[mol_idx + 1] = target_atoms[mol_end - 1] + 1;
        int num_atoms = mol_end - mol_start;

        assert(num_atoms == sample_atoms);

        for (int i = 0; i < num_atoms; i++) {
            output_atom_idxs[idx * num_samples * sample_atoms + i] = target_atoms[mol_start + i];
        }

        idx += gridDim.x * blockDim.x;
    }
}

template <typename RealType>
void __global__ k_attempt_exchange_move(
    const int N,
    const RealType *__restrict__ rand,               // [1]
    const RealType *__restrict__ before_log_sum_exp, // [2]
    const RealType *__restrict__ after_log_sum_exp,  // [2]
    const double *__restrict__ moved_coords,         // [N, 3]
    double *__restrict__ dest_coords,                // [N, 3]
    size_t *__restrict__ num_accepted                // [1]
) {
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // All kernels compute the same acceptance
    // TBD investigate shared memory for speed
    RealType before_log_prob = convert_nan_to_inf(compute_logsumexp_final<RealType>(before_log_sum_exp));
    RealType after_log_prob = convert_nan_to_inf(compute_logsumexp_final<RealType>(after_log_sum_exp));

    RealType log_acceptance_prob = min(before_log_prob - after_log_prob, static_cast<RealType>(0.0));
    const bool accepted = rand[0] < exp(log_acceptance_prob);
    if (atom_idx == 0 && accepted) {
        num_accepted[0]++;
    }

    // If accepted, move the coords into place
    while (accepted && atom_idx < N) {
        dest_coords[atom_idx * 3 + 0] = moved_coords[atom_idx * 3 + 0];
        dest_coords[atom_idx * 3 + 1] = moved_coords[atom_idx * 3 + 1];
        dest_coords[atom_idx * 3 + 2] = moved_coords[atom_idx * 3 + 2];
        atom_idx += gridDim.x * blockDim.x;
    }
}

template <typename RealType>
void __global__ k_store_accepted_log_probability(
    const int num_weights,
    const RealType *__restrict__ rand,              // [1]
    RealType *__restrict__ before_log_sum_exp,      // [2]
    const RealType *__restrict__ after_log_sum_exp, // [2]
    RealType *__restrict__ before_weights,          // [num_weights]
    const RealType *__restrict__ after_weights      // [num_weights]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0) {
        return; // Only one block can run this
    }

    RealType before_log_prob = convert_nan_to_inf(compute_logsumexp_final<RealType>(before_log_sum_exp));
    RealType after_log_prob = convert_nan_to_inf(compute_logsumexp_final<RealType>(after_log_sum_exp));

    RealType log_acceptance_prob = min(before_log_prob - after_log_prob, static_cast<RealType>(0.0));
    const bool accepted = rand[0] < exp(log_acceptance_prob);
    if (!accepted) {
        return;
    }
    __syncthreads();
    // Swap the values after all threads have computed the log probability
    if (idx == 0) {
        before_log_sum_exp[0] = after_log_sum_exp[0];
        before_log_sum_exp[1] = after_log_sum_exp[1];
    }
    // Copy over the weights
    while (idx < num_weights) {
        before_weights[idx] = after_weights[idx];
        idx += gridDim.x * blockDim.x;
    }
}

// k_adjust_weights takes a set of molecules and either subtracts (Negated=true) or adds (Negated=false)
// the sum of the per atom weights for the molecules from some initial weights.
// This is used to do the transposition trick where we subtract off the weight contribution of the
// moved atom followed by adding back in the weight of the sampled mol in the new position.
// Does NOT special case the weight of the sampled mol and instead use `k_set_sampled_weight`.
template <typename RealType, bool Negated>
void __global__ k_adjust_weights(
    const int N,
    const int num_target_mols,
    const int mol_size,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const RealType *__restrict__ per_atom_energies,
    const RealType inv_kT, // 1 / kT
    RealType *__restrict__ log_weights) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (mol_idx < num_target_mols) {

        RealType current_log_weight = log_weights[mol_idx];
        __int128 weight_accumulator = 0;

        int mol_start = mol_offsets[mol_idx];
        int mol_end = mol_offsets[mol_idx + 1];
        int min_atom_idx = mol_atoms_idxs[mol_start];
        int max_atom_idx = mol_atoms_idxs[mol_end - 1];

        // A loop that in the case of water will be 3x3
        for (int i = 0; i < mol_size; i++) {
            for (int j = min_atom_idx; j <= max_atom_idx; j++) {
                weight_accumulator += FLOAT_TO_FIXED_ENERGY<RealType>(inv_kT * per_atom_energies[i * N + j]);
            }
        }

        weight_accumulator = Negated ? FLOAT_TO_FIXED_ENERGY<RealType>(current_log_weight) - weight_accumulator
                                     : FLOAT_TO_FIXED_ENERGY<RealType>(current_log_weight) + weight_accumulator;

        log_weights[mol_idx] =
            fixed_point_overflow(weight_accumulator) ? INFINITY : FIXED_ENERGY_TO_FLOAT<RealType>(weight_accumulator);

        mol_idx += gridDim.x * blockDim.x;
    }
}

template <typename RealType, int THREADS_PER_BLOCK>
void __global__ k_set_sampled_weight(
    const int N,
    const int mol_size,
    const int num_samples,
    const int *__restrict__ samples, // [num_samples]
    const int *__restrict__ target_atoms,
    const int *__restrict__ mol_offsets,
    const RealType *__restrict__ per_atom_energies,
    const RealType inv_kT, // 1 / kT
    RealType *__restrict__ log_weights) {
    static_assert(THREADS_PER_BLOCK <= 512 && (THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0);
    assert(num_samples == 1);
    assert(gridDim.x == 1);
    __shared__ __int128 accumulators[THREADS_PER_BLOCK];

    int sample_idx = samples[0];
    int min_atom_idx = target_atoms[0];
    int max_atom_idx = target_atoms[mol_size - 1];

    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    __int128 accumulator = 0;
    // Zero all of the accumulators
    while (atom_idx < N) {
        if (atom_idx < min_atom_idx || atom_idx > max_atom_idx) {
            for (int i = 0; i < mol_size; i++) {
                accumulator += FLOAT_TO_FIXED_ENERGY<RealType>(inv_kT * per_atom_energies[i * N + atom_idx]);
            }
        }
        atom_idx += gridDim.x * blockDim.x;
    }
    accumulators[threadIdx.x] = accumulator;
    __syncthreads();
    block_energy_reduce<THREADS_PER_BLOCK>(accumulators, threadIdx.x);
    if (threadIdx.x == 0) {
        log_weights[sample_idx] =
            fixed_point_overflow(accumulators[0]) ? INFINITY : FIXED_ENERGY_TO_FLOAT<RealType>(accumulators[0]);
    }
}

} // namespace timemachine
