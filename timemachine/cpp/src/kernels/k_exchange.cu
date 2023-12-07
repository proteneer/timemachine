
#include "../gpu_utils.cuh"
#include "k_exchange.cuh"
#include "k_fixed_point.cuh"
#include "k_logsumexp.cuh"
#include <assert.h>

namespace timemachine {

void __global__ k_setup_sample_atoms(
    const int sample_atoms,          // number of atoms in each sample
    const int *__restrict__ samples, // [1]
    const int *__restrict__ target_atoms,
    const int *__restrict__ mol_offsets,
    int *__restrict__ output_atom_idxs,
    int *__restrict__ output_mol_offsets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(gridDim.x == 1);
    if (idx > 0) {
        return;
    }
    int mol_idx = samples[idx];
    int mol_start = mol_offsets[mol_idx];
    int mol_end = mol_offsets[mol_idx + 1];
    output_mol_offsets[mol_idx] = target_atoms[mol_start];
    output_mol_offsets[mol_idx + 1] = target_atoms[mol_end - 1] + 1;
    int num_atoms = mol_end - mol_start;

    assert(num_atoms == sample_atoms);

    for (int i = 0; i < num_atoms; i++) {
        output_atom_idxs[idx * sample_atoms + i] = target_atoms[mol_start + i];
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
    RealType before_log_prob = convert_nan_to_inf<RealType>(compute_logsumexp_final<RealType>(before_log_sum_exp));
    RealType after_log_prob = convert_nan_to_inf<RealType>(compute_logsumexp_final<RealType>(after_log_sum_exp));

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

template void __global__ k_attempt_exchange_move<float>(
    const int N,
    const float *__restrict__ rand,
    const float *__restrict__ before_log_sum_exp,
    const float *__restrict__ after_log_sum_exp,
    const double *__restrict__ moved_coords,
    double *__restrict__ dest_coords,
    size_t *__restrict__ num_accepted);
template void __global__ k_attempt_exchange_move<double>(
    const int N,
    const double *__restrict__ rand,
    const double *__restrict__ before_log_sum_exp,
    const double *__restrict__ after_log_sum_exp,
    const double *__restrict__ moved_coords,
    double *__restrict__ dest_coords,
    size_t *__restrict__ num_accepted);

template <typename RealType>
void __global__ k_attempt_exchange_move_targeted(
    const int N,
    const int *__restrict__ targeting_inner_volume,
    const RealType *__restrict__ box_vol, // [1]
    const RealType inner_volume,
    const RealType *__restrict__ rand,               // [1]
    const RealType *__restrict__ before_log_sum_exp, // [2]
    const RealType *__restrict__ after_log_sum_exp,  // [2]
    const double *__restrict__ moved_coords,         // [N, 3]
    double *__restrict__ dest_coords,                // [N, 3]
    size_t *__restrict__ num_accepted                // [1]
) {
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int targeting_inner = targeting_inner_volume[0];

    const RealType outer_vol = box_vol[0] - inner_volume;

    const RealType log_vol_prob =
        targeting_inner == 1 ? log(inner_volume) - log(outer_vol) : log(outer_vol) - log(inner_volume);

    // All kernels compute the same acceptance
    // TBD investigate shared memory for speed
    RealType before_log_prob = convert_nan_to_inf<RealType>(compute_logsumexp_final<RealType>(before_log_sum_exp));
    RealType after_log_prob = convert_nan_to_inf<RealType>(compute_logsumexp_final<RealType>(after_log_sum_exp));

    RealType log_acceptance_prob = min(before_log_prob - after_log_prob + log_vol_prob, static_cast<RealType>(0.0));

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

template void __global__ k_attempt_exchange_move_targeted<float>(
    const int N,
    const int *__restrict__ targeting_inner_volume,
    const float *__restrict__ box_vol,
    const float inner_volume,
    const float *__restrict__ rand,
    const float *__restrict__ before_log_sum_exp,
    const float *__restrict__ after_log_sum_exp,
    const double *__restrict__ moved_coords,
    double *__restrict__ dest_coords,
    size_t *__restrict__ num_accepted);
template void __global__ k_attempt_exchange_move_targeted<double>(
    const int N,
    const int *__restrict__ targeting_inner_volume,
    const double *__restrict__ box_vol,
    const double inner_volume,
    const double *__restrict__ rand,
    const double *__restrict__ before_log_sum_exp,
    const double *__restrict__ after_log_sum_exp,
    const double *__restrict__ moved_coords,
    double *__restrict__ dest_coords,
    size_t *__restrict__ num_accepted);

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
    assert(gridDim.x == 1);
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

template void __global__ k_store_accepted_log_probability<float>(
    const int num_weights,
    const float *__restrict__ rand,
    float *__restrict__ before_log_sum_exp,
    const float *__restrict__ after_log_sum_exp,
    float *__restrict__ before_weights,
    const float *__restrict__ after_weights);
template void __global__ k_store_accepted_log_probability<double>(
    const int num_weights,
    const double *__restrict__ rand,
    double *__restrict__ before_log_sum_exp,
    const double *__restrict__ after_log_sum_exp,
    double *__restrict__ before_weights,
    const double *__restrict__ after_weights);

template <typename RealType>
void __global__ k_store_accepted_log_probability_targeted(
    const int num_weights,
    const int *__restrict__ targeting_inner_volume,
    const RealType *__restrict__ box_vol, // [1]
    const RealType inner_volume,
    const RealType *__restrict__ rand,              // [1]
    RealType *__restrict__ before_log_sum_exp,      // [2]
    const RealType *__restrict__ after_log_sum_exp, // [2]
    RealType *__restrict__ before_weights,          // [num_weights]
    const RealType *__restrict__ after_weights      // [num_weights]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(gridDim.x == 1);
    if (blockIdx.x > 0) {
        return; // Only one block can run this
    }

    int flag = targeting_inner_volume[0];

    const RealType outer_vol = box_vol[0] - inner_volume;

    const RealType log_vol_prob = flag == 1 ? log(inner_volume) - log(outer_vol) : log(outer_vol) - log(inner_volume);

    // All kernels compute the same acceptance
    // TBD investigate shared memory for speed
    RealType before_log_prob = convert_nan_to_inf<RealType>(compute_logsumexp_final<RealType>(before_log_sum_exp));
    RealType after_log_prob = convert_nan_to_inf<RealType>(compute_logsumexp_final<RealType>(after_log_sum_exp));

    RealType log_acceptance_prob = min(before_log_prob - after_log_prob + log_vol_prob, static_cast<RealType>(0.0));
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

template void __global__ k_store_accepted_log_probability_targeted<float>(
    const int num_weights,
    const int *__restrict__ targeting_inner_volume,
    const float *__restrict__ box_vol,
    const float inner_volume,
    const float *__restrict__ rand,
    float *__restrict__ before_log_sum_exp,
    const float *__restrict__ after_log_sum_exp,
    float *__restrict__ before_weights,
    const float *__restrict__ after_weights);
template void __global__ k_store_accepted_log_probability_targeted<double>(
    const int num_weights,
    const int *__restrict__ targeting_inner_volume,
    const double *__restrict__ box_vol,
    const double inner_volume,
    const double *__restrict__ rand,
    double *__restrict__ before_log_sum_exp,
    const double *__restrict__ after_log_sum_exp,
    double *__restrict__ before_weights,
    const double *__restrict__ after_weights);

template <typename RealType>
void __global__ k_compute_box_volume(
    const double *__restrict__ box,      // [3]
    RealType *__restrict__ output_volume // [1]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) {
        return;
    }
    RealType out = box[0 * 3 + 0] * box[1 * 3 + 1] * box[2 * 3 + 2];
    output_volume[idx] = out;
}

template void __global__ k_compute_box_volume<float>(
    const double *__restrict__ box,   // [3]
    float *__restrict__ output_volume // [1]
);
template void __global__ k_compute_box_volume<double>(
    const double *__restrict__ box,    // [3]
    double *__restrict__ output_volume // [1]
);

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

template void __global__ k_adjust_weights<float, 0>(
    const int N,
    const int num_target_mols,
    const int mol_size,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const float *__restrict__ per_atom_energies,
    const float inv_kT,
    float *__restrict__ log_weights);
template void __global__ k_adjust_weights<float, 1>(
    const int N,
    const int num_target_mols,
    const int mol_size,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const float *__restrict__ per_atom_energies,
    const float inv_kT,
    float *__restrict__ log_weights);

template void __global__ k_adjust_weights<double, 0>(
    const int N,
    const int num_target_mols,
    const int mol_size,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const double *__restrict__ per_atom_energies,
    const double inv_kT,
    double *__restrict__ log_weights);
template void __global__ k_adjust_weights<double, 1>(
    const int N,
    const int num_target_mols,
    const int mol_size,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const double *__restrict__ per_atom_energies,
    const double inv_kT,
    double *__restrict__ log_weights);

template <typename RealType, int THREADS_PER_BLOCK>
void __global__ k_set_sampled_weight(
    const int N,
    const int mol_size,
    const int *__restrict__ samples, // [1]
    const int *__restrict__ target_atoms,
    const int *__restrict__ mol_offsets,
    const RealType *__restrict__ per_atom_energies,
    const RealType inv_kT, // 1 / kT
    RealType *__restrict__ log_weights) {
    static_assert(THREADS_PER_BLOCK <= 512 && (THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0);
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

template void __global__ k_set_sampled_weight<float, 512>(
    const int N,
    const int mol_size,
    const int *__restrict__ samples, // [1]
    const int *__restrict__ target_atoms,
    const int *__restrict__ mol_offsets,
    const float *__restrict__ per_atom_energies,
    const float inv_kT,
    float *__restrict__ log_weights);
template void __global__ k_set_sampled_weight<double, 512>(
    const int N,
    const int mol_size,
    const int *__restrict__ samples, // [1]
    const int *__restrict__ target_atoms,
    const int *__restrict__ mol_offsets,
    const double *__restrict__ per_atom_energies,
    const double inv_kT,
    double *__restrict__ log_weights);

template <typename RealType>
void __global__ k_compute_centroid_of_atoms(
    const int num_atoms,
    const int *__restrict__ atom_idxs, // [num_atoms]
    const double *__restrict__ coords, // [N, 3]
    RealType *__restrict__ centroid    // [3]
) {
    __shared__ unsigned long long fixed_centroid[3];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Can only have a single block
    assert(blockIdx.x == 0);

    int atom_idx;

    if (threadIdx.x < 3) {
        fixed_centroid[threadIdx.x] = 0;
    }
    __syncthreads();

    while (idx < num_atoms) {
        atom_idx = atom_idxs[idx];

        // TBD: Account for pbc? Could be atoms from different mols theoretically
        atomicAdd(fixed_centroid + 0, FLOAT_TO_FIXED<RealType>(coords[atom_idx * 3 + 0]));
        atomicAdd(fixed_centroid + 1, FLOAT_TO_FIXED<RealType>(coords[atom_idx * 3 + 1]));
        atomicAdd(fixed_centroid + 2, FLOAT_TO_FIXED<RealType>(coords[atom_idx * 3 + 2]));

        idx += gridDim.x * blockDim.x;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        centroid[0] = FIXED_TO_FLOAT<RealType>(fixed_centroid[0]) / static_cast<RealType>(num_atoms);
        centroid[1] = FIXED_TO_FLOAT<RealType>(fixed_centroid[1]) / static_cast<RealType>(num_atoms);
        centroid[2] = FIXED_TO_FLOAT<RealType>(fixed_centroid[2]) / static_cast<RealType>(num_atoms);
    }
}

template void __global__ k_compute_centroid_of_atoms<float>(
    const int num_atoms,
    const int *__restrict__ atom_idxs,
    const double *__restrict__ coords,
    float *__restrict__ centroid);
template void __global__ k_compute_centroid_of_atoms<double>(
    const int num_atoms,
    const int *__restrict__ atom_idxs,
    const double *__restrict__ coords,
    double *__restrict__ centroid);

template <typename RealType>
void __global__ k_flag_mols_inner_outer(
    const int num_molecules,
    const int *__restrict__ atom_idxs,
    const int *__restrict__ mol_offsets, // [num_molecules + 1]
    const RealType *__restrict__ center, // [3] Center that determines inner vs outer
    const RealType square_radius,        // squared radius from center that defines inner
    const double *__restrict__ coords,   // [N, 3]
    const double *__restrict__ box,      // [3, 3]
    int *__restrict__ inner_flags        // [num_molecules]
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const RealType box_x = box[0 * 3 + 0];
    const RealType box_y = box[1 * 3 + 1];
    const RealType box_z = box[2 * 3 + 2];

    const RealType inv_box_x = 1 / box_x;
    const RealType inv_box_y = 1 / box_y;
    const RealType inv_box_z = 1 / box_z;

    const RealType target_x = center[0];
    const RealType target_y = center[1];
    const RealType target_z = center[2];

    while (mol_idx < num_molecules) {
        int mol_start = mol_offsets[mol_idx];
        int mol_end = mol_offsets[mol_idx + 1];
        int start_idx = atom_idxs[mol_start];
        int num_atoms = mol_end - mol_start;

        unsigned long long centroid_accum_x = 0;
        unsigned long long centroid_accum_y = 0;
        unsigned long long centroid_accum_z = 0;
        for (int atom_idx = start_idx; atom_idx < start_idx + num_atoms; atom_idx++) {
            centroid_accum_x += FLOAT_TO_FIXED<RealType>(coords[atom_idx * 3 + 0]);
            centroid_accum_y += FLOAT_TO_FIXED<RealType>(coords[atom_idx * 3 + 1]);
            centroid_accum_z += FLOAT_TO_FIXED<RealType>(coords[atom_idx * 3 + 2]);
        }

        RealType centroid_x = FIXED_TO_FLOAT<RealType>(centroid_accum_x) / static_cast<RealType>(num_atoms);
        RealType centroid_y = FIXED_TO_FLOAT<RealType>(centroid_accum_y) / static_cast<RealType>(num_atoms);
        RealType centroid_z = FIXED_TO_FLOAT<RealType>(centroid_accum_z) / static_cast<RealType>(num_atoms);

        centroid_x -= target_x;
        centroid_y -= target_y;
        centroid_z -= target_z;

        centroid_x -= box_x * nearbyint(centroid_x * inv_box_x);
        centroid_y -= box_y * nearbyint(centroid_y * inv_box_y);
        centroid_z -= box_z * nearbyint(centroid_z * inv_box_z);

        RealType dist = (centroid_x * centroid_x) + (centroid_y * centroid_y) + (centroid_z * centroid_z);
        inner_flags[mol_idx] = dist < square_radius ? 1 : 0;

        mol_idx += gridDim.x * blockDim.x;
    }
}

template void __global__ k_flag_mols_inner_outer<float>(
    const int num_molecules,
    const int *__restrict__ atom_idxs,
    const int *__restrict__ mol_offsets, // [num_molecules + 1]
    const float *__restrict__ center,    // [3] Center that determines inner vs outer
    const float square_radius,           // squared radius from center that defines inner
    const double *__restrict__ coords,   // [N, 3]
    const double *__restrict__ box,      // [3, 3]
    int *__restrict__ inner_flags        // [num_molecules]
);

template void __global__ k_flag_mols_inner_outer<double>(
    const int num_molecules,
    const int *__restrict__ atom_idxs,
    const int *__restrict__ mol_offsets, // [num_molecules + 1]
    const double *__restrict__ center,   // [3] Center that determines inner vs outer
    const double square_radius,          // squared radius from center that defines inner
    const double *__restrict__ coords,   // [N, 3]
    const double *__restrict__ box,      // [3, 3]
    int *__restrict__ inner_flags        // [num_molecules]
);

template <typename RealType>
void __global__ k_decide_targeted_move(
    const int num_target_mols,
    const RealType *__restrict__ rand,
    const int *__restrict__ inner_count,
    int *__restrict__ targeting_inner_volume) {
    const int count_inside = inner_count[0];
    const int count_outside = num_target_mols - count_inside;
    if (count_inside == 0 && count_outside == 0) {
        assert(0);
    } else if (count_inside > 0 && count_outside == 0) {
        targeting_inner_volume[0] = 0;
    } else if (count_inside == 0 && count_outside > 0) {
        targeting_inner_volume[0] = 1;
    } else if (count_inside > 0 && count_outside > 0) {
        if (rand[0] < static_cast<RealType>(0.5)) {
            targeting_inner_volume[0] = 1;
        } else {
            targeting_inner_volume[0] = 0;
        }
    } else {
        assert(0);
    }
}

template void __global__ k_decide_targeted_move<float>(
    const int num_target_mols,
    const float *__restrict__ rand,
    const int *__restrict__ inner_count,
    int *__restrict__ targeting_inner_volume);
template void __global__ k_decide_targeted_move<double>(
    const int num_target_mols,
    const double *__restrict__ rand,
    const int *__restrict__ inner_count,
    int *__restrict__ targeting_inner_volume);

// k_separate_weights_for_targeted takes the flag and the mol indices and writes them out
// to a new buffer the weights associated with the source molecules.
template <typename RealType>
void __global__ k_separate_weights_for_targeted(
    const int num_target_mols,
    const int *__restrict__ targeting_inner_volume, // [1]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    const RealType *__restrict__ weights,           // [num_target_mols]
    RealType *__restrict__ output_weights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int target_inner = targeting_inner_volume[0];
    const int local_inner_count = inner_count[0];
    const int outer_count = num_target_mols - local_inner_count;
    const int count = target_inner == 1 ? outer_count : local_inner_count;
    const int offset = target_inner == 1 ? local_inner_count : 0;
    while (idx < count) {
        output_weights[idx] = weights[partitioned_indices[idx + offset]];

        idx += gridDim.x * blockDim.x;
    }
}

template void __global__ k_separate_weights_for_targeted<float>(
    const int num_target_mols,
    const int *__restrict__ targeting_inner_volume, // [1]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    const float *__restrict__ weights,              // [num_target_mols]
    float *__restrict__ output_weights);
template void __global__ k_separate_weights_for_targeted<double>(
    const int num_target_mols,
    const int *__restrict__ targeting_inner_volume, // [1]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    const double *__restrict__ weights,             // [num_target_mols]
    double *__restrict__ output_weights);

template <typename RealType>
void __global__ k_setup_destination_weights_for_targeted(
    const int num_target_mols,
    const int *__restrict__ samples,                // [1]
    const int *__restrict__ targeting_inner_volume, // [1]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    const RealType *__restrict__ weights,           // [num_target_mols]
    RealType *__restrict__ output_weights           // [num_target_mols] Only access up to count + 1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int target_inner = targeting_inner_volume[0];
    const int local_inner_count = inner_count[0];
    const int outer_count = num_target_mols - local_inner_count;
    const int count = target_inner == 1 ? local_inner_count : outer_count;
    const int offset = target_inner == 1 ? 0 : local_inner_count;
    // Handle the sampled molecule being moved from one region to another by appending
    // the sampled mol's weight to the target region's weight.
    if (idx == 0) {
        int sample_idx = samples[idx];
        output_weights[count + idx] = weights[sample_idx];
    }
    while (idx < count) {
        output_weights[idx] = weights[partitioned_indices[idx + offset]];

        idx += gridDim.x * blockDim.x;
    }
}

template void __global__ k_setup_destination_weights_for_targeted<float>(
    const int num_target_mols,
    const int *__restrict__ samples,                // [1]
    const int *__restrict__ targeting_inner_volume, // [1]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    const float *__restrict__ weights,              // [num_target_mols]
    float *__restrict__ output_weights);

template void __global__ k_setup_destination_weights_for_targeted<double>(
    const int num_target_mols,
    const int *__restrict__ samples,                // [1]
    const int *__restrict__ targeting_inner_volume, // [1]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    const double *__restrict__ weights,             // [num_target_mols]
    double *__restrict__ output_weights);

void __global__ k_adjust_sample_idx(
    const int *__restrict__ targeting_inner_volume, // [1]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    int *__restrict__ sample_idxs                   // [1]
) {
    const int target_inner = targeting_inner_volume[0];
    const int local_inner_count = inner_count[0];
    const int offset = target_inner == 1 ? local_inner_count : 0;
    sample_idxs[0] = partitioned_indices[sample_idxs[0] + offset];
}

} // namespace timemachine
