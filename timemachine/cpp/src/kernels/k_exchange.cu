
#include "../gpu_utils.cuh"
#include "k_exchange.cuh"
#include "k_fixed_point.cuh"
#include <assert.h>

namespace timemachine {

template <typename RealType>
void __global__
k_copy_batch(const int N, const int batch_size, const RealType *__restrict__ src, RealType *__restrict__ dest) {
    int idx_in_batch = blockIdx.y;
    while (idx_in_batch < batch_size) {
        int offset = idx_in_batch * N;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < N) {
            dest[offset + idx] = src[idx];

            idx += gridDim.x * blockDim.x;
        }
        idx_in_batch += gridDim.y * blockDim.y;
    }
}

template void __global__
k_copy_batch<float>(const int N, const int batch_size, const float *__restrict__ src, float *__restrict__ dest);

template void __global__
k_copy_batch<double>(const int N, const int batch_size, const double *__restrict__ src, double *__restrict__ dest);

// k_setup_proposals takes a set of sampled indices and constructs buffers containing the molecule offsets
// as well as the atom indices (refer to src/mol_utils.hpp for impl) and setups up the offsets and the atom
// indices for each sample. Note that the output mol offsets are constructed so that the start of the mol is
// the starting atom idx rather than the prefix sum of mol lengths that the mol_offsets is.
void __global__ k_setup_proposals(
    const int total_proposals,
    const int batch_size,            // Number of proposals to setup
    const int num_atoms_in_each_mol, // number of atoms in each sample
    const int *__restrict__ rand_offset,
    const int *__restrict__ mol_idx_per_batch, // [batch_size] The index of the molecules to sample
    const int *__restrict__ atom_indices,      // [N]
    const int *__restrict__ mol_offsets,       // [num_target_mols]
    int *__restrict__ output_atom_idxs,        // [batch_size, num_atoms_in_each_mol]
    int *__restrict__ output_mol_offsets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int noise_offset = rand_offset[0];
    while (idx < batch_size) {
        if (noise_offset + idx >= total_proposals) {
            return;
        }
        int mol_idx = mol_idx_per_batch[idx];
        int mol_start = mol_offsets[mol_idx];

        int mol_end = mol_offsets[mol_idx + 1];
        output_mol_offsets[mol_idx] = atom_indices[mol_start];
        output_mol_offsets[mol_idx + 1] = atom_indices[mol_end - 1] + 1;
        int num_atoms = mol_end - mol_start;

        assert(num_atoms == num_atoms_in_each_mol);

        for (int i = 0; i < num_atoms; i++) {
            output_atom_idxs[idx * num_atoms_in_each_mol + i] = atom_indices[mol_start + i];
        }
        idx += gridDim.x * blockDim.x;
    }
}

void __global__ k_accepted_exchange_move(
    const int batch_size,
    const int num_atoms_in_each_mol,
    const int *__restrict__ accepted_batched_move, // [1]
    const int *__restrict__ mol_idx_per_batch,     // [batch_size]
    const int *__restrict__ mol_offsets,           // [num_target_mols]
    const double *__restrict__ moved_coords,       // [batch_size, num_atoms_in_each_mol, 3]
    double *__restrict__ dest_coords,              // [N, 3]
    size_t *__restrict__ num_accepted,             // [1],
    int *__restrict__ rand_offset                  // [1]
) {
    // Note that this kernel does not handle multiple proposals, expects that the proposals
    // have been reduced down to a single proposal beforehand.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(idx == 0);

    const int batch_idx = accepted_batched_move[0];
    // If the selected batch idx is not less than the total batch size, no proposal was accepted, we can exit immediately.
    if (batch_idx >= batch_size) {
        rand_offset[0] += batch_size;
        return;
    }
    const int mol_idx = mol_idx_per_batch[batch_idx];
    const int mol_start = mol_offsets[mol_idx];
    // Increment offset by the index + 1, IE the Nth item in the batch being accepted results in incrementing by N + 1
    rand_offset[0] += batch_idx + 1;
    if (idx == 0) {
        num_accepted[0]++;
    }

    // If accepted, move the coords of the selected mol into place
    for (int i = 0; i < num_atoms_in_each_mol; i++) {
        dest_coords[(mol_start + i) * 3 + 0] = moved_coords[num_atoms_in_each_mol * batch_idx * 3 + i * 3 + 0];
        dest_coords[(mol_start + i) * 3 + 1] = moved_coords[num_atoms_in_each_mol * batch_idx * 3 + i * 3 + 1];
        dest_coords[(mol_start + i) * 3 + 2] = moved_coords[num_atoms_in_each_mol * batch_idx * 3 + i * 3 + 2];
    }
}

template <typename RealType>
void __global__ k_attempt_exchange_move_targeted(
    const int num_target_mols,
    const int *__restrict__ targeting_inner_volume,
    const int *__restrict__ inner_count,  // [1]
    const RealType *__restrict__ box_vol, // [1]
    const RealType inner_volume,
    const RealType *__restrict__ rand,           // [1]
    const int *__restrict__ samples,             // [1]
    const RealType *__restrict__ before_max,     // [1]
    const RealType *__restrict__ before_log_sum, // [1]
    const RealType *__restrict__ after_max,      // [1]
    const RealType *__restrict__ after_log_sum,  // [1]
    const int *__restrict__ mol_offsets,         // [num_mols + 1]
    const double *__restrict__ moved_coords,     // [num_atoms_in_each_mol, 3]
    double *__restrict__ dest_coords,            // [N, 3]
    RealType *__restrict__ before_weights,       // [num_target_mols]
    RealType *__restrict__ after_weights,        // [num_target_mols]
    int *__restrict__ inner_flags,               // [num_target_mols]
    size_t *__restrict__ num_accepted            // [1]
) {
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int targeting_inner = targeting_inner_volume[0];

    const RealType outer_vol = box_vol[0] - inner_volume;

    const int local_inner_count = inner_count[0];

    const int moved_mol_idx = samples[0];
    const int mol_start = mol_offsets[moved_mol_idx];
    const int mol_end = mol_offsets[moved_mol_idx + 1];
    const int num_atoms_in_each_mol = mol_end - mol_start;

    const RealType raw_log_acceptance = compute_raw_log_probability_targeted<RealType>(
        targeting_inner,
        inner_volume,
        outer_vol,
        local_inner_count,
        num_target_mols,
        before_max,
        before_log_sum,
        after_max,
        after_log_sum);

    RealType log_acceptance_prob = min(raw_log_acceptance, static_cast<RealType>(0.0));

    const bool accepted = rand[0] < exp(log_acceptance_prob);
    if (atom_idx == 0 && accepted) {
        num_accepted[0]++;
        // XOR 1 to flip the flag from 0 to 1 or 1 to 0
        inner_flags[moved_mol_idx] ^= 1;
    }

    // If accepted, move the coords into place
    // Always copy the weights, either copying from before to after or after to before
    while (atom_idx < num_target_mols || atom_idx < num_atoms_in_each_mol) {
        if (accepted && atom_idx < num_atoms_in_each_mol) {
            dest_coords[(mol_start + atom_idx) * 3 + 0] = moved_coords[atom_idx * 3 + 0];
            dest_coords[(mol_start + atom_idx) * 3 + 1] = moved_coords[atom_idx * 3 + 1];
            dest_coords[(mol_start + atom_idx) * 3 + 2] = moved_coords[atom_idx * 3 + 2];
        }
        // If accepted store the after weights as before weights else copy the before weights to the after weights
        // so the next iteration can incrementally update the weights. The copying of the before to the after is
        // to avoid an additional memcpy kernel.
        if (atom_idx < num_target_mols) {
            if (accepted) {
                before_weights[atom_idx] = after_weights[atom_idx];
            } else {
                after_weights[atom_idx] = before_weights[atom_idx];
            }
        }
        atom_idx += gridDim.x * blockDim.x;
    }
}

template void __global__ k_attempt_exchange_move_targeted<float>(
    const int num_target_mols,
    const int *__restrict__ targeting_inner_volume,
    const int *__restrict__ inner_count,
    const float *__restrict__ box_vol,
    const float inner_volume,
    const float *__restrict__ rand,
    const int *__restrict__ samples,
    const float *__restrict__ before_max,
    const float *__restrict__ before_log_sum,
    const float *__restrict__ after_max,
    const float *__restrict__ after_log_sum,
    const int *__restrict__ mol_offsets,
    const double *__restrict__ moved_coords,
    double *__restrict__ dest_coords,
    float *__restrict__ before_weights,
    float *__restrict__ after_weights,
    int *__restrict__ inner_flags,
    size_t *__restrict__ num_accepted);
template void __global__ k_attempt_exchange_move_targeted<double>(
    const int num_target_mols,
    const int *__restrict__ targeting_inner_volume,
    const int *__restrict__ inner_count,
    const double *__restrict__ box_vol,
    const double inner_volume,
    const double *__restrict__ rand,
    const int *__restrict__ samples,
    const double *__restrict__ before_max,
    const double *__restrict__ before_log_sum,
    const double *__restrict__ after_max,
    const double *__restrict__ after_log_sum,
    const int *__restrict__ mol_offsets,
    const double *__restrict__ moved_coords,
    double *__restrict__ dest_coords,
    double *__restrict__ before_weights,
    double *__restrict__ after_weights,
    int *__restrict__ inner_flags,
    size_t *__restrict__ num_accepted);

template <typename RealType>
void __global__ k_store_accepted_log_probability(
    const int num_weights,
    const int batch_size,
    const int *__restrict__ accepted_batched_move, // [1]
    RealType *__restrict__ before_max,             // [1]
    RealType *__restrict__ before_log_sum,         // [1]
    const RealType *__restrict__ after_max,        // [batch_size]
    const RealType *__restrict__ after_log_sum,    // [batch_size]
    RealType *__restrict__ before_weights,         // [num_weights]
    const RealType *__restrict__ after_weights     // [batch_size, num_weights]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int batch_idx = accepted_batched_move[0];
    // If the selected mol is not less than the total batch size, no proposal was accepted, we can exit immediately.
    if (batch_idx >= batch_size) {
        return;
    }

    // Swap the values after all threads have computed the log probability
    if (idx == 0) {
        before_max[0] = after_max[batch_idx];
        before_log_sum[0] = after_log_sum[batch_idx];
    }
    // Copy over the weights
    while (idx < num_weights) {
        before_weights[idx] = after_weights[batch_idx * num_weights + idx];
        idx += gridDim.x * blockDim.x;
    }
}

template void __global__ k_store_accepted_log_probability<float>(
    const int num_weights,
    const int batch_size,
    const int *__restrict__ accepted_batched_move,
    float *__restrict__ before_max,
    float *__restrict__ before_log_sum,
    const float *__restrict__ after_max,
    const float *__restrict__ after_log_sum,
    float *__restrict__ before_weights,
    const float *__restrict__ after_weights);
template void __global__ k_store_accepted_log_probability<double>(
    const int num_weights,
    const int batch_size,
    const int *__restrict__ accepted_batched_move,
    double *__restrict__ before_max,
    double *__restrict__ before_log_sum,
    const double *__restrict__ after_max,
    const double *__restrict__ after_log_sum,
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
// Does NOT special case the weight of the sampled mol and instead use `k_set_sampled_weight_block`.
template <typename RealType, bool Negated>
void __global__ k_adjust_weights(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_weights,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const RealType *__restrict__ per_atom_energies,
    const RealType inv_kT, // 1 / kT
    RealType *__restrict__ log_weights) {

    int idx_in_batch = blockIdx.y;
    while (idx_in_batch < batch_size) {

        int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (mol_idx < num_weights) {

            RealType current_log_weight = log_weights[idx_in_batch * num_weights + mol_idx];
            __int128 weight_accumulator = 0;

            int mol_start = mol_offsets[mol_idx];
            int mol_end = mol_offsets[mol_idx + 1];
            int min_atom_idx = mol_atoms_idxs[mol_start];
            int max_atom_idx = mol_atoms_idxs[mol_end - 1];

            // A loop that in the case of water will be 3x3
            for (int i = idx_in_batch * mol_size; i < (idx_in_batch + 1) * mol_size; i++) {
                for (int j = min_atom_idx; j <= max_atom_idx; j++) {
                    weight_accumulator += FLOAT_TO_FIXED_ENERGY<RealType>(inv_kT * per_atom_energies[i * N + j]);
                }
            }

            weight_accumulator = Negated ? FLOAT_TO_FIXED_ENERGY<RealType>(current_log_weight) - weight_accumulator
                                         : FLOAT_TO_FIXED_ENERGY<RealType>(current_log_weight) + weight_accumulator;

            log_weights[idx_in_batch * num_weights + mol_idx] =
                fixed_point_overflow(weight_accumulator) ? INFINITY
                                                         : FIXED_ENERGY_TO_FLOAT<RealType>(weight_accumulator);

            mol_idx += gridDim.x * blockDim.x;
        }

        idx_in_batch += gridDim.y * blockDim.y;
    }
}

template void __global__ k_adjust_weights<float, 0>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_weights,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const float *__restrict__ per_atom_energies,
    const float inv_kT,
    float *__restrict__ log_weights);
template void __global__ k_adjust_weights<float, 1>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_weights,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const float *__restrict__ per_atom_energies,
    const float inv_kT,
    float *__restrict__ log_weights);

template void __global__ k_adjust_weights<double, 0>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_weights,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const double *__restrict__ per_atom_energies,
    const double inv_kT,
    double *__restrict__ log_weights);
template void __global__ k_adjust_weights<double, 1>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_weights,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const double *__restrict__ per_atom_energies,
    const double inv_kT,
    double *__restrict__ log_weights);

template <typename RealType, int THREADS_PER_BLOCK>
void __global__ k_set_sampled_weight_block(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_weights,
    const int *__restrict__ target_atoms, // [batch_size, mol_size]
    const RealType *__restrict__ per_atom_energies,
    const RealType inv_kT,                    // 1 / kT
    __int128 *__restrict__ intermediate_accum // [batch_size, ceil_divide(N, THREADS_PER_BLOCK)]
) {
    __shared__ __int128 accumulators[THREADS_PER_BLOCK];

    int idx_in_batch = blockIdx.y;

    while (idx_in_batch < batch_size) {
        int min_atom_idx = target_atoms[idx_in_batch * mol_size];
        int max_atom_idx = target_atoms[(idx_in_batch + 1) * mol_size - 1];

        int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
        // Zero all of the accumulators
        __int128 accumulator = 0;
        while (atom_idx < N) {
            if (atom_idx < min_atom_idx || atom_idx > max_atom_idx) {
                for (int i = idx_in_batch * mol_size; i < (idx_in_batch + 1) * mol_size; i++) {
                    accumulator += FLOAT_TO_FIXED_ENERGY<RealType>(inv_kT * per_atom_energies[i * N + atom_idx]);
                }
            }
            atom_idx += gridDim.x * blockDim.x;
        }
        accumulators[threadIdx.x] = accumulator;
        __syncthreads();
        block_energy_reduce<THREADS_PER_BLOCK>(accumulators, threadIdx.x);
        if (threadIdx.x == 0) {
            intermediate_accum[idx_in_batch * gridDim.x + blockIdx.x] = accumulators[0];
        }
        idx_in_batch += gridDim.y * blockDim.y;
    }
}

template void __global__ k_set_sampled_weight_block<float, 512>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_weights,
    const int *__restrict__ target_atoms,
    const float *__restrict__ per_atom_energies,
    const float inv_kT,
    __int128 *__restrict__ intermediate_accum);
template void __global__ k_set_sampled_weight_block<double, 512>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_weights,
    const int *__restrict__ target_atoms,
    const double *__restrict__ per_atom_energies,
    const double inv_kT,
    __int128 *__restrict__ intermediate_accum);

template <typename RealType, int THREADS_PER_BLOCK>
void __global__ k_set_sampled_weight_reduce(
    const int batch_size,
    const int num_weights,
    const int num_intermediates,
    const int *__restrict__ samples,                 // [batch_size]
    const __int128 *__restrict__ intermediate_accum, // [batch_size, num_intermediates]
    RealType *__restrict__ log_weights               // [batch_size, num_weights]
) {
    __shared__ __int128 accumulators[THREADS_PER_BLOCK];

    // One y block per set of weights, used instead of x block to avoid nuance of setting idx based only on
    // the thread idx
    int idx_in_batch = blockIdx.y;

    while (idx_in_batch < batch_size) {
        int offset = idx_in_batch * num_intermediates;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Zero all of the accumulators
        __int128 accumulator = 0;
        while (idx < num_intermediates) {
            accumulator += intermediate_accum[offset + idx];
            idx += gridDim.x * blockDim.x;
        }
        // Each block reduces on a specific set of weights which is why just threadIdx.x
        accumulators[threadIdx.x] = accumulator;
        __syncthreads();
        block_energy_reduce<THREADS_PER_BLOCK>(accumulators, threadIdx.x);
        if (threadIdx.x == 0) {
            int mol_idx = samples[idx_in_batch];
            log_weights[idx_in_batch * num_weights + mol_idx] =
                fixed_point_overflow(accumulators[0]) ? INFINITY : FIXED_ENERGY_TO_FLOAT<RealType>(accumulators[0]);
        }

        idx_in_batch += gridDim.y * blockDim.y;
    }
}

template void __global__ k_set_sampled_weight_reduce<float, 512>(
    const int batch_size,
    const int num_intermediates,
    const int num_weights,
    const int *__restrict__ samples,                 // [batch_size]
    const __int128 *__restrict__ intermediate_accum, // [batch_size, num_intermediates]
    float *__restrict__ log_weights                  // [batch_size, num_weights]
);
template void __global__ k_set_sampled_weight_reduce<double, 512>(
    const int batch_size,
    const int num_intermediates,
    const int num_weights,
    const int *__restrict__ samples,                 // [batch_size]
    const __int128 *__restrict__ intermediate_accum, // [batch_size, num_intermediates]
    double *__restrict__ log_weights                 // [batch_size, num_weights]
);

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
void __global__ k_decide_targeted_moves(
    const int batch_size,
    const int num_target_mols,
    const RealType *__restrict__ rand,         // [batch_size]
    const int *__restrict__ inner_count,       // [1]
    const RealType *__restrict__ translations, // [batch_size, 2, 3] first translation is inside, second is outer
    int *__restrict__ targeting_inner_volume,  // [batch_size]
    int *__restrict__ src_weights_counts,      // [batch_size]
    int *__restrict__ target_weights_counts,   // [batch_size]
    RealType *__restrict__ output_translation  // [batch_size, 3]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int count_inside = inner_count[0];
    const int count_outside = num_target_mols - count_inside;
    int flag;
    while (idx < batch_size) {
        if (count_inside == 0 && count_outside == 0) {
            assert(0);
        } else if (count_inside > 0 && count_outside == 0) {
            flag = 0;
        } else if (count_inside == 0 && count_outside > 0) {
            flag = 1;

        } else if (count_inside > 0 && count_outside > 0) {
            // TBD determine if accessing rand in if matters
            if (rand[idx] < static_cast<RealType>(0.5)) {
                flag = 1;
            } else {
                flag = 0;
            }
        } else {
            assert(0);
        }
        output_translation[idx * 3 + 0] =
            flag == 1 ? translations[idx * (3 * 2) + 0] : translations[idx * (3 * 2) + 3 + 0];
        output_translation[idx * 3 + 1] =
            flag == 1 ? translations[idx * (3 * 2) + 1] : translations[idx * (3 * 2) + 3 + 1];
        output_translation[idx * 3 + 2] =
            flag == 1 ? translations[idx * (3 * 2) + 2] : translations[idx * (3 * 2) + 3 + 2];

        // Will look at the src weights
        src_weights_counts[idx] = flag == 1 ? count_outside : count_inside;
        // The target weights will be the count of the target region + 1, the new mol being inserted
        target_weights_counts[idx] = flag == 1 ? count_inside + 1 : count_outside + 1;
        targeting_inner_volume[idx] = flag;
        idx += gridDim.x * blockDim.x;
    }
}

template void __global__ k_decide_targeted_moves<float>(
    const int batch_size,
    const int num_target_mols,
    const float *__restrict__ rand,
    const int *__restrict__ inner_count,
    const float *__restrict__ translations,
    int *__restrict__ targeting_inner_volume,
    int *__restrict__ src_weights_counts,
    int *__restrict__ target_weights_counts,
    float *__restrict__ output_translation);
template void __global__ k_decide_targeted_moves<double>(
    const int batch_size,
    const int num_target_mols,
    const double *__restrict__ rand,
    const int *__restrict__ inner_count,
    const double *__restrict__ translations,
    int *__restrict__ targeting_inner_volume,
    int *__restrict__ src_weights_counts,
    int *__restrict__ target_weights_counts,
    double *__restrict__ output_translation);

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

void __global__ k_adjust_sample_idxs(
    const int batch_size,
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    int *__restrict__ sample_idxs                   // [batch_size]
) {
    const int local_inner_count = inner_count[0];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // At the moment we only have one sample
    while (idx < batch_size) {
        const int target_inner = targeting_inner_volume[idx];
        const int offset = target_inner == 1 ? local_inner_count : 0;
        sample_idxs[idx] = partitioned_indices[sample_idxs[idx] + offset];

        idx += gridDim.x * blockDim.x;
    }
}

// k_accept_first_valid_move selects the first sample within a batch of samples. Use AtomicMin to
// find the lowest index sample that was selected, then store that sample in accepted sample field. If no sample is
// accepted store N in the accepted sample, indicating no valid sample. The accepted sample can also be used as the offset
// for shifting the noise for backtracking
template <typename RealType>
void __global__ k_accept_first_valid_move(
    const int total_proposals,
    const int num_target_mols,
    const int batch_size,
    const int *__restrict__ noise_offset,        // [1]
    const int *__restrict__ samples,             // [batch_size]
    const RealType *__restrict__ before_max,     // [1]
    const RealType *__restrict__ before_log_sum, // [1]
    const RealType *__restrict__ after_max,      // [batch_size]
    const RealType *__restrict__ after_log_sum,  // [batch_size]
    const RealType *__restrict__ rand,           // [total_proposals]
    int *__restrict__ accepted_sample            // [1]
) {
    __shared__ int selected_idx;
    assert(blockIdx.x == 0);
    if (threadIdx.x == 0) {
        selected_idx = batch_size;
    }
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const RealType before_log_prob =
        convert_nan_to_inf<RealType>(compute_logsumexp_final<RealType>(before_max[0], before_log_sum[0]));
    const int rand_offset = noise_offset[0];
    while (idx < batch_size) {
        if (rand_offset + idx >= total_proposals) {
            break;
        }

        const RealType after_log_prob =
            convert_nan_to_inf<RealType>(compute_logsumexp_final<RealType>(after_max[idx], after_log_sum[idx]));

        const RealType log_acceptance_prob = min(before_log_prob - after_log_prob, static_cast<RealType>(0.0));
        const bool accepted = rand[rand_offset + idx] < exp(log_acceptance_prob);
        if (accepted) {
            atomicMin(&selected_idx, idx);
            // Idx is increasing so the first accepted is the min value the thread can accept
            // making it safe to break early.
            break;
        }

        idx += gridDim.x * blockDim.x;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        accepted_sample[0] = selected_idx;
    }
}

template void __global__ k_accept_first_valid_move<float>(
    const int total_proposals,
    const int num_target_mols,
    const int batch_size,
    const int *__restrict__ rand_offset,      // [1]
    const int *__restrict__ samples,          // [batch_size]
    const float *__restrict__ before_max,     // [1]
    const float *__restrict__ before_log_sum, // [1]
    const float *__restrict__ after_max,      // [batch_size]
    const float *__restrict__ after_log_sum,  // [batch_size]
    const float *__restrict__ rand,           // [total_proposals]
    int *__restrict__ accepted_sample         // [1]
);

template void __global__ k_accept_first_valid_move<double>(
    const int total_proposals,
    const int num_target_mols,
    const int batch_size,
    const int *__restrict__ rand_offset,       // [1]
    const int *__restrict__ samples,           // [batch_size]
    const double *__restrict__ before_max,     // [1]
    const double *__restrict__ before_log_sum, // [1]
    const double *__restrict__ after_max,      // [batch_size]
    const double *__restrict__ after_log_sum,  // [batch_size]
    const double *__restrict__ rand,           // [total_proposals]
    int *__restrict__ accepted_sample          // [1]
);

} // namespace timemachine
