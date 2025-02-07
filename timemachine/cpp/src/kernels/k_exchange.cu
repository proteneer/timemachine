
#include "../gpu_utils.cuh"
#include "k_exchange.cuh"

namespace timemachine {

template void __global__
k_copy_batch<float>(const int N, const int batch_size, const float *__restrict__ src, float *__restrict__ dest);

template void __global__
k_copy_batch<double>(const int N, const int batch_size, const double *__restrict__ src, double *__restrict__ dest);

template void __global__ k_copy_batch<__int128>(
    const int N, const int batch_size, const __int128 *__restrict__ src, __int128 *__restrict__ dest);

template void __global__ k_convert_energies_to_log_weights<float>(
    const int N, const float inv_beta, const __int128 *__restrict__ energies, float *__restrict__ log_weights);
template void __global__ k_convert_energies_to_log_weights<double>(
    const int N, const double inv_beta, const __int128 *__restrict__ energies, double *__restrict__ log_weights);

// k_setup_proposals takes a set of sampled indices and constructs buffers containing the molecule offsets
// as well as the atom indices (refer to src/mol_utils.hpp for impl) and setups up the offsets and the atom
// indices for each sample. Note that the output mol offsets are constructed so that the start of the mol is
// the starting atom idx rather than the prefix sum of mol lengths that the mol_offsets is.
void __global__ k_setup_proposals(
    const int total_proposals,
    const int batch_size,                      // Number of proposals to setup
    const int num_atoms_in_each_mol,           // number of atoms in each sample
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

void __global__ k_store_exchange_move(
    const int batch_size,
    const int num_target_mols,
    const int *__restrict__ accepted_batched_move, // [1]
    const int *__restrict__ mol_idx_per_batch,     // [batch_size]
    const int *__restrict__ mol_offsets,           // [num_mols + 1]
    const int *__restrict__ segment_offsets,       // [batch_size + 1]
    const double *__restrict__ moved_coords,       // [num_atoms_in_each_mol, 3]
    double *__restrict__ dest_coords,              // [N, 3]
    __int128 *__restrict__ before_energies,        // [num_target_mols]
    __int128 *__restrict__ after_energies,         // [batch_size, num_target_mols]
    int *__restrict__ rand_offset,                 // [1]
    int *__restrict__ inner_flags,                 // [num_target_mols] or nullptr
    size_t *__restrict__ num_accepted              // [1]
) {
    // Note that this kernel does not handle multiple proposals, expects that the proposals
    // have been reduced down to a single proposal beforehand.
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int batch_idx = accepted_batched_move[0];
    // If the selected batch idx is not less than the total batch size, no proposal was accepted, we can exit immediately.
    const bool accepted = batch_idx < batch_size;

    const int mol_idx = accepted ? mol_idx_per_batch[batch_idx] : 0;
    const int mol_start = accepted ? mol_offsets[mol_idx] : 0;
    const int mol_end = accepted ? mol_offsets[mol_idx + 1] : 0;
    const int num_atoms_in_mol = mol_end - mol_start;

    // Increment offset by the index + 1, IE the Nth item in the batch being accepted results in incrementing by N + 1
    if (atom_idx == 0) {
        if (accepted) {
            rand_offset[0] += batch_idx + 1;
            num_accepted[0]++;
            // When not using targeted, this will be null
            if (inner_flags != nullptr) {
                // XOR 1 to flip the flag from 0 to 1 or 1 to 0
                inner_flags[mol_idx] ^= 1;
            }
        } else {
            rand_offset[0] += batch_size;
        }
    }

    // Need to reset all of the before and source energies
    // either the after energies to the before energies or the accepted batches energies to the before and the
    // other after energies
    const int energies_copy_count = num_target_mols * batch_size;

    // If accepted, move the coords into place
    // Always copy the energies, either copying from before to after or after to before
    while (atom_idx < energies_copy_count || atom_idx < num_atoms_in_mol) {
        if (accepted && atom_idx < num_atoms_in_mol) {
            dest_coords[(mol_start + atom_idx) * 3 + 0] =
                moved_coords[num_atoms_in_mol * batch_idx * 3 + atom_idx * 3 + 0];
            dest_coords[(mol_start + atom_idx) * 3 + 1] =
                moved_coords[num_atoms_in_mol * batch_idx * 3 + atom_idx * 3 + 1];
            dest_coords[(mol_start + atom_idx) * 3 + 2] =
                moved_coords[num_atoms_in_mol * batch_idx * 3 + atom_idx * 3 + 2];
        }
        // At the end of batch of proposals we need to update the before and after energies to the correct state.
        // In the case of not accepting any moves we want to reset the after energies to be the before energies so that
        // kernels can update the after energies.
        // In the case of accepting a move the before energies need to be updated with the after energies associated with
        //  the accepted batch. The accepted energies also need to be copied into the other after energy batches so that
        // the next step can accumulate energies correctly.
        // We will use `k_convert_energies_to_log_weights` to generate the correct weights separately.
        if (atom_idx < energies_copy_count) {
            if (accepted) {
                // Before energies is only num_target_mols long
                if (atom_idx < num_target_mols) {
                    before_energies[atom_idx] = after_energies[batch_idx * num_target_mols + atom_idx];
                }
                if (atom_idx != batch_idx * num_target_mols + (atom_idx % num_target_mols)) {
                    after_energies[atom_idx] =
                        after_energies[batch_idx * num_target_mols + (atom_idx % num_target_mols)];
                }
            } else {
                after_energies[atom_idx] = before_energies[atom_idx % num_target_mols];
            }
        }
        atom_idx += gridDim.x * blockDim.x;
    }
}

template void __global__ k_store_accepted_log_probability<float>(
    const int num_energies,
    const int batch_size,
    const int *__restrict__ accepted_batched_move,
    float *__restrict__ before_max,
    float *__restrict__ before_log_sum,
    const float *__restrict__ after_max,
    const float *__restrict__ after_log_sum);
template void __global__ k_store_accepted_log_probability<double>(
    const int num_energies,
    const int batch_size,
    const int *__restrict__ accepted_batched_move,
    double *__restrict__ before_max,
    double *__restrict__ before_log_sum,
    const double *__restrict__ after_max,
    const double *__restrict__ after_log_sum);

template void __global__ k_compute_box_volume<float>(
    const double *__restrict__ box,    // [3]
    float *__restrict__ output_volume  // [1]
);
template void __global__ k_compute_box_volume<double>(
    const double *__restrict__ box,    // [3]
    double *__restrict__ output_volume // [1]
);

template void __global__ k_adjust_energies<float, 0>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_energies,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const float *__restrict__ per_atom_energies,
    __int128 *__restrict__ mol_energies);
template void __global__ k_adjust_energies<float, 1>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_energies,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const float *__restrict__ per_atom_energies,
    __int128 *__restrict__ mol_energies);
template void __global__ k_adjust_energies<double, 0>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_energies,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const double *__restrict__ per_atom_energies,
    __int128 *__restrict__ mol_energies);
template void __global__ k_adjust_energies<double, 1>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_energies,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const double *__restrict__ per_atom_energies,
    __int128 *__restrict__ mol_energies);

template void __global__ k_set_sampled_energy_block<float, 512>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_energies,
    const int *__restrict__ target_atoms,
    const float *__restrict__ per_atom_energies,
    __int128 *__restrict__ intermediate_accum);

template void __global__ k_set_sampled_energy_block<double, 512>(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_energies,
    const int *__restrict__ target_atoms,
    const double *__restrict__ per_atom_energies,
    __int128 *__restrict__ intermediate_accum);

template void __global__ k_set_sampled_energy_reduce<512>(
    const int batch_size,
    const int num_intermediates,
    const int num_energies,
    const int *__restrict__ samples,                 // [batch_size]
    const __int128 *__restrict__ intermediate_accum, // [batch_size, num_intermediates]
    __int128 *__restrict__ mol_energies              // [batch_size, num_energies]
);

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

template void __global__ k_decide_targeted_moves<float>(
    const int total_proposals,
    const int batch_size,
    const int num_target_mols,
    const int *__restrict__ noise_offset,
    const float *__restrict__ rand,
    const int *__restrict__ inner_count,
    const float *__restrict__ translations,
    int *__restrict__ targeting_inner_volume,
    int *__restrict__ src_weights_counts,
    int *__restrict__ target_weights_counts,
    float *__restrict__ output_translation);
template void __global__ k_decide_targeted_moves<double>(
    const int total_proposals,
    const int batch_size,
    const int num_target_mols,
    const int *__restrict__ noise_offset,
    const double *__restrict__ rand,
    const int *__restrict__ inner_count,
    const double *__restrict__ translations,
    int *__restrict__ targeting_inner_volume,
    int *__restrict__ src_weights_counts,
    int *__restrict__ target_weights_counts,
    double *__restrict__ output_translation);

template void __global__ k_separate_weights_for_targeted<float>(
    const int batch_size,
    const int num_target_mols,
    const int *__restrict__ weight_offsets,         // [batch_size + 1]
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [num_target_mols]
    const float *__restrict__ weights,              // [num_target_mols]
    float *__restrict__ output_weights);
template void __global__ k_separate_weights_for_targeted<double>(
    const int batch_size,
    const int num_target_mols,
    const int *__restrict__ weight_offsets,         // [batch_size + 1]
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [num_target_mols]
    const double *__restrict__ weights,             // [num_target_mols]
    double *__restrict__ output_weights);

template void __global__ k_setup_destination_weights_for_targeted<float>(
    const int total_proposals,
    const int batch_size,
    const int num_target_mols,
    const int *__restrict__ noise_offset,           // [1]
    const int *__restrict__ samples,                // [batch_size]
    const int *__restrict__ weight_offsets,         // [batch_size + 1]
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [num_target_mols]
    const float *__restrict__ weights,              // [num_target_mols]
    float *__restrict__ output_weights              // [batch_size, num_target_mols]
);

template void __global__ k_setup_destination_weights_for_targeted<double>(
    const int total_proposals,
    const int batch_size,
    const int num_target_mols,
    const int *__restrict__ noise_offset,           // [1]
    const int *__restrict__ samples,                // [batch_size]
    const int *__restrict__ weight_offsets,         // [batch_size + 1]
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [num_target_mols]
    const double *__restrict__ weights,             // [num_target_mols]
    double *__restrict__ output_weights             // [batch_size, num_target_mols]
);

void __global__ k_adjust_sample_idxs(
    const int total_proposals,
    const int batch_size,
    const int *__restrict__ noise_offset,           // [1]
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [num_mols] Total number of molecules being sampled
    int *__restrict__ sample_idxs                   // [batch_size]
) {
    const int num_prev_proposals = *noise_offset;
    const int local_inner_count = *inner_count;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only evaluate up to idx + num_prev_proposals < total proposals. The samples beyond this point
    // contain stale data due to k_decide_targeted_moves not constructing them. All downstream kernels will
    // use stale samples, but no proposals past the total number proposals will be evaluated in k_accept_first_valid_move_targeted
    while (idx < batch_size && idx + num_prev_proposals < total_proposals) {
        const int target_inner = targeting_inner_volume[idx];
        const int offset = target_inner == 1 ? local_inner_count : 0;
        const int before = sample_idxs[idx];
        sample_idxs[idx] = partitioned_indices[before + offset];
        idx += gridDim.x * blockDim.x;
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

template void __global__ k_accept_first_valid_move_targeted<float>(
    const int total_proposals,
    const int num_target_mols,
    const int batch_size,
    const float inner_volume,
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const float *__restrict__ box_vol,              // [1]
    const int *__restrict__ noise_offset,           // [1]
    const int *__restrict__ samples,                // [batch_size]
    const float *__restrict__ before_max,           // [1]
    const float *__restrict__ before_log_sum,       // [1]
    const float *__restrict__ after_max,            // [batch_size]
    const float *__restrict__ after_log_sum,        // [batch_size]
    const float *__restrict__ rand,                 // [total_proposals]
    int *__restrict__ accepted_sample               // [1]
);

template void __global__ k_accept_first_valid_move_targeted<double>(
    const int total_proposals,
    const int num_target_mols,
    const int batch_size,
    const double inner_volume,
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const double *__restrict__ box_vol,             // [1]
    const int *__restrict__ noise_offset,           // [1]
    const int *__restrict__ samples,                // [batch_size]
    const double *__restrict__ before_max,          // [1]
    const double *__restrict__ before_log_sum,      // [1]
    const double *__restrict__ after_max,           // [batch_size]
    const double *__restrict__ after_log_sum,       // [batch_size]
    const double *__restrict__ rand,                // [total_proposals]
    int *__restrict__ accepted_sample               // [1]
);

} // namespace timemachine
