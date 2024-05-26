#pragma once

#include "k_logsumexp.cuh"
#include <assert.h>

namespace timemachine {

// When we are considering exchange we want to treat Nan probabilities as inf
// Allows us to go from a clashy state to a non-clashy state. And no nan poisoning
// When we are considering exchange we want to treat Nan probabilities as inf
// Allows us to go from a clashy state to a non-clashy state. And no nan poisoning
template <typename RealType> RealType __host__ __device__ __forceinline__ convert_nan_to_inf(const RealType input) {
    return isnan(input) ? INFINITY : input;
}

template <typename T>
void __global__ k_copy_batch(const int N, const int batch_size, const T *__restrict__ src, T *__restrict__ dest);

template <typename RealType>
RealType __host__ __device__ __forceinline__
compute_log_proposal_probabilities_given_counts(const int src_count, const int dest_count) {
    if (src_count > 0 && dest_count > 0) {
        return static_cast<RealType>(log(0.5));
    } else if (src_count > 0 && dest_count == 0) {
        return static_cast<RealType>(log(1.0));
    } else if (src_count == 0 && dest_count > 0) {
        return static_cast<RealType>(log(1.0));
    }
    // Invalid case
    assert(0);
    return 0.0; // Here to ensure the code compiles, assertion will trigger failure
}

template <typename RealType>
void __global__ k_convert_energies_to_log_weights(
    const int N, const RealType inv_beta, const __int128 *__restrict__ energies, RealType *__restrict__ log_weights);

template <typename RealType>
RealType __host__ __device__ __forceinline__ compute_raw_log_probability_targeted(
    const int targeting_inner_volume, // 1 or 0
    const RealType inner_volume,
    const RealType outer_vol,
    const int inner_count,
    const int num_target_mols,
    const RealType *__restrict__ before_max,     // [1]
    const RealType *__restrict__ before_log_sum, // [1]
    const RealType *__restrict__ after_max,      // [1]
    const RealType *__restrict__ after_log_sum   // [1]
) {
    const RealType log_vol_prob =
        targeting_inner_volume == 1 ? log(inner_volume) - log(outer_vol) : log(outer_vol) - log(inner_volume);

    // Account for the proposal probability, in the case of there being 0 or 1 mol in a volume
    // the probability will not be symmetric.
    int src_count = targeting_inner_volume == 1 ? num_target_mols - inner_count : inner_count;
    int dest_count = targeting_inner_volume == 1 ? inner_count : num_target_mols - inner_count;
    const RealType log_fwd_prob = compute_log_proposal_probabilities_given_counts<RealType>(src_count, dest_count);
    const RealType log_rev_prob =
        compute_log_proposal_probabilities_given_counts<RealType>(src_count - 1, dest_count + 1);

    RealType before_log_prob =
        convert_nan_to_inf<RealType>(compute_logsumexp_final<RealType>(before_max[0], before_log_sum[0]));
    RealType after_log_prob =
        convert_nan_to_inf<RealType>(compute_logsumexp_final<RealType>(after_max[0], after_log_sum[0]));

    return before_log_prob - after_log_prob + log_vol_prob + (log_rev_prob - log_fwd_prob);
}

void __global__ k_setup_proposals(
    const int total_proposals,
    const int batch_size,            // Number of molecules to setup
    const int num_atoms_in_each_mol, // number of atoms in each sample
    const int *__restrict__ rand_offset,
    const int *__restrict__ mol_idx_per_batch, // [batch_size] The index of the molecules to sample
    const int *__restrict__ atom_indices,      // [N]
    const int *__restrict__ mol_offsets,       // [num_target_mols]
    int *__restrict__ output_atom_idxs,        // [batch_size, num_atoms_in_each_mol]
    int *__restrict__ output_mol_offsets);

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
);

template <typename RealType>
void __global__ k_store_accepted_log_probability(
    const int num_energies,
    const int batch_size,
    const int *__restrict__ accepted_batched_move, // [1]
    RealType *__restrict__ before_max,             // [1]
    RealType *__restrict__ before_log_sum,         // [1]
    const RealType *__restrict__ after_max,        // [batch_size]
    const RealType *__restrict__ after_log_sum     // [batch_size]
);

template <typename RealType>
void __global__ k_compute_box_volume(const double *__restrict__ box, RealType *__restrict__ output_volume);

// k_adjust_energies takes a set of molecules and either subtracts (Negated=true) or adds (Negated=false)
// the sum of the per atom energies for the molecules from some initial energies.
// This is used to do the transposition trick where we subtract off the weight contribution of the
// moved atom followed by adding back in the weight of the sampled mol in the new position.
// Does NOT special case the weight of the sampled mol and instead use `k_set_sampled_energy`.
template <typename RealType, bool Negated>
void __global__ k_adjust_energies(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_energies,
    const int *__restrict__ mol_atoms_idxs,
    const int *__restrict__ mol_offsets,
    const RealType *__restrict__ per_atom_energies,
    __int128 *__restrict__ mol_energies);

template <typename RealType, int THREADS_PER_BLOCK>
void __global__ k_set_sampled_energy_block(
    const int N,
    const int batch_size,
    const int mol_size,
    const int num_energies,
    const int *__restrict__ target_atoms,
    const RealType *__restrict__ per_atom_energies,
    __int128 *__restrict__ intermediate_accum);

template <int THREADS_PER_BLOCK>
void __global__ k_set_sampled_energy_reduce(
    const int batch_size,
    const int num_energies,
    const int num_intermediates,
    const int *__restrict__ samples, // [batch_size]
    const __int128 *__restrict__ intermediate_accum,
    __int128 *__restrict__ mol_energies);

template <typename RealType>
void __global__ k_compute_centroid_of_atoms(
    const int num_atoms,
    const int *__restrict__ atom_idxs, // [num_atoms]
    const double *__restrict__ coords, // [N, 3]
    RealType *__restrict__ centroid    // [3]
);

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
);

template <typename RealType>
void __global__ k_decide_targeted_moves(
    const int total_proposals,
    const int batch_size,
    const int num_target_mols,
    const int *__restrict__ noise_offset,
    const RealType *__restrict__ rand,         // [batch_size]
    const int *__restrict__ inner_count,       // [1]
    const RealType *__restrict__ translations, // [batch_size, 2, 3] first translation is inside, second is outer
    int *__restrict__ targeting_inner_volume,  // [batch_size]
    int *__restrict__ src_weights_counts,
    int *__restrict__ target_weights_counts,
    RealType *__restrict__ output_translation // [batch_size, 3]
);

template <typename RealType>
void __global__ k_separate_weights_for_targeted(
    const int batch_size,
    const int num_target_mols,
    const int *__restrict__ weight_offsets,         // [batch_size + 1]
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [batch_size]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    const RealType *__restrict__ weights,           // [num_target_mols]
    RealType *__restrict__ output_weights);

template <typename RealType>
void __global__ k_setup_destination_weights_for_targeted(
    const int batch_size,
    const int num_target_mols,
    const int *__restrict__ samples,                // [1]
    const int *__restrict__ weight_offsets,         // [batch_size + 1]
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    const RealType *__restrict__ weights,           // [num_target_mols]
    RealType *__restrict__ output_weights);

void __global__ k_adjust_sample_idxs(
    const int batch_size,
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ partitioned_indices,    // [inner_count]
    int *__restrict__ sample_idxs                   // [batch_size]
);

template <typename RealType>
void __global__ k_accept_first_valid_move(
    const int total_proposals,
    const int num_target_mols,
    const int batch_size,
    const int *__restrict__ rand_offset,         // [1]
    const int *__restrict__ samples,             // [batch_size]
    const RealType *__restrict__ before_max,     // [1]
    const RealType *__restrict__ before_log_sum, // [1]
    const RealType *__restrict__ after_max,      // [batch_size]
    const RealType *__restrict__ after_log_sum,  // [batch_size]
    const RealType *__restrict__ rand,           // [total_proposals]
    int *__restrict__ accepted_sample            // [1]
);

template <typename RealType>
void __global__ k_accept_first_valid_move_targeted(
    const int total_proposals,
    const int num_target_mols,
    const int batch_size,
    const RealType inner_volume,
    const int *__restrict__ targeting_inner_volume, // [batch_size]
    const int *__restrict__ inner_count,            // [1]
    const RealType *__restrict__ box_vol,           // [1]
    const int *__restrict__ noise_offset,           // [1]
    const int *__restrict__ samples,                // [batch_size]
    const RealType *__restrict__ before_max,        // [batch_size]
    const RealType *__restrict__ before_log_sum,    // [batch_size]
    const RealType *__restrict__ after_max,         // [batch_size]
    const RealType *__restrict__ after_log_sum,     // [batch_size]
    const RealType *__restrict__ rand,              // [total_proposals]
    int *__restrict__ accepted_sample               // [1]
);

} // namespace timemachine
