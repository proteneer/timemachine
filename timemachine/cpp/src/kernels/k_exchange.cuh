#pragma once

namespace timemachine {

// When we are considering exchange we want to treat Nan probabilities as inf
// Allows us to go from a clashy state to a non-clashy state. And no nan poisoning
template <typename RealType> RealType __host__ __device__ convert_nan_to_inf(const RealType input);

void __global__ k_setup_sample_atoms(
    const int sample_atoms,          // number of atoms in each sample
    const int *__restrict__ samples, // [1]
    const int *__restrict__ target_atoms,
    const int *__restrict__ mol_offsets,
    int *__restrict__ output_atom_idxs,
    int *__restrict__ output_mol_offsets);

template <typename RealType>
void __global__ k_attempt_exchange_move(
    const int N,
    const RealType *__restrict__ rand,               // [1]
    const RealType *__restrict__ before_log_sum_exp, // [2]
    const RealType *__restrict__ after_log_sum_exp,  // [2]
    const double *__restrict__ moved_coords,         // [N, 3]
    double *__restrict__ dest_coords,                // [N, 3]
    size_t *__restrict__ num_accepted                // [1]
);

template <typename RealType>
void __global__ k_attempt_exchange_move_targeted(
    const int N,
    const int *__restrict__ targeting_inner_volume,
    const RealType *__restrict__ box_vol,
    const RealType inner_volume,
    const RealType *__restrict__ rand,               // [1]
    const RealType *__restrict__ before_log_sum_exp, // [2]
    const RealType *__restrict__ after_log_sum_exp,  // [2]
    const double *__restrict__ moved_coords,         // [N, 3]
    double *__restrict__ dest_coords,                // [N, 3]
    size_t *__restrict__ num_accepted                // [1]
);

template <typename RealType>
void __global__ k_store_accepted_log_probability(
    const int num_weights,
    const RealType *__restrict__ rand,              // [1]
    RealType *__restrict__ before_log_sum_exp,      // [2]
    const RealType *__restrict__ after_log_sum_exp, // [2]
    RealType *__restrict__ before_weights,          // [num_weights]
    const RealType *__restrict__ after_weights      // [num_weights]
);

template <typename RealType>
void __global__ k_store_accepted_log_probability_targeted(
    const int num_weights,
    const int *__restrict__ targeting_inner_volume,
    const RealType *__restrict__ box_vol,
    const RealType inner_volume,
    const RealType *__restrict__ rand,              // [1]
    RealType *__restrict__ before_log_sum_exp,      // [2]
    const RealType *__restrict__ after_log_sum_exp, // [2]
    RealType *__restrict__ before_weights,          // [num_weights]
    const RealType *__restrict__ after_weights      // [num_weights]
);

template <typename RealType>
void __global__ k_compute_box_volume(const double *__restrict__ box, RealType *__restrict__ output_volume);

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
    RealType *__restrict__ log_weights);

template <typename RealType, int THREADS_PER_BLOCK>
void __global__ k_set_sampled_weight(
    const int N,
    const int mol_size,
    const int *__restrict__ samples, // [1]
    const int *__restrict__ target_atoms,
    const int *__restrict__ mol_offsets,
    const RealType *__restrict__ per_atom_energies,
    const RealType inv_kT, // 1 / kT
    RealType *__restrict__ log_weights);

template <typename RealType>
void __global__ k_compute_centroid_of_atoms(
    const int num_atoms,
    const int *__restrict__ atom_idxs, // [num_atoms]
    const double *__restrict__ coords, // [N, 3]
    RealType *__restrict__ centroid    // [3]
);

template <typename RealType>
void __global__ k_split_mols_inner_outer(
    const int num_molecules,
    const int *__restrict__ atom_idxs,
    const int *__restrict__ mol_offsets, // [num_molecules + 1]
    const RealType *__restrict__ center, // [3] Center that determines inner vs outer
    const RealType square_radius,        // squared radius from center that defines inner
    const double *__restrict__ coords,   // [N, 3]
    const double *__restrict__ box,      // [3, 3]
    int *__restrict__ inner_count,       // [1]
    int *__restrict__ inner_mols,        // [num_molecules]
    int *__restrict__ outer_count,       // [1]
    int *__restrict__ outer_mols         // [num_molecules]
);

template <typename RealType>
void __global__ k_decide_targeted_move(
    const RealType *__restrict__ rand,
    const int *__restrict__ inner_count,
    const int *__restrict__ outer_count,
    int *__restrict__ targeting_inner_volume);

template <typename RealType>
void __global__ k_separate_weights_for_targeted(
    const int num_target_mols,
    const int *__restrict__ targeting_inner_volume, // [1]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ outer_count,            // [1]
    const int *__restrict__ inner_idxs,             // [inner_count]
    const int *__restrict__ outer_idxs,             // [outer_count]
    const RealType *__restrict__ weights,           // [num_target_mols]
    RealType *__restrict__ output_weights);

template <typename RealType>
void __global__ k_setup_destination_weights_for_targeted(
    const int num_target_mols,
    const int *__restrict__ samples,                // [1]
    const int *__restrict__ targeting_inner_volume, // [1]
    const int *__restrict__ inner_count,            // [1]
    const int *__restrict__ outer_count,            // [1]
    const int *__restrict__ inner_idxs,             // [inner_count]
    const int *__restrict__ outer_idxs,             // [outer_count]
    const RealType *__restrict__ weights,           // [num_target_mols]
    RealType *__restrict__ output_weights);

void __global__ k_adjust_sample_idx(const int *__restrict__ mol_indices, int *__restrict__ sample_idxs);

} // namespace timemachine
