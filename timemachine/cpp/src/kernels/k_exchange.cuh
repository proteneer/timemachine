#include "k_logsumexp.cuh"
#include "k_rotations.cuh"

namespace timemachine {

template <typename RealType>
void __global__ k_attempt_exchange_move(
    const int N,
    const RealType *__restrict__ rand,               // [4] last value is of interest
    const RealType *__restrict__ before_log_sum_exp, // [2]
    const RealType *__restrict__ after_log_sum_exp,  // [2]
    const double *__restrict__ moved_coords,
    double *__restrict__ dest_coords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // All kernels compute the same acceptance
    // TBD investigate shared memory for speed
    RealType log_acceptance_prob = min(
        compute_logsumexp_final<RealType>(before_log_sum_exp) - compute_logsumexp_final<RealType>(after_log_sum_exp),
        static_cast<RealType>(0.0));
    const bool accepted = rand[3] < exp(log_acceptance_prob);
    // If accepted, move the coords into place
    while (accepted && idx < N) {
        dest_coords[idx * 3 + 0] = moved_coords[idx * 3 + 0];
        dest_coords[idx * 3 + 1] = moved_coords[idx * 3 + 1];
        dest_coords[idx * 3 + 2] = moved_coords[idx * 3 + 2];
        idx += gridDim.x * blockDim.x;
    }
}

template <typename RealType>
void __global__ k_rotate_and_translate_mols(
    const int num_samples,
    const double *__restrict__ coords, // [N, 3]
    const double *__restrict__ box,    // [3, 3]
    const int *__restrict__ samples,   // [num_samples]
    const int *__restrict__ mol_offsets,
    const RealType *__restrict__ quaternions,  // [num_samples, 4]
    const RealType *__restrict__ translations, // [num_samples, 3]
    double *__restrict__ coords_out) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= num_samples) {
        return;
    }
    const int mol_sample = samples[sample_idx];
    const int mol_start = mol_offsets[mol_sample];
    const int mol_end = mol_offsets[mol_sample + 1];
    const int num_atoms = mol_end - mol_start;
    const RealType box_x = box[0 * 3 + 0];
    const RealType box_y = box[1 * 3 + 1];
    const RealType box_z = box[2 * 3 + 2];

    const RealType translation_x = box_x * translations[sample_idx * 3 + 0];
    const RealType translation_y = box_y * translations[sample_idx * 3 + 1];
    const RealType translation_z = box_z * translations[sample_idx * 3 + 2];

    RealType quat[4];
    RealType local_coords[4];
    for (int i = 0; i < num_atoms; i++) {
        // TBD avoid multiple global reads for the same quaternion, maybe doesn't matter with L1 cache
        quat[0] = quaternions[sample_idx * 4 + 0];
        quat[1] = quaternions[sample_idx * 4 + 1];
        quat[2] = quaternions[sample_idx * 4 + 2];
        quat[3] = quaternions[sample_idx * 4 + 3];

        local_coords[0] = 0;
        local_coords[1] = coords[(mol_start + i) * 3 + 0];
        local_coords[2] = coords[(mol_start + i) * 3 + 1];
        local_coords[3] = coords[(mol_start + i) * 3 + 2];

        rotate_coordinates_by_quaternion(local_coords, quat);

        coords_out[(mol_start + i) * 3 + 0] = local_coords[1] + translation_x;
        coords_out[(mol_start + i) * 3 + 1] = local_coords[2] + translation_y;
        coords_out[(mol_start + i) * 3 + 2] = local_coords[3] + translation_z;
    }
}

} // namespace timemachine
