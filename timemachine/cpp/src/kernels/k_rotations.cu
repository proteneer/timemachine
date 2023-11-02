#include "k_rotations.cuh"

namespace timemachine {

template <typename RealType>
void __device__ __forceinline__
hamilton_product(const RealType *__restrict__ q1, const RealType *__restrict__ q2, RealType *__restrict__ out) {
    RealType w1 = q1[0];
    RealType x1 = q1[1];
    RealType y1 = q1[2];
    RealType z1 = q1[3];

    RealType w2 = q2[0];
    RealType x2 = q2[1];
    RealType y2 = q2[2];
    RealType z2 = q2[3];

    out[0] = (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2);
    out[1] = (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2);
    out[2] = (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2);
    out[3] = (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2);
}

template <typename RealType>
void __device__ __forceinline__
rotate_coordinates_by_quaternion(RealType *__restrict__ local_coords, RealType *__restrict__ quaternion) {
    RealType intermediate[4];

    RealType quat2 = quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2] +
                     quaternion[3] * quaternion[3];
    RealType inv_norm = rsqrt(quat2);
    quaternion[0] = quaternion[0] * inv_norm;
    quaternion[1] = quaternion[1] * inv_norm;
    quaternion[2] = quaternion[2] * inv_norm;
    quaternion[3] = quaternion[3] * inv_norm;

    hamilton_product(quaternion, local_coords, intermediate);

    RealType quat_conjugate[4];
    quat_conjugate[0] = quaternion[0];
    quat_conjugate[1] = -quaternion[1];
    quat_conjugate[2] = -quaternion[2];
    quat_conjugate[3] = -quaternion[3];

    hamilton_product(intermediate, quat_conjugate, local_coords);
}

template <typename RealType>
void __global__ k_rotate_coordinates(
    const int N,                              // Number of coordinates
    const int n_rotations,                    // Number of quaternions
    const double *__restrict__ coords,        // [N, 3]
    const RealType *__restrict__ quaternions, // [n_rotations, 4]
    double *__restrict__ rotated_coords       // [N * n_rotations, 3]
) {
    const int rotation_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rotation_idx >= n_rotations) {
        return;
    }
    const int coord_idx = blockIdx.y;
    if (coord_idx >= N) {
        return;
    }
    // TBD: Potentially could optimize by loading in different coords in each thread and shuffling the coords
    RealType intermediate[4];

    RealType local_coords[4];
    local_coords[0] = 0; // Prepend a zero to the local coords
    local_coords[1] = coords[coord_idx * 3 + 0];
    local_coords[2] = coords[coord_idx * 3 + 1];
    local_coords[3] = coords[coord_idx * 3 + 2];

    RealType quat[4];
    quat[0] = quaternions[rotation_idx * 4 + 0];
    quat[1] = quaternions[rotation_idx * 4 + 1];
    quat[2] = quaternions[rotation_idx * 4 + 2];
    quat[3] = quaternions[rotation_idx * 4 + 3];

    rotate_coordinates_by_quaternion(local_coords, quat);

    // Truncate off the first of the local coords
    rotated_coords[(coord_idx * n_rotations * 3) + (rotation_idx * 3) + 0] = local_coords[1];
    rotated_coords[(coord_idx * n_rotations * 3) + (rotation_idx * 3) + 1] = local_coords[2];
    rotated_coords[(coord_idx * n_rotations * 3) + (rotation_idx * 3) + 2] = local_coords[3];
}

template <typename RealType>
void __global__ k_rotate_and_translate_mols(
    const int num_samples,
    const double *__restrict__ coords,         // [N, 3]
    const double *__restrict__ box,            // [3, 3]
    const int *__restrict__ samples,           // [num_samples]
    const int *__restrict__ mol_offsets,       // [max_sample_idx+1]
    const RealType *__restrict__ quaternions,  // [num_samples, 4]
    const RealType *__restrict__ translations, // [num_samples, 3]
    double *__restrict__ coords_out) {

    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const RealType box_x = box[0 * 3 + 0];
    const RealType box_y = box[1 * 3 + 1];
    const RealType box_z = box[2 * 3 + 2];

    while (sample_idx < num_samples) {
        int mol_sample = samples[sample_idx];
        int mol_start = mol_offsets[mol_sample];
        int mol_end = mol_offsets[mol_sample + 1];
        int num_atoms = mol_end - mol_start;

        RealType translation_x = box_x * translations[sample_idx * 3 + 0];
        RealType translation_y = box_y * translations[sample_idx * 3 + 1];
        RealType translation_z = box_z * translations[sample_idx * 3 + 2];

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
        sample_idx += gridDim.x * blockDim.x;
    }
}

template void __global__ k_rotate_coordinates<float>(int, int, const double *, const float *, double *);
template void __global__ k_rotate_coordinates<double>(int, int, const double *, const double *, double *);

template void __global__ k_rotate_and_translate_mols<float>(
    int, const double *, const double *, const int *, const int *, const float *, const float *, double *);
template void __global__ k_rotate_and_translate_mols<double>(
    int, const double *, const double *, const int *, const int *, const double *, const double *, double *);

} // namespace timemachine
