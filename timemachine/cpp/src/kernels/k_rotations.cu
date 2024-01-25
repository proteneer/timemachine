#include "k_fixed_point.cuh"
#include "k_rotations.cuh"
#include "stdio.h"

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

template <typename RealType, bool SCALE>
void __global__ k_rotate_and_translate_mols(
    const int batch_size,
    const double *__restrict__ coords,         // [N, 3]
    const double *__restrict__ box,            // [3, 3]
    const int *__restrict__ samples,           // [batch_size]
    const int *__restrict__ mol_offsets,       // [batch_size + 1]
    const RealType *__restrict__ quaternions,  // [batch_size, 4]
    const RealType *__restrict__ translations, // [batch_size, 3]
    double *__restrict__ coords_out            // [batch_size, num_atoms, 3]
) {

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const RealType box_x = box[0 * 3 + 0];
    const RealType box_y = box[1 * 3 + 1];
    const RealType box_z = box[2 * 3 + 2];

    const RealType inv_box_x = 1 / box_x;
    const RealType inv_box_y = 1 / box_y;
    const RealType inv_box_z = 1 / box_z;

    while (batch_idx < batch_size) {
        int mol_sample = samples[batch_idx];
        int mol_start = mol_offsets[mol_sample];
        int mol_end = mol_offsets[mol_sample + 1];
        int num_atoms = mol_end - mol_start;

        RealType ref_quat[4];
        ref_quat[0] = quaternions[batch_idx * 4 + 0];
        ref_quat[1] = quaternions[batch_idx * 4 + 1];
        ref_quat[2] = quaternions[batch_idx * 4 + 2];
        ref_quat[3] = quaternions[batch_idx * 4 + 3];

        RealType translation_x = SCALE ? box_x * translations[batch_idx * 3 + 0] : translations[batch_idx * 3 + 0];
        RealType translation_y = SCALE ? box_y * translations[batch_idx * 3 + 1] : translations[batch_idx * 3 + 1];
        RealType translation_z = SCALE ? box_z * translations[batch_idx * 3 + 2] : translations[batch_idx * 3 + 2];

        // Image the translation in the home box
        translation_x -= box_x * floor(translation_x * inv_box_x);
        translation_y -= box_y * floor(translation_y * inv_box_y);
        translation_z -= box_z * floor(translation_z * inv_box_z);

        unsigned long long centroid_accum_x = 0;
        unsigned long long centroid_accum_y = 0;
        unsigned long long centroid_accum_z = 0;
        for (int i = 0; i < num_atoms; i++) {
            centroid_accum_x += FLOAT_TO_FIXED<RealType>(coords[(mol_start + i) * 3 + 0]);
            centroid_accum_y += FLOAT_TO_FIXED<RealType>(coords[(mol_start + i) * 3 + 1]);
            centroid_accum_z += FLOAT_TO_FIXED<RealType>(coords[(mol_start + i) * 3 + 2]);
        }

        RealType centroid_x = FIXED_TO_FLOAT<RealType>(centroid_accum_x) / static_cast<RealType>(num_atoms);
        RealType centroid_y = FIXED_TO_FLOAT<RealType>(centroid_accum_y) / static_cast<RealType>(num_atoms);
        RealType centroid_z = FIXED_TO_FLOAT<RealType>(centroid_accum_z) / static_cast<RealType>(num_atoms);

        RealType quat[4];
        RealType local_coords[4];
        for (int i = 0; i < num_atoms; i++) {
            // Load in the quaternion from the reference buffer
            quat[0] = ref_quat[0];
            quat[1] = ref_quat[1];
            quat[2] = ref_quat[2];
            quat[3] = ref_quat[3];

            local_coords[0] = 0;
            local_coords[1] = coords[(mol_start + i) * 3 + 0] - centroid_x;
            local_coords[2] = coords[(mol_start + i) * 3 + 1] - centroid_y;
            local_coords[3] = coords[(mol_start + i) * 3 + 2] - centroid_z;

            rotate_coordinates_by_quaternion(local_coords, quat);

            // After rotating coordinates, set the new centroid
            local_coords[1] += translation_x;
            local_coords[2] += translation_y;
            local_coords[3] += translation_z;
            coords_out[(batch_idx * num_atoms) + (i * 3) + 0] = local_coords[1];
            coords_out[(batch_idx * num_atoms) + (i * 3) + 1] = local_coords[2];
            coords_out[(batch_idx * num_atoms) + (i * 3) + 2] = local_coords[3];
        }

        batch_idx += gridDim.x * blockDim.x;
    }
}

template void __global__ k_rotate_coordinates<float>(int, int, const double *, const float *, double *);
template void __global__ k_rotate_coordinates<double>(int, int, const double *, const double *, double *);

template void __global__ k_rotate_and_translate_mols<float, false>(
    int, const double *, const double *, const int *, const int *, const float *, const float *, double *);
template void __global__ k_rotate_and_translate_mols<float, true>(
    int, const double *, const double *, const int *, const int *, const float *, const float *, double *);
template void __global__ k_rotate_and_translate_mols<double, false>(
    int, const double *, const double *, const int *, const int *, const double *, const double *, double *);
template void __global__ k_rotate_and_translate_mols<double, true>(
    int, const double *, const double *, const int *, const int *, const double *, const double *, double *);

} // namespace timemachine
