#pragma once

namespace timemachine {

template <typename RealType>
void __device__ __forceinline__
hamilton_product(const RealType *__restrict__ q1, const RealType *__restrict__ q2, RealType *__restrict__ out);

template <typename RealType>
void __device__ __forceinline__
rotate_coordinates_by_quaternion(RealType *__restrict__ local_coords, RealType *__restrict__ quaternion);

template <typename RealType>
void __global__ k_rotate_coordinates(
    const int N,                              // Number of coordinates
    const int n_rotations,                    // Number of quaternions
    const double *__restrict__ coords,        // [N, 3]
    const RealType *__restrict__ quaternions, // [n_rotations, 4]
    double *__restrict__ rotated_coords       // [N * n_rotations, 3]
);

template <typename RealType, bool SCALE>
void __global__ k_rotate_and_translate_mols(
    const int num_samples,
    const double *__restrict__ coords,         // [N, 3]
    const double *__restrict__ box,            // [3, 3]
    const int *__restrict__ samples,           // [num_samples]
    const int *__restrict__ mol_offsets,       // [max_sample_idx+1]
    const RealType *__restrict__ quaternions,  // [num_samples, 4]
    const RealType *__restrict__ translations, // [num_samples, 3]
    double *__restrict__ coords_out);

} // namespace timemachine
