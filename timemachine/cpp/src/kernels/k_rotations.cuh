

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

    // Normalize the quaternions
    RealType quat2 = quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3];
    RealType inv_norm = rsqrt(quat2);
    quat[0] = quat[0] * inv_norm;
    quat[1] = quat[1] * inv_norm;
    quat[2] = quat[2] * inv_norm;
    quat[3] = quat[3] * inv_norm;

    hamilton_product(quat, local_coords, intermediate);

    RealType quat_conjugate[4];
    quat_conjugate[0] = quat[0];
    quat_conjugate[1] = -quat[1];
    quat_conjugate[2] = -quat[2];
    quat_conjugate[3] = -quat[3];

    hamilton_product(intermediate, quat_conjugate, local_coords);

    // Truncate off the first of the local coords
    rotated_coords[(coord_idx * n_rotations * 3) + (rotation_idx * 3) + 0] = local_coords[1];
    rotated_coords[(coord_idx * n_rotations * 3) + (rotation_idx * 3) + 1] = local_coords[2];
    rotated_coords[(coord_idx * n_rotations * 3) + (rotation_idx * 3) + 2] = local_coords[3];
}

} // namespace timemachine
