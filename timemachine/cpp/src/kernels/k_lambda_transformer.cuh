template <typename NumericType> NumericType __device__ __forceinline__ transform_lambda_charge(NumericType lambda) {
    return lambda;
}

template <typename NumericType> NumericType __device__ __forceinline__ transform_lambda_sigma(NumericType lambda) {
    return lambda;
}

template <typename NumericType> NumericType __device__ __forceinline__ transform_lambda_epsilon(NumericType lambda) {
    return lambda;
}

template <typename NumericType> NumericType __device__ __forceinline__ transform_lambda_w(NumericType lambda) {
    return lambda;
}

void __global__ k_compute_w_coords(
    const int N,
    const double lambda,
    const double cutoff,
    const int *__restrict__ lambda_plane_idxs, // 0 or 1, shift
    const int *__restrict__ lambda_offset_idxs,
    double *__restrict__ coords_w);
