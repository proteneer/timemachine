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
    double *__restrict__ coords_w,
    double *__restrict__ dw_dl);

void __global__ k_gather_interpolated(
    const double lambda,
    const int N,
    const unsigned int *__restrict__ idxs,
    const double *__restrict__ d_p0, // [P] initial parameters
    const double *__restrict__ d_p1, // [P] final parameters
    double *__restrict__ d_gathered_p,
    double *__restrict__ d_gathered_dp_dl);

void __global__ k_add_du_dp_interpolated(
    const double lambda,
    const int N,
    const unsigned long long *__restrict__ src,
    unsigned long long *__restrict__ dest);
