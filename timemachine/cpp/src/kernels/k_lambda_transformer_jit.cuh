jit_program

#include "timemachine/cpp/src/kernels/surreal.cuh"
#include "timemachine/cpp/src/kernels/k_fixed_point.cuh"

#define PI 3.141592653589793115997963468544185161

template <typename NumericType>
NumericType __device__ __forceinline__ transform_lambda_charge(NumericType lambda, int atom_idx, int N) {
    CUSTOM_EXPRESSION_CHARGE;
}

template <typename NumericType>
NumericType __device__ __forceinline__ transform_lambda_sigma(NumericType lambda, int atom_idx, int N) {
    CUSTOM_EXPRESSION_SIGMA;
}

template <typename NumericType>
NumericType __device__ __forceinline__ transform_lambda_epsilon(NumericType lambda, int atom_idx, int N) {
    CUSTOM_EXPRESSION_EPSILON;
}

template <typename NumericType>
NumericType __device__ __forceinline__ transform_lambda_w(NumericType lambda, int atom_idx, int N) {
    CUSTOM_EXPRESSION_W;
}

void __global__ k_compute_w_coords(
    const int N,
    const double lambda,
    const double cutoff,
    const int * __restrict__ lambda_plane_idxs, // 0 or 1, shift
    const int * __restrict__ lambda_offset_idxs,
    double * __restrict__ coords_w,
    double * __restrict__ dw_dl) {

    int atom_i_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(atom_i_idx >= N) {
        return;
    }

    int lambda_offset_i = atom_i_idx < N ? lambda_offset_idxs[atom_i_idx] : 0;
    int lambda_plane_i = atom_i_idx < N ? lambda_plane_idxs[atom_i_idx] : 0;

    double f_lambda = transform_lambda_w(lambda, atom_i_idx, N);

    double step = 1e-7;
    Surreal<double> lambda_surreal(lambda, step);
    double f_lambda_grad = (transform_lambda_w(lambda_surreal, atom_i_idx, N).imag)/step;

    double coords_w_i = (lambda_plane_i + lambda_offset_i*f_lambda)*cutoff;
    double dw_dl_i = lambda_offset_i*f_lambda_grad*cutoff;

    coords_w[atom_i_idx] = coords_w_i;
    dw_dl[atom_i_idx] = dw_dl_i;

} // 0 or 1, how much we offset from the plane by )

void __global__ k_permute_interpolated(
    const double lambda,
    const int N,
    const unsigned int * __restrict__ perm,
    const double * __restrict__ d_p,
    double * __restrict__ d_sorted_p,
    double * __restrict__ d_sorted_dp_dl) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if(idx >= N) {
        return;
    }

    int size = N*stride;

    int source_idx = idx*stride+stride_idx;
    int target_idx = perm[idx]*stride+stride_idx;

    double step = 1e-7;
    Surreal<double> lambda_surreal(lambda, step);

    double f_lambda;
    double f_lambda_grad;

    if(stride_idx == 0) {
        f_lambda = transform_lambda_charge(lambda, idx, N);
        f_lambda_grad = (transform_lambda_charge(lambda_surreal, idx, N).imag)/step;
    }
    if(stride_idx == 1) {
        f_lambda = transform_lambda_sigma(lambda, idx, N);
        f_lambda_grad = (transform_lambda_sigma(lambda_surreal, idx, N).imag)/step;
    }
    if(stride_idx == 2) {
        f_lambda = transform_lambda_epsilon(lambda, idx, N);
        f_lambda_grad = (transform_lambda_epsilon(lambda_surreal, idx, N).imag)/step;
    }

    d_sorted_p[source_idx] = (1-f_lambda)*d_p[target_idx] + f_lambda*d_p[size+target_idx];
    d_sorted_dp_dl[source_idx] = f_lambda_grad*(d_p[size+target_idx] - d_p[target_idx]);

}

void __global__ k_add_ull_to_real_interpolated(
    const double lambda,
    const int N,
    const unsigned long long * __restrict__ ull_array,
    double * __restrict__ real_array) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if(idx >= N) {
        return;
    }

    int size = N*stride;
    int target_idx = idx*stride+stride_idx;

    // handle charges, sigmas, epsilons with different exponents
    if(stride_idx == 0) {
        double f_lambda = transform_lambda_charge(lambda, idx, N);
        real_array[target_idx] += (1-f_lambda)*FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DCHARGE>(ull_array[target_idx]);
        real_array[size+target_idx] += f_lambda*FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DCHARGE>(ull_array[target_idx]);
    } else if(stride_idx == 1) {
        double f_lambda = transform_lambda_sigma(lambda, idx, N);
        real_array[target_idx] += (1-f_lambda)*FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DSIG>(ull_array[target_idx]);
        real_array[size+target_idx] += f_lambda*FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DSIG>(ull_array[target_idx]);
    } else if(stride_idx == 2) {
        double f_lambda = transform_lambda_epsilon(lambda, idx, N);
        real_array[target_idx] += (1-f_lambda)*FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DEPS>(ull_array[target_idx]);
        real_array[size+target_idx] += f_lambda*FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DEPS>(ull_array[target_idx]);
    }

}
