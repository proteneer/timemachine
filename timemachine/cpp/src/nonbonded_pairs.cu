#include "gpu_utils.cuh"
#include "k_nonbonded_pairs.cuh"
#include "math_utils.cuh"
#include "nonbonded_pairs.hpp"
#include <stdexcept>
#include <vector>

namespace timemachine {

template <typename RealType, bool Negated, bool Interpolated>
NonbondedPairs<RealType, Negated, Interpolated>::NonbondedPairs(
    const std::vector<int> &pair_idxs,          // [M, 2]
    const std::vector<double> &scales,          // [M, 2]
    const std::vector<int> &lambda_plane_idxs,  // [N]
    const std::vector<int> &lambda_offset_idxs, // [N]
    const double beta,
    const double cutoff,
    const std::string &kernel_src)
    : N_(lambda_offset_idxs.size()), M_(pair_idxs.size() / 2), beta_(beta), cutoff_(cutoff),
      compute_w_coords_instance_(kernel_cache_.program(kernel_src.c_str()).kernel("k_compute_w_coords").instantiate()),
      compute_permute_interpolated_(
          kernel_cache_.program(kernel_src.c_str()).kernel("k_permute_interpolated").instantiate()),
      compute_add_du_dp_interpolated_(
          kernel_cache_.program(kernel_src.c_str()).kernel("k_add_du_dp_interpolated").instantiate()) {

    if (pair_idxs.size() % 2 != 0) {
        throw std::runtime_error("pair_idxs.size() must be exactly 2*M");
    }

    for (int i = 0; i < M_; i++) {
        auto src = pair_idxs[i * 2 + 0];
        auto dst = pair_idxs[i * 2 + 1];
        if (src == dst) {
            throw std::runtime_error("illegal pair with src == dst");
        }
    }

    if (scales.size() / 2 != M_) {
        throw std::runtime_error("bad scales size!");
    }

    gpuErrchk(cudaMalloc(&d_pair_idxs_, M_ * 2 * sizeof(*d_pair_idxs_)));
    gpuErrchk(cudaMemcpy(d_pair_idxs_, &pair_idxs[0], M_ * 2 * sizeof(*d_pair_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_plane_idxs_, N_ * sizeof(*d_lambda_plane_idxs_)));
    gpuErrchk(cudaMemcpy(
        d_lambda_plane_idxs_, &lambda_plane_idxs[0], N_ * sizeof(*d_lambda_plane_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_ * sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(
        d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_ * sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_w_, N_ * sizeof(*d_w_)));
    gpuErrchk(cudaMalloc(&d_dw_dl_, N_ * sizeof(*d_dw_dl_)));
    gpuErrchk(cudaMalloc(&d_du_dp_buffer_, N_ * 3 * sizeof(*d_du_dp_buffer_)));

    gpuErrchk(cudaMalloc(&d_p_interp_, N_ * 3 * sizeof(*d_p_interp_)));
    gpuErrchk(cudaMalloc(&d_dp_dl_, N_ * 3 * sizeof(*d_dp_dl_)));

    gpuErrchk(cudaMalloc(&d_scales_, M_ * 2 * sizeof(*d_scales_)));
    gpuErrchk(cudaMemcpy(d_scales_, &scales[0], M_ * 2 * sizeof(*d_scales_), cudaMemcpyHostToDevice));

    if (Interpolated) {

        gpuErrchk(cudaMalloc(&d_scales_, M_ * 2 * sizeof(*d_scales_)));
        gpuErrchk(cudaMemcpy(d_scales_, &scales[0], M_ * 2 * sizeof(*d_scales_), cudaMemcpyHostToDevice));

        // initialize identity permutation
        std::vector<int> perm = std::vector<int>(N_);
        for (int i = 0; i < N_; i++) {
            perm[i] = i;
        }
        gpuErrchk(cudaMalloc(&d_perm_, N_ * sizeof(*d_perm_)));
        gpuErrchk(cudaMemcpy(d_perm_, &perm[0], N_ * sizeof(*d_perm_), cudaMemcpyHostToDevice));
    }
};

template <typename RealType, bool Negated, bool Interpolated>
NonbondedPairs<RealType, Negated, Interpolated>::~NonbondedPairs() {
    gpuErrchk(cudaFree(d_pair_idxs_));
    gpuErrchk(cudaFree(d_scales_));
    gpuErrchk(cudaFree(d_lambda_plane_idxs_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));
    gpuErrchk(cudaFree(d_du_dp_buffer_));
    gpuErrchk(cudaFree(d_w_));
    gpuErrchk(cudaFree(d_dw_dl_));
    gpuErrchk(cudaFree(d_p_interp_));
    gpuErrchk(cudaFree(d_dp_dl_));

    if (Interpolated) {
        gpuErrchk(cudaFree(d_perm_));
    }
};

template <typename RealType, bool Negated, bool Interpolated>
void NonbondedPairs<RealType, Negated, Interpolated>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    const double lambda,
    unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    unsigned long long *d_du_dl,
    unsigned long long *d_u,
    cudaStream_t stream) {

    const int tpb = 32;

    int num_blocks = ceil_divide(N, tpb);
    dim3 dimGrid(num_blocks, 3, 1);

    CUresult result = compute_w_coords_instance_.configure(dimGrid, tpb, 0, stream)
                          .launch(N, lambda, cutoff_, d_lambda_plane_idxs_, d_lambda_offset_idxs_, d_w_, d_dw_dl_);
    if (result != 0) {
        throw std::runtime_error("Driver call to k_compute_w_coords");
    }

    if (Interpolated) {
        CUresult result = compute_permute_interpolated_.configure(dimGrid, tpb, 0, stream)
                              .launch(lambda, N, d_perm_, d_p, d_p_interp_, d_dp_dl_);
        if (result != 0) {
            throw std::runtime_error("Driver call to k_permute_interpolated failed");
        }
    } else {
        gpuErrchk(cudaMemsetAsync(d_dp_dl_, 0, N * 3 * sizeof(*d_dp_dl_), stream))
    }

    if (d_du_dp) {
        gpuErrchk(cudaMemsetAsync(d_du_dp_buffer_, 0, N * 3 * sizeof(*d_du_dp_buffer_), stream))
    }

    int num_blocks_pairs = ceil_divide(M_, tpb);

    k_nonbonded_pairs<RealType, Negated><<<num_blocks_pairs, tpb, 0, stream>>>(
        M_,
        d_x,
        Interpolated ? d_p_interp_ : d_p,
        d_box,
        d_dp_dl_,
        d_w_,
        d_dw_dl_,
        d_pair_idxs_,
        d_scales_,
        beta_,
        cutoff_,
        d_du_dx,
        d_du_dp_buffer_,
        d_du_dl,
        d_u);

    gpuErrchk(cudaPeekAtLastError());

    if (d_du_dp) {
        if (Interpolated) {
            CUresult result = compute_add_du_dp_interpolated_.configure(dimGrid, tpb, 0, stream)
                                  .launch(lambda, N, d_du_dp_buffer_, d_du_dp);
            if (result != 0) {
                throw std::runtime_error("Driver call to k_add_du_dp_interpolated failed");
            }
        } else {
            k_add_ull_to_ull<<<dimGrid, tpb, 0, stream>>>(N, d_du_dp_buffer_, d_du_dp);
        }
        gpuErrchk(cudaPeekAtLastError());
    }
}

// TODO: this implementation is duplicated from NonbondedDense. Worth adding NonbondedBase?
template <typename RealType, bool Negated, bool Interpolated>
void NonbondedPairs<RealType, Negated, Interpolated>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {

    // In the interpolated case we have derivatives for the initial and final parameters
    const int num_tuples = Interpolated ? N * 2 : N;

    for (int i = 0; i < num_tuples; i++) {
        const int idx_charge = i * 3 + 0, idx_sig = i * 3 + 1, idx_eps = i * 3 + 2;
        du_dp_float[idx_charge] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DCHARGE>(du_dp[idx_charge]);
        du_dp_float[idx_sig] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DSIG>(du_dp[idx_sig]);
        du_dp_float[idx_eps] = FIXED_TO_FLOAT_DU_DP<double, FIXED_EXPONENT_DU_DEPS>(du_dp[idx_eps]);
    }
}

template class NonbondedPairs<double, true, true>;
template class NonbondedPairs<float, true, true>;

template class NonbondedPairs<double, false, true>;
template class NonbondedPairs<float, false, true>;

template class NonbondedPairs<double, true, false>;
template class NonbondedPairs<float, true, false>;

template class NonbondedPairs<double, false, false>;
template class NonbondedPairs<float, false, false>;

} // namespace timemachine
