#include "gpu_utils.cuh"
#include "k_nonbonded_pairs.cuh"
#include "nonbonded_pairs.hpp"
#include <stdexcept>
#include <vector>

namespace timemachine {

template <typename RealType, bool Interpolated>
NonbondedPairs<RealType, Interpolated>::NonbondedPairs(
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
          kernel_cache_.program(kernel_src.c_str()).kernel("k_permute_interpolated").instantiate()) {

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

    gpuErrchk(cudaMalloc(&d_dp_dl_, N_ * 3 * sizeof(*d_dp_dl_)));

    gpuErrchk(cudaMalloc(&d_scales_, M_ * 2 * sizeof(*d_scales_)));
    gpuErrchk(cudaMemcpy(d_scales_, &scales[0], M_ * 2 * sizeof(*d_scales_), cudaMemcpyHostToDevice));

    // construct identity permutation
    if (Interpolated) {
        gpuErrchk(cudaMalloc(&d_perm_, N_ * sizeof(*d_perm_)));
        for (int i = 0; i < N_; i++) {
            d_perm_[i] = i;
        }
    }
};

template <typename RealType, bool Interpolated> NonbondedPairs<RealType, Interpolated>::~NonbondedPairs() {
    gpuErrchk(cudaFree(d_pair_idxs_));
    gpuErrchk(cudaFree(d_scales_));
    gpuErrchk(cudaFree(d_lambda_plane_idxs_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));
    gpuErrchk(cudaFree(d_du_dp_buffer_));
    gpuErrchk(cudaFree(d_w_));
    gpuErrchk(cudaFree(d_dw_dl_));
    gpuErrchk(cudaFree(d_dp_dl_));
};

template <typename RealType, bool Interpolated>
void NonbondedPairs<RealType, Interpolated>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    const double lambda,
    unsigned long long *d_du_dx,
    double *d_du_dp,
    unsigned long long *d_du_dl,
    unsigned long long *d_u,
    cudaStream_t stream) {

    const int tpb = 32;

    dim3 dimGrid((M_ + tpb - 1) / tpb, 1, 1);

    CUresult result = compute_w_coords_instance_.configure(M_, tpb, 0, stream)
                          .launch(N, lambda, cutoff_, d_lambda_plane_idxs_, d_lambda_offset_idxs_, d_w_, d_dw_dl_);
    if (result != 0) {
        throw std::runtime_error("Driver call to k_compute_w_coords");
    }

    if (Interpolated) {
        CUresult result = compute_permute_interpolated_.configure(dimGrid, tpb, 0, stream)
                              .launch(lambda, N, d_perm_, d_p, d_p, d_dp_dl_);
        if (result != 0) {
            throw std::runtime_error("Driver call to k_permute_interpolated failed");
        }
    } else {
        gpuErrchk(cudaMemsetAsync(d_dp_dl_, 0, N * 3 * sizeof(*d_dp_dl_), stream))
    }

    k_nonbonded_pairs<RealType><<<dimGrid, tpb, 0, stream>>>(
        M_,
        d_x,
        d_p,
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
}

template class NonbondedPairs<double, true>;
template class NonbondedPairs<float, true>;

template class NonbondedPairs<double, false>;
template class NonbondedPairs<float, false>;

} // namespace timemachine
