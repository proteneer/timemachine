#include "centroid_restraint.hpp"
#include "gpu_utils.cuh"
#include "k_centroid_restraint.cuh"
#include <chrono>
#include <complex>
#include <iostream>
#include <vector>

namespace timemachine {

template <typename RealType>
CentroidRestraint<RealType>::CentroidRestraint(
    const std::vector<int> &group_a_idxs, const std::vector<int> &group_b_idxs, const double kb, const double b0)
    : N_A_(group_a_idxs.size()), N_B_(group_b_idxs.size()), kb_(kb), b0_(b0) {

    gpuErrchk(cudaMalloc(&d_group_a_idxs_, N_A_ * sizeof(*d_group_a_idxs_)));
    gpuErrchk(cudaMemcpy(d_group_a_idxs_, &group_a_idxs[0], N_A_ * sizeof(*d_group_a_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_group_b_idxs_, N_B_ * sizeof(*d_group_b_idxs_)));
    gpuErrchk(cudaMemcpy(d_group_b_idxs_, &group_b_idxs[0], N_B_ * sizeof(*d_group_b_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_centroid_a_, 3 * sizeof(*d_centroid_a_)));
    gpuErrchk(cudaMalloc(&d_centroid_b_, 3 * sizeof(*d_centroid_b_)));
};

template <typename RealType> CentroidRestraint<RealType>::~CentroidRestraint() {
    gpuErrchk(cudaFree(d_group_a_idxs_));
    gpuErrchk(cudaFree(d_group_b_idxs_));
    gpuErrchk(cudaFree(d_centroid_a_));
    gpuErrchk(cudaFree(d_centroid_b_));
};

template <typename RealType>
void CentroidRestraint<RealType>::execute_device(
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

    int tpb = 32;

    int blocks = (N_B_ + N_A_ + tpb - 1) / tpb;
    gpuErrchk(cudaMemsetAsync(d_centroid_a_, 0.0, 3 * sizeof(*d_centroid_a_), stream));
    gpuErrchk(cudaMemsetAsync(d_centroid_b_, 0.0, 3 * sizeof(*d_centroid_b_), stream));
    k_calc_centroid<RealType>
        <<<blocks, tpb, 0, stream>>>(d_x, d_group_a_idxs_, d_group_b_idxs_, N_A_, N_B_, d_centroid_a_, d_centroid_b_);

    gpuErrchk(cudaPeekAtLastError());

    k_centroid_restraint<RealType><<<blocks, tpb, 0, stream>>>(
        d_x, d_group_a_idxs_, d_group_b_idxs_, N_A_, N_B_, d_centroid_a_, d_centroid_b_, kb_, b0_, d_du_dx, d_u);
    gpuErrchk(cudaPeekAtLastError());
};

template class CentroidRestraint<double>;
template class CentroidRestraint<float>;

} // namespace timemachine
