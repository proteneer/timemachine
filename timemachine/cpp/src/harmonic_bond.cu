#include "gpu_utils.cuh"
#include "harmonic_bond.hpp"
#include "k_harmonic_bond.cuh"
#include <chrono>
#include <complex>
#include <iostream>
#include <vector>

namespace timemachine {

template <typename RealType>
HarmonicBond<RealType>::HarmonicBond(
    const std::vector<int> &bond_idxs, const std::vector<int> &lambda_mult, const std::vector<int> &lambda_offset)
    : B_(bond_idxs.size() / 2) {

    if (bond_idxs.size() % 2 != 0) {
        throw std::runtime_error("bond_idxs.size() must be exactly 2*k!");
    }

    if (lambda_mult.size() > 0 && lambda_mult.size() != B_) {
        throw std::runtime_error("bad lambda_mult size()");
    }

    if (lambda_offset.size() > 0 && lambda_offset.size() != B_) {
        throw std::runtime_error("bad lambda_offset size()");
    }

    for (int b = 0; b < B_; b++) {
        auto src = bond_idxs[b * 2 + 0];
        auto dst = bond_idxs[b * 2 + 1];
        if (src == dst) {
            throw std::runtime_error("src == dst");
        }
    }

    gpuErrchk(cudaMalloc(&d_bond_idxs_, B_ * 2 * sizeof(*d_bond_idxs_)));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], B_ * 2 * sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_mult_, B_ * sizeof(*d_lambda_mult_)));
    gpuErrchk(cudaMalloc(&d_lambda_offset_, B_ * sizeof(*d_lambda_offset_)));

    if (lambda_mult.size() > 0) {
        gpuErrchk(cudaMemcpy(d_lambda_mult_, &lambda_mult[0], B_ * sizeof(*d_lambda_mult_), cudaMemcpyHostToDevice));
    } else {
        initializeArray(B_, d_lambda_mult_, 0);
    }

    if (lambda_offset.size() > 0) {
        gpuErrchk(
            cudaMemcpy(d_lambda_offset_, &lambda_offset[0], B_ * sizeof(*d_lambda_offset_), cudaMemcpyHostToDevice));
    } else {
        // can't memset this
        initializeArray(B_, d_lambda_offset_, 1);
    }
};

template <typename RealType> HarmonicBond<RealType>::~HarmonicBond() {
    gpuErrchk(cudaFree(d_bond_idxs_));
    gpuErrchk(cudaFree(d_lambda_mult_));
    gpuErrchk(cudaFree(d_lambda_offset_));
};

template <typename RealType>
void HarmonicBond<RealType>::execute_device(
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

    if (B_ > 0) {
        int tpb = 32;
        int blocks = (B_ + tpb - 1) / tpb;

        k_harmonic_bond<RealType><<<blocks, tpb, 0, stream>>>(
            B_, d_x, d_p, lambda, d_lambda_mult_, d_lambda_offset_, d_bond_idxs_, d_du_dx, d_du_dp, d_du_dl, d_u);
        gpuErrchk(cudaPeekAtLastError());
    }
};

template class HarmonicBond<double>;
template class HarmonicBond<float>;

} // namespace timemachine
