#include "gpu_utils.cuh"
#include "harmonic_angle.hpp"
#include "k_harmonic_angle.cuh"
#include <chrono>
#include <complex>
#include <iostream>
#include <vector>

namespace timemachine {

template <typename RealType>
HarmonicAngle<RealType>::HarmonicAngle(
    const std::vector<int> &angle_idxs, // [A, 3]
    const std::vector<int> &lambda_mult,
    const std::vector<int> &lambda_offset)
    : A_(angle_idxs.size() / 3) {

    if (angle_idxs.size() % 3 != 0) {
        throw std::runtime_error("angle_idxs.size() must be exactly 3*A");
    }

    if (lambda_mult.size() > 0 && lambda_mult.size() != A_) {
        throw std::runtime_error("bad lambda_mult size()");
    }

    if (lambda_offset.size() > 0 && lambda_offset.size() != A_) {
        throw std::runtime_error("bad lambda_offset size()");
    }

    for (int a = 0; a < A_; a++) {
        auto i = angle_idxs[a * 3 + 0];
        auto j = angle_idxs[a * 3 + 1];
        auto k = angle_idxs[a * 3 + 2];
        if (i == j || j == k || i == k) {
            throw std::runtime_error("angle triplets must be unique");
        }
    }

    gpuErrchk(cudaMalloc(&d_angle_idxs_, A_ * 3 * sizeof(*d_angle_idxs_)));
    gpuErrchk(cudaMemcpy(d_angle_idxs_, &angle_idxs[0], A_ * 3 * sizeof(*d_angle_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_mult_, A_ * sizeof(*d_lambda_mult_)));
    gpuErrchk(cudaMalloc(&d_lambda_offset_, A_ * sizeof(*d_lambda_offset_)));

    if (lambda_mult.size() > 0) {
        gpuErrchk(cudaMemcpy(d_lambda_mult_, &lambda_mult[0], A_ * sizeof(*d_lambda_mult_), cudaMemcpyHostToDevice));
    } else {
        initializeArray(A_, d_lambda_mult_, 0);
    }

    if (lambda_offset.size() > 0) {
        gpuErrchk(
            cudaMemcpy(d_lambda_offset_, &lambda_offset[0], A_ * sizeof(*d_lambda_offset_), cudaMemcpyHostToDevice));
    } else {
        // can't memset this
        initializeArray(A_, d_lambda_offset_, 1);
    }
};

template <typename RealType> HarmonicAngle<RealType>::~HarmonicAngle() {
    gpuErrchk(cudaFree(d_angle_idxs_));
    gpuErrchk(cudaFree(d_lambda_mult_));
    gpuErrchk(cudaFree(d_lambda_offset_));
};

template <typename RealType>
void HarmonicAngle<RealType>::execute_device(
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
    int blocks = (A_ + tpb - 1) / tpb;

    if (A_ > 0) {
        k_harmonic_angle_inference<RealType, 3><<<blocks, tpb, 0, stream>>>(
            A_, d_x, d_p, lambda, d_lambda_mult_, d_lambda_offset_, d_angle_idxs_, d_du_dx, d_du_dp, d_du_dl, d_u);
        gpuErrchk(cudaPeekAtLastError());
    }
}

template class HarmonicAngle<double>;
template class HarmonicAngle<float>;

} // namespace timemachine
