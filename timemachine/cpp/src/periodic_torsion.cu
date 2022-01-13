#include "gpu_utils.cuh"
#include "k_periodic_torsion.cuh"
#include "periodic_torsion.hpp"
#include <chrono>
#include <complex>
#include <iostream>
#include <vector>

namespace timemachine {

template <typename RealType>
PeriodicTorsion<RealType>::PeriodicTorsion(
    const std::vector<int> &torsion_idxs, // [A, 4]
    const std::vector<int> &lambda_mult,
    const std::vector<int> &lambda_offset)
    : T_(torsion_idxs.size() / 4) {

    if (torsion_idxs.size() % 4 != 0) {
        throw std::runtime_error("torsion_idxs.size() must be exactly 4*k");
    }

    if (lambda_mult.size() > 0 && lambda_mult.size() != T_) {
        throw std::runtime_error("bad lambda_mult size()");
    }

    if (lambda_offset.size() > 0 && lambda_offset.size() != T_) {
        throw std::runtime_error("bad lambda_offset size()");
    }

    for (int a = 0; a < T_; a++) {
        auto i = torsion_idxs[a * 4 + 0];
        auto j = torsion_idxs[a * 4 + 1];
        auto k = torsion_idxs[a * 4 + 2];
        auto l = torsion_idxs[a * 4 + 3];
        if (i == j || i == k || i == l || j == k || j == l || k == l) {
            throw std::runtime_error("torsion quads must be unique");
        }
    }

    gpuErrchk(cudaMalloc(&d_torsion_idxs_, T_ * 4 * sizeof(*d_torsion_idxs_)));
    gpuErrchk(cudaMemcpy(d_torsion_idxs_, &torsion_idxs[0], T_ * 4 * sizeof(*d_torsion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_mult_, T_ * sizeof(*d_lambda_mult_)));
    gpuErrchk(cudaMalloc(&d_lambda_offset_, T_ * sizeof(*d_lambda_offset_)));

    if (lambda_mult.size() > 0) {
        gpuErrchk(cudaMemcpy(d_lambda_mult_, &lambda_mult[0], T_ * sizeof(*d_lambda_mult_), cudaMemcpyHostToDevice));
    } else {
        initializeArray(T_, d_lambda_mult_, 0);
    }

    if (lambda_offset.size() > 0) {
        gpuErrchk(
            cudaMemcpy(d_lambda_offset_, &lambda_offset[0], T_ * sizeof(*d_lambda_offset_), cudaMemcpyHostToDevice));
    } else {
        // can't memset this
        initializeArray(T_, d_lambda_offset_, 1);
    }
};

template <typename RealType> PeriodicTorsion<RealType>::~PeriodicTorsion() {
    gpuErrchk(cudaFree(d_torsion_idxs_));
    gpuErrchk(cudaFree(d_lambda_mult_));
    gpuErrchk(cudaFree(d_lambda_offset_));
};

template <typename RealType>
void PeriodicTorsion<RealType>::execute_device(
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
    int blocks = (T_ + tpb - 1) / tpb;

    const int D = 3;

    if (blocks > 0) {
        k_periodic_torsion<RealType, D><<<blocks, tpb, 0, stream>>>(
            T_, d_x, d_p, lambda, d_lambda_mult_, d_lambda_offset_, d_torsion_idxs_, d_du_dx, d_du_dp, d_du_dl, d_u);
        gpuErrchk(cudaPeekAtLastError());
    }
};

template class PeriodicTorsion<double>;
template class PeriodicTorsion<float>;

} // namespace timemachine
