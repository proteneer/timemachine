#include <iostream>

#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "potential.hpp"
#include "surreal.cuh"

namespace timemachine {

void Potential::execute_host(
    const int N,
    const int P,
    const double *h_x,           // [N,3]
    const double *h_p,           // [P,]
    const double *h_box,         // [3, 3]
    const double lambda,         // [1]
    unsigned long long *h_du_dx, // [N,3]
    unsigned long long *h_du_dp, // [P]
    unsigned long long *h_du_dl, //
    unsigned long long *h_u) {

    const int &D = Potential::D;

    double *d_x;
    double *d_p;
    double *d_box;

    gpuErrchk(cudaMalloc(&d_x, N * D * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_x, h_x, N * D * sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_p, P * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_p, h_p, P * sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_box, D * D * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_box, h_box, D * D * sizeof(double), cudaMemcpyHostToDevice));

    unsigned long long *d_du_dx = nullptr;
    unsigned long long *d_du_dp = nullptr;
    unsigned long long *d_du_dl = nullptr;
    unsigned long long *d_u = nullptr;

    // very important that these are initialized to zero since the kernels themselves just accumulate
    if (h_du_dx) {
        gpuErrchk(cudaMalloc(&d_du_dx, N * D * sizeof(unsigned long long)));
        gpuErrchk(cudaMemset(d_du_dx, 0, N * D * sizeof(unsigned long long)));
    }
    if (h_du_dp) {
        gpuErrchk(cudaMalloc(&d_du_dp, P * sizeof(unsigned long long)));
        gpuErrchk(cudaMemset(d_du_dp, 0, P * sizeof(unsigned long long)));
    }
    if (h_du_dl) {
        gpuErrchk(cudaMalloc(&d_du_dl, N * sizeof(*d_du_dl)));
        gpuErrchk(cudaMemset(d_du_dl, 0, N * sizeof(*d_du_dl)));
    }
    if (h_u) {
        gpuErrchk(cudaMalloc(&d_u, N * sizeof(*d_u)));
        gpuErrchk(cudaMemset(d_u, 0, N * sizeof(*d_u)));
    }

    this->execute_device(N, P, d_x, d_p, d_box, lambda, d_du_dx, d_du_dp, d_du_dl, d_u, static_cast<cudaStream_t>(0));

    // outputs
    if (h_du_dx) {
        gpuErrchk(cudaMemcpy(h_du_dx, d_du_dx, N * D * sizeof(*h_du_dx), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_du_dx));
    }
    if (h_du_dp) {
        gpuErrchk(cudaMemcpy(h_du_dp, d_du_dp, P * sizeof(*h_du_dp), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_du_dp));
    }
    if (h_du_dl) {
        gpuErrchk(cudaMemcpy(h_du_dl, d_du_dl, N * sizeof(*h_du_dl), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_du_dl));
    }
    if (h_u) {
        gpuErrchk(cudaMemcpy(h_u, d_u, N * sizeof(*h_u), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_u));
    }

    // inputs
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_p));
    gpuErrchk(cudaFree(d_box));
};

void Potential::execute_host_du_dx(
    const int N,
    const int P,
    const double *h_x,   // [N,3]
    const double *h_p,   // [P,]
    const double *h_box, // [3, 3]
    const double lambda, // [1]
    unsigned long long *h_du_dx) {

    const int &D = Potential::D;

    double *d_x;
    double *d_p;
    double *d_box;

    gpuErrchk(cudaMalloc(&d_x, N * D * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_x, h_x, N * D * sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_p, P * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_p, h_p, P * sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_box, D * D * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_box, h_box, D * D * sizeof(double), cudaMemcpyHostToDevice));

    unsigned long long *d_du_dx; // du/dx

    // very important that these are initialized to zero since the kernels themselves just accumulate
    gpuErrchk(cudaMalloc(&d_du_dx, N * D * sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_du_dx, 0, N * D * sizeof(unsigned long long)));

    this->execute_device(
        N, P, d_x, d_p, d_box, lambda, d_du_dx, nullptr, nullptr, nullptr, static_cast<cudaStream_t>(0));

    gpuErrchk(cudaMemcpy(h_du_dx, d_du_dx, N * D * sizeof(*h_du_dx), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_du_dx));
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_p));
    gpuErrchk(cudaFree(d_box));
};

void Potential::du_dp_fixed_to_float(const int N, const int P, const unsigned long long *du_dp, double *du_dp_float) {
    for (int i = 0; i < P; i++) {
        du_dp_float[i] = FIXED_TO_FLOAT<double>(du_dp[i]);
    }
}

} // namespace timemachine
