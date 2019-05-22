#include "potential.hpp"
#include "kernel_utils.cuh"

namespace timemachine {

template<typename RealType>
void Potential<RealType>::derivatives_host(
    const int num_atoms,
    const int num_params,
    const RealType *h_coords,
    const RealType *h_params,
    const RealType *h_dxdps,
    RealType *h_E,
    RealType *h_dE_dp,
    RealType *h_dE_dx,
    RealType *h_d2E_dxdp) const {

    const auto N = num_atoms;
    const auto P = num_params;

    RealType* d_coords = nullptr;
    RealType* d_params = nullptr;
    RealType* d_dxdps = nullptr;
    RealType* d_E = nullptr;
    RealType* d_dE_dp = nullptr;
    RealType* d_dE_dx = nullptr;
    RealType* d_d2E_dxdp = nullptr;

    gpuErrchk(cudaMalloc((void**)&d_coords, N*3*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_params, P*sizeof(RealType)));

    gpuErrchk(cudaMemcpy(d_coords, h_coords, N*3*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_params, h_params, P*sizeof(RealType), cudaMemcpyHostToDevice));

    if(h_dxdps != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_dxdps, P*N*3*sizeof(RealType)));
        gpuErrchk(cudaMemcpy(d_dxdps, h_dxdps, P*N*3*sizeof(RealType), cudaMemcpyHostToDevice));
    }
    if(h_E != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_E, sizeof(RealType)));
    }
    if(h_dE_dp != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_dE_dp, P*sizeof(RealType)));
    }
    if(h_dE_dx != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_dE_dx, N*3*sizeof(RealType)));
    }
    if(h_d2E_dxdp != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_d2E_dxdp, P*N*3*sizeof(RealType)));
    }

    this->derivatives_device(
        N,
        P,
        d_coords,
        d_params,
        d_dxdps,
        d_E,
        d_dE_dp,
        d_dE_dx,
        d_d2E_dxdp
    );

    gpuErrchk(cudaPeekAtLastError());

    if(h_E != nullptr) {
        gpuErrchk(cudaMemcpy(h_E, d_E, sizeof(RealType), cudaMemcpyDeviceToHost));
    }
    if(h_dE_dp != nullptr) {
        gpuErrchk(cudaMemcpy(h_dE_dp, d_dE_dp, P*sizeof(RealType), cudaMemcpyDeviceToHost));
    }
    if(h_dE_dx != nullptr) {
        gpuErrchk(cudaMemcpy(h_dE_dx, d_dE_dx, N*3*sizeof(RealType), cudaMemcpyDeviceToHost));
    }
    if(h_d2E_dxdp != nullptr) {
        gpuErrchk(cudaMemcpy(h_d2E_dxdp, d_d2E_dxdp, P*N*3*sizeof(RealType), cudaMemcpyDeviceToHost));
    }

    cudaFree(d_coords);
    cudaFree(d_params);
    cudaFree(d_dxdps);
    cudaFree(d_E);
    cudaFree(d_dE_dp);
    cudaFree(d_dE_dx);
    cudaFree(d_d2E_dxdp);

}

template class Potential<float>;
template class Potential<double>;

}