#include "potential.hpp"
#include "kernel_utils.cuh"

namespace timemachine {

template<typename RealType>
void Potential<RealType>::derivatives_host(
    const int num_atoms,
    const int num_params,
    const RealType *h_coords, // not null
    const RealType *h_params, // not null
    RealType *h_E, // not null
    RealType *h_dE_dx,
    // parameter derivatives
    const RealType *h_dx_dp,
    const int *h_dp_idxs, // not null but can be size zero
    const int num_dp_idxs,
    RealType *h_dE_dp,
    RealType *h_d2E_dxdp) const {

    const auto N = num_atoms;
    const auto P = num_params;
    const auto DP = num_dp_idxs;

    RealType* d_coords = nullptr;
    RealType* d_params = nullptr;
    RealType* d_dx_dp = nullptr;

    int* d_dp_idxs = nullptr;

    RealType* d_E = nullptr;
    RealType* d_dE_dx = nullptr;
    RealType* d_dE_dp = nullptr;
    RealType* d_d2E_dxdp = nullptr;

    gpuErrchk(cudaMalloc((void**)&d_coords, N*3*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_params, P*sizeof(RealType)));

    gpuErrchk(cudaMemcpy(d_coords, h_coords, N*3*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_params, h_params, P*sizeof(RealType), cudaMemcpyHostToDevice));

    if(num_dp_idxs > 0) {
        // the device function always take in a nullptr if this is of size zero
        gpuErrchk(cudaMalloc((void**)&d_dp_idxs, num_dp_idxs*sizeof(int)));
        gpuErrchk(cudaMemcpy(d_dp_idxs, h_dp_idxs, num_dp_idxs*sizeof(int), cudaMemcpyHostToDevice));
    }

    if(h_dx_dp != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_dx_dp, DP*N*3*sizeof(RealType)));
        gpuErrchk(cudaMemcpy(d_dx_dp, h_dx_dp, DP*N*3*sizeof(RealType), cudaMemcpyHostToDevice));
    }

    gpuErrchk(cudaMalloc((void**)&d_E, sizeof(RealType)));
    gpuErrchk(cudaMemset(d_E, 0, sizeof(RealType)));

    if(h_dE_dx != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_dE_dx, N*3*sizeof(RealType)));
        gpuErrchk(cudaMemset(d_dE_dx, 0, N*3*sizeof(RealType)));
    }
    if(h_dE_dp != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_dE_dp, DP*sizeof(RealType)));
        gpuErrchk(cudaMemset(d_dE_dp, 0, DP*sizeof(RealType)));
    }
    if(h_d2E_dxdp != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_d2E_dxdp, DP*N*3*sizeof(RealType)));
        gpuErrchk(cudaMemset(d_d2E_dxdp, 0, DP*N*3*sizeof(RealType)));
    }

    this->derivatives_device(
        N,
        P,
        d_coords,
        d_params,
        d_E, // never null
        d_dE_dx,

        // parameter derivatives
        d_dx_dp,
        d_dp_idxs,
        num_dp_idxs,
        d_dE_dp,
        d_d2E_dxdp
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(h_E, d_E, sizeof(RealType), cudaMemcpyDeviceToHost));

    if(h_dE_dx != nullptr) {
        gpuErrchk(cudaMemcpy(h_dE_dx, d_dE_dx, N*3*sizeof(RealType), cudaMemcpyDeviceToHost));        
    }
    if(h_dE_dp != nullptr) {
        gpuErrchk(cudaMemcpy(h_dE_dp, d_dE_dp, DP*sizeof(RealType), cudaMemcpyDeviceToHost));
    }
    if(h_d2E_dxdp != nullptr) {
        gpuErrchk(cudaMemcpy(h_d2E_dxdp, d_d2E_dxdp, DP*N*3*sizeof(RealType), cudaMemcpyDeviceToHost));
    }

    cudaFree(d_coords);
    cudaFree(d_params);
    cudaFree(d_dx_dp);
    cudaFree(d_dp_idxs);
    cudaFree(d_E);
    cudaFree(d_dE_dp);
    cudaFree(d_dE_dx);
    cudaFree(d_d2E_dxdp);

}

template class Potential<float>;
template class Potential<double>;

}