#include "potential.hpp"
#include "kernel_utils.cuh"

namespace timemachine {

template<typename RealType>
void Potential<RealType>::derivatives_host(
    const int num_confs,
    const int num_atoms,
    const int num_params,
    const RealType *h_coords,
    const RealType *h_params,
    RealType *h_E,
    RealType *h_dE_dx,
    RealType *h_d2E_dx2,
    // parameter derivatives
    const int num_dp,
    const int *h_param_gather_idxs,
    RealType *h_dE_dp,
    RealType *h_d2E_dxdp) const {

    const auto C = num_confs;
    const auto N = num_atoms;
    const auto P = num_params;
    const auto DP = num_dp;

    RealType* d_coords = nullptr;
    RealType* d_params = nullptr;
    int* d_param_gather_idxs = nullptr;

    RealType* d_E = nullptr;
    RealType* d_dE_dx = nullptr;
    RealType* d_d2E_dx2 = nullptr;

    RealType* d_dE_dp = nullptr;
    RealType* d_d2E_dxdp = nullptr;

    gpuErrchk(cudaMalloc((void**)&d_coords, C*N*3*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_params, P*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_param_gather_idxs, P*sizeof(int)));

    gpuErrchk(cudaMemcpy(d_coords, h_coords, C*N*3*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_params, h_params, P*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_param_gather_idxs, h_param_gather_idxs, P*sizeof(int), cudaMemcpyHostToDevice));

    if(h_E != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_E, C*sizeof(RealType)));
        gpuErrchk(cudaMemset(d_E, 0, C*sizeof(RealType)));        
    }

    if(h_dE_dx != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_dE_dx, C*N*3*sizeof(RealType)));
        gpuErrchk(cudaMemset(d_dE_dx, 0, C*N*3*sizeof(RealType)));
    }

    if(h_d2E_dx2 != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_d2E_dx2, C*N*3*N*3*sizeof(RealType)));
        gpuErrchk(cudaMemset(d_d2E_dx2, 0, C*N*3*N*3*sizeof(RealType)));     
    }

    if(h_dE_dp != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_dE_dp, C*DP*sizeof(RealType)));
        gpuErrchk(cudaMemset(d_dE_dp, 0, C*DP*sizeof(RealType)));
    }
    if(h_d2E_dxdp != nullptr) {
        gpuErrchk(cudaMalloc((void**)&d_d2E_dxdp, C*DP*N*3*sizeof(RealType)));
        gpuErrchk(cudaMemset(d_d2E_dxdp, 0, C*DP*N*3*sizeof(RealType)));
    }

    this->derivatives_device(
        C,
        N,
        d_coords,
        d_params,
        d_E,
        d_dE_dx,
        d_d2E_dx2,

        // parameter derivatives
        num_dp,
        d_param_gather_idxs,
        d_dE_dp,
        d_d2E_dxdp
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(h_E, d_E, C*sizeof(RealType), cudaMemcpyDeviceToHost));

    if(h_E != nullptr) {
        gpuErrchk(cudaMemcpy(h_E, d_E, C*sizeof(RealType), cudaMemcpyDeviceToHost));
    }
    if(h_dE_dx != nullptr) {
        gpuErrchk(cudaMemcpy(h_dE_dx, d_dE_dx, C*N*3*sizeof(RealType), cudaMemcpyDeviceToHost));        
    }
    if(h_d2E_dx2 != nullptr) {
        gpuErrchk(cudaMemcpy(h_d2E_dx2, d_d2E_dx2, C*N*3*N*3*sizeof(RealType), cudaMemcpyDeviceToHost));        
    }
    if(h_dE_dp != nullptr) {
        gpuErrchk(cudaMemcpy(h_dE_dp, d_dE_dp, C*DP*sizeof(RealType), cudaMemcpyDeviceToHost));
    }
    if(h_d2E_dxdp != nullptr) {
        gpuErrchk(cudaMemcpy(h_d2E_dxdp, d_d2E_dxdp, C*DP*N*3*sizeof(RealType), cudaMemcpyDeviceToHost));
    }

    cudaFree(d_coords);
    cudaFree(d_params);

    cudaFree(d_E);
    cudaFree(d_dE_dx);
    cudaFree(d_d2E_dx2);

    cudaFree(d_param_gather_idxs);
    cudaFree(d_dE_dp);
    cudaFree(d_d2E_dxdp);

}

template class Potential<float>;
template class Potential<double>;

}