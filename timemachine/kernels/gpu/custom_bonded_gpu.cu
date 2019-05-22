#include "custom_bonded_gpu.hpp"
#include "harmonic_bond_impl.cuh"

#include <ctime>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


namespace timemachine {

template <typename RealType>
HarmonicBond<RealType>::HarmonicBond(
    std::vector<int> bond_idxs,
    std::vector<int> param_idxs
) : n_bonds_(bond_idxs.size()/2) {

    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_bond_idxs_, bond_idxs.size()*sizeof(*d_bond_idxs_)));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], bond_idxs.size()*sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType>
HarmonicBond<RealType>::~HarmonicBond() {
    gpuErrchk(cudaFree(d_bond_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
};

template <typename RealType>
void HarmonicBond<RealType>::derivatives_host(
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
    const auto B = n_bonds_;

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

    int tpb = 32;
    int n_blocks = (B + tpb - 1) / tpb;
    int dim_y = P;

    // we don't need the other derivatives if we don't need
    // parameter derivatives
    if(d_dE_dp == nullptr && d_d2E_dxdp == nullptr) {
        dim_y = 1;
    }

    dim3 dimBlock(tpb);
    dim3 dimGrid(n_blocks, dim_y); // x, y

    harmonic_bond_derivatives<<<dimGrid, dimBlock>>>(
        N,
        P,
        d_coords,
        d_params,
        d_dxdps,
        B,
        d_bond_idxs_,
        d_param_idxs_,
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

};

template class HarmonicBond<float>;
template class HarmonicBond<double>;

} // namespace timemachine