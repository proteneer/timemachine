#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "harmonic_bond.hpp"
#include "kernel_utils.cuh"
#include "k_bonded_deterministic.cuh"

namespace timemachine {

template <typename RealType, int D>
HarmonicBond<RealType, D>::HarmonicBond(
    const std::vector<int> &bond_idxs, // [N]
    const std::vector<int> &param_idxs
) : B_(bond_idxs.size()/2) {

    if(bond_idxs.size() % 2 != 0) {
        throw std::runtime_error("bond_idxs.size() must be exactly 2*k");
    }


    for(int b=0; b < B_; b++) {
        auto src = bond_idxs[b*2+0];
        auto dst = bond_idxs[b*2+1];
        if(src == dst) {
            throw std::runtime_error("src == dst");
        }
    }

    gpuErrchk(cudaMalloc(&d_bond_idxs_, B_*2*sizeof(*d_bond_idxs_)));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], B_*2*sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_param_idxs_, B_*2*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], B_*2*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType, int D>
HarmonicBond<RealType, D>::~HarmonicBond() {
    gpuErrchk(cudaFree(d_bond_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
};

template <typename RealType, int D>
void HarmonicBond<RealType, D>::execute_device(
    const int N,
    const int P,
    const RealType *d_coords,
    const RealType *d_coords_tangents,
    const RealType *d_params,
    unsigned long long *d_out_coords,
    RealType *d_out_coords_tangents,
    RealType *d_out_params_tangents
) {

    int tpb = 32;
    int blocks = (B_+tpb-1)/tpb;

    auto start = std::chrono::high_resolution_clock::now();
    if(d_coords_tangents == nullptr) {

        k_harmonic_bond_inference<RealType, D><<<blocks, tpb>>>(
            B_,
            d_coords,
            d_params,
            d_bond_idxs_,
            d_param_idxs_,
            d_out_coords
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "HarmonicBond Elapsed time: " << elapsed.count() << " s\n";

    } else {


        k_harmonic_bond_jvp<RealType, D><<<blocks, tpb>>>(
            B_,
            d_coords,
            d_coords_tangents,
            d_params,
            d_bond_idxs_,
            d_param_idxs_,
            d_out_coords_tangents,
            d_out_params_tangents
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "HarmonicBond JVP Elapsed time: " << elapsed.count() << " s\n";

    }


};

template class HarmonicBond<double, 4>;
template class HarmonicBond<double, 3>;


template class HarmonicBond<float, 4>;
template class HarmonicBond<float, 3>;




} // namespace timemachine