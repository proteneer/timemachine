#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "harmonic_bond.hpp"
#include "gpu_utils.cuh"
#include "k_bonded_deterministic.cuh"

namespace timemachine {

template <typename RealType, int D>
HarmonicBond<RealType, D>::HarmonicBond(
    const std::vector<int> &bond_idxs, // [B, 2]
    const std::vector<int> &param_idxs, // [B, 2]
    const std::vector<int> &lambda_idxs // [B]
) : B_(bond_idxs.size()/2) {

    // alloc lambda_idxs

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

    if(lambda_idxs.size() != B_) {
        throw std::runtime_error("lambda_idxs must have size B");
    }

    gpuErrchk(cudaMalloc(&d_lambda_idxs_, B_*sizeof(*d_lambda_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_idxs_, &lambda_idxs[0], B_*sizeof(*d_lambda_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_bond_idxs_, B_*2*sizeof(*d_bond_idxs_)));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], B_*2*sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_param_idxs_, B_*2*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], B_*2*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType, int D>
HarmonicBond<RealType, D>::~HarmonicBond() {
    gpuErrchk(cudaFree(d_bond_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
    gpuErrchk(cudaFree(d_lambda_idxs_));
};

template <typename RealType, int D>
void HarmonicBond<RealType, D>::execute_lambda_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double *d_params_primals,
    const double lambda_primal,
    const double lambda_tangent,
    unsigned long long *d_out_coords,
    double *d_out_du_dl,
    double *d_out_coords_tangents,
    double *d_out_params_tangents,
    cudaStream_t stream
) {


    int tpb = 32;
    int blocks = (B_+tpb-1)/tpb;


    // auto start = std::chrono::high_resolution_clock::now();
    if(d_coords_tangents == nullptr) {

        k_harmonic_bond_inference<RealType, 3><<<blocks, tpb, 0, stream>>>(
            B_,
            d_coords_primals,
            d_params_primals,
            lambda_primal,
            d_lambda_idxs_,
            d_bond_idxs_,
            d_param_idxs_,
            d_out_coords,
            d_out_du_dl
        );

        // cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "HarmonicBond Elapsed time: " << elapsed.count() << " s\n";

    } else {


        k_harmonic_bond_jvp<RealType, 3><<<blocks, tpb, 0, stream>>>(
            B_,
            d_coords_primals,
            d_coords_tangents,
            d_params_primals,
            lambda_primal,
            lambda_tangent,
            d_lambda_idxs_,
            d_bond_idxs_,
            d_param_idxs_,
            d_out_coords_tangents,
            d_out_params_tangents
        );

        // cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "HarmonicBond JVP Elapsed time: " << elapsed.count() << " s\n";

    }

}

template class HarmonicBond<double, 4>;
template class HarmonicBond<double, 3>;


template class HarmonicBond<float, 4>;
template class HarmonicBond<float, 3>;




} // namespace timemachine