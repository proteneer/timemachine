#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "restraint.hpp"
#include "gpu_utils.cuh"
#include "k_restraint.cuh"

namespace timemachine {

template <typename RealType>
Restraint<RealType>::Restraint(
    const std::vector<int> &bond_idxs, // [N]
    const std::vector<int> &param_idxs,
    const std::vector<int> &lambda_flags
) : B_(bond_idxs.size()/2) {


    if(bond_idxs.size() % 2 != 0) {
        throw std::runtime_error("bond_idxs.size() must be exactly 2*k");
    }


    if(param_idxs.size() % 3 != 0) {
        throw std::runtime_error("param_idxs.size() must be exactly 2*k");
    }

    for(int b=0; b < B_; b++) {
        auto src = bond_idxs[b*2+0];
        auto dst = bond_idxs[b*2+1];
        if(src == dst) {
            throw std::runtime_error("src == dst");
        }
    }

    gpuErrchk(cudaMalloc(&d_lambda_flags_, B_*sizeof(*d_lambda_flags_)));
    gpuErrchk(cudaMemcpy(d_lambda_flags_, &lambda_flags[0], B_*sizeof(*d_lambda_flags_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_bond_idxs_, B_*2*sizeof(*d_bond_idxs_)));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], B_*2*sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_bond_idxs_, B_*2*sizeof(*d_bond_idxs_)));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], B_*2*sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_param_idxs_, B_*3*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], B_*3*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType>
Restraint<RealType>::~Restraint() {
    gpuErrchk(cudaFree(d_bond_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
    gpuErrchk(cudaFree(d_lambda_flags_));
};

template <typename RealType>
void Restraint<RealType>::execute_lambda_inference_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_params_primals,
    const double lambda_primal,
    unsigned long long *d_out_coords_primals, // du/dx
    double *d_out_lambda_primals, // du/dl
    double *d_out_energy_primal, // U
    cudaStream_t stream) {

    int tpb = 32;
    int blocks = (B_+tpb-1)/tpb;
    k_restraint_inference<RealType><<<blocks, tpb, 0, stream>>>(
        B_,
        d_coords_primals,
        d_params_primals,
        lambda_primal,
        d_bond_idxs_,
        d_param_idxs_,
        d_lambda_flags_,
        d_out_coords_primals,
        d_out_lambda_primals,
        d_out_energy_primal
    );
    gpuErrchk(cudaPeekAtLastError());

    // auto finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = finish - start;
    // std::cout << "Restraint Elapsed time: " << elapsed.count() << " s\n";

};

template <typename RealType>
void Restraint<RealType>::execute_lambda_jvp_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double *d_params_primals,
    const double lambda_primal, // unused
    const double lambda_tangent, // unused
    double *d_out_coords_primals,
    double *d_out_coords_tangents,
    double *d_out_params_primals,
    double *d_out_params_tangents,
    cudaStream_t stream) {

    // int tpb = 32;
    // int blocks = (B_+tpb-1)/tpb;

    // k_flat_bottom_jvp<RealType><<<blocks, tpb, 0, stream>>>(
    //     B_,
    //     d_coords_primals,
    //     d_coords_tangents,
    //     d_params_primals,
    //     lambda_primal,
    //     lambda_tangent,
    //     d_bond_idxs_,
    //     d_param_idxs_,
    //     d_lambda_flags_,
    //     d_out_coords_primals,
    //     d_out_coords_tangents,
    //     d_out_params_primals,
    //     d_out_params_tangents
    // );

    // // cudaDeviceSynchronize();
    // gpuErrchk(cudaPeekAtLastError());

    // // auto finish = std::chrono::high_resolution_clock::now();
    // // std::chrono::duration<double> elapsed = finish - start;
    // // std::cout << "Restraint Elapsed time: " << elapsed.count() << " s\n";

}

template class Restraint<double>;
template class Restraint<float>;

} // namespace timemachine