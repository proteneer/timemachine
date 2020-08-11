#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "boresch_like_restraint.hpp"
#include "gpu_utils.cuh"
#include "k_boresch.cuh"

namespace timemachine {

template <typename RealType>
BoreschLikeRestraint<RealType>::BoreschLikeRestraint(
    const std::vector<int> &bond_idxs,
    const std::vector<int> &angle_idxs,
    const std::vector<double> &bond_params,
    const std::vector<double> &angle_params,
    const int lambda_flag,
    const int lambda_offset
) : N_A_(angle_idxs.size()/3),
    N_B_(bond_idxs.size()/2),
    lambda_flag_(lambda_flag),
    lambda_offset_(lambda_offset) {

    if(bond_idxs.size() % 2 != 0) {
        throw std::runtime_error("Fatal on bond_idxs");
    }

    if(angle_idxs.size() % 3 != 0) {
        throw std::runtime_error("Fatal on angle_idxs");
    }

    gpuErrchk(cudaMalloc(&d_bond_idxs_, bond_idxs.size()*sizeof(*d_bond_idxs_)));
    gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0], bond_idxs.size()*sizeof(*d_bond_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_bond_params_, bond_params.size()*sizeof(*d_bond_params_)));
    gpuErrchk(cudaMemcpy(d_bond_params_, &bond_params[0], bond_params.size()*sizeof(*d_bond_params_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_angle_idxs_, angle_idxs.size()*sizeof(*d_angle_idxs_)));
    gpuErrchk(cudaMemcpy(d_angle_idxs_, &angle_idxs[0], angle_idxs.size()*sizeof(*d_angle_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_angle_params_, angle_params.size()*sizeof(*d_angle_params_)));
    gpuErrchk(cudaMemcpy(d_angle_params_, &angle_params[0], angle_params.size()*sizeof(*d_angle_params_), cudaMemcpyHostToDevice));

};

template <typename RealType>
BoreschLikeRestraint<RealType>::~BoreschLikeRestraint() {
    gpuErrchk(cudaFree(d_bond_idxs_));
    gpuErrchk(cudaFree(d_angle_idxs_));
    gpuErrchk(cudaFree(d_bond_params_));
    gpuErrchk(cudaFree(d_angle_params_));
};


template <typename RealType>
void BoreschLikeRestraint<RealType>::execute_lambda_inference_device(
    const int N,
    const double *d_coords_primals,
    const double lambda_primal,
    unsigned long long *d_out_coords_primals, // du/dx
    double *d_out_lambda_primals, // du/dl
    double *d_out_energy_primal, // U
    cudaStream_t stream) {

    int tpb = 32;
    int blocks = (N_A_+tpb-1)/tpb;

    auto start = std::chrono::high_resolution_clock::now();

    k_boresch_angle_inference<RealType, 3><<<blocks, tpb, 0, stream>>>(
        N_A_,
        d_coords_primals,
        lambda_primal,
        lambda_flag_,
        lambda_offset_,
        d_angle_params_,
        d_angle_idxs_,
        d_out_coords_primals,
        d_out_lambda_primals,
        d_out_energy_primal);

    blocks = (N_B_+tpb-1)/tpb;
    k_boresch_bond_inference<RealType><<<blocks, tpb, 0, stream>>>(
        N_B_,
        d_coords_primals,
        lambda_primal,
        lambda_flag_,
        lambda_offset_,
        d_bond_params_,
        d_bond_idxs_,
        d_out_coords_primals,
        d_out_lambda_primals,
        d_out_energy_primal);

    gpuErrchk(cudaPeekAtLastError());

    
};

template <typename RealType>
void BoreschLikeRestraint<RealType>::execute_lambda_jvp_device(
    const int N,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double lambda_primal, // unused
    const double lambda_tangent, // unused
    double *d_out_coords_primals,
    double *d_out_coords_tangents,
    cudaStream_t stream) {

    // int tpb = 32;
    // int blocks = (B_+tpb-1)/tpb;

    // k_BoreschLikeRestraint_jvp<RealType><<<blocks, tpb, 0, stream>>>(
    //     B_,
    //     d_coords_primals,
    //     d_coords_tangents,
    //     d_params_,
    //     lambda_primal,
    //     lambda_tangent,
    //     d_bond_idxs_,
    //     d_lambda_flags_,
    //     d_out_coords_primals,
    //     d_out_coords_tangents,
    //     d_du_dp_primals_,
    //     d_du_dp_tangents_
    // );

    // // cudaDeviceSynchronize();
    // gpuErrchk(cudaPeekAtLastError());

    // // auto finish = std::chrono::high_resolution_clock::now();
    // // std::chrono::duration<double> elapsed = finish - start;
    // // std::cout << "BoreschLikeRestraint Elapsed time: " << elapsed.count() << " s\n";

}

template class BoreschLikeRestraint<double>;
template class BoreschLikeRestraint<float>;

} // namespace timemachine