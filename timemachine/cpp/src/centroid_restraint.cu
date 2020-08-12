#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "centroid_restraint.hpp"
#include "gpu_utils.cuh"
#include "k_centroid_restraint.cuh"

namespace timemachine {

template <typename RealType>
CentroidRestraint<RealType>::CentroidRestraint(
    const std::vector<int> &group_a_idxs,
    const std::vector<int> &group_b_idxs,
    const std::vector<double> &masses,
    const double kb,
    const double b0,
    const int lambda_flag,
    const int lambda_offset
) : N_(masses.size()),
    N_A_(group_a_idxs.size()),
    N_B_(group_b_idxs.size()),
    kb_(kb),
    b0_(b0),
    lambda_flag_(lambda_flag),
    lambda_offset_(lambda_offset) {

    for(int i=0; i < group_a_idxs.size(); i++) {
        if(group_a_idxs[i] >= N_ || group_a_idxs[i] < 0) {
            throw std::runtime_error("Invalid group_a_idx!");
        }
    }

    for(int i=0; i < group_b_idxs.size(); i++) {
        if(group_b_idxs[i] >= N_ || group_b_idxs[i] < 0) {
            throw std::runtime_error("Invalid group_a_idx!");
        }
    }

    gpuErrchk(cudaMalloc(&d_masses_, N_*sizeof(*d_masses_)));
    gpuErrchk(cudaMemcpy(d_masses_, &masses[0], N_*sizeof(*d_masses_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_group_a_idxs_, N_A_*sizeof(*d_group_a_idxs_)));
    gpuErrchk(cudaMemcpy(d_group_a_idxs_, &group_a_idxs[0], N_A_*sizeof(*d_group_a_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_group_b_idxs_, N_B_*sizeof(*d_group_b_idxs_)));
    gpuErrchk(cudaMemcpy(d_group_b_idxs_, &group_b_idxs[0], N_B_*sizeof(*d_group_b_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType>
CentroidRestraint<RealType>::~CentroidRestraint() {
    gpuErrchk(cudaFree(d_masses_));
    gpuErrchk(cudaFree(d_group_a_idxs_));
    gpuErrchk(cudaFree(d_group_b_idxs_));
};


template <typename RealType>
void CentroidRestraint<RealType>::execute_lambda_inference_device(
    const int N,
    const double *d_coords_primals,
    const double lambda_primal,
    unsigned long long *d_out_coords_primals, // du/dx
    double *d_out_lambda_primals, // du/dl
    double *d_out_energy_primal, // U
    cudaStream_t stream) {

    int tpb = 32;
    // int blocks = (B_+tpb-1)/tpb;
    // printf("LAMBDA PRIMAL %f\n", lambda_primal);

    k_centroid_restraint_inference<RealType><<<1, tpb, 0, stream>>>(
        N_,
        d_coords_primals,
        lambda_primal,
        lambda_flag_,
        lambda_offset_,
        d_group_a_idxs_,
        d_group_b_idxs_,
        N_A_,
        N_B_,
        d_masses_,
        kb_,
        b0_,
        d_out_coords_primals,
        d_out_lambda_primals,
        d_out_energy_primal
    );
    gpuErrchk(cudaPeekAtLastError());

    // auto finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = finish - start;
    // std::cout << "CentroidRestraint Elapsed time: " << elapsed.count() << " s\n";

};

template <typename RealType>
void CentroidRestraint<RealType>::execute_lambda_jvp_device(
    const int N,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double lambda_primal,
    const double lambda_tangent,
    double *d_out_coords_primals,
    double *d_out_coords_tangents,
    cudaStream_t stream) {

    int tpb = 32;

    k_centroid_restraint_jvp<RealType><<<1, tpb, 0, stream>>>(
        N_,
        d_coords_primals,
        d_coords_tangents,
        lambda_primal,
        lambda_tangent,
        lambda_flag_,
        lambda_offset_,
        d_group_a_idxs_,
        d_group_b_idxs_,
        N_A_,
        N_B_,
        d_masses_,
        kb_,
        b0_,
        d_out_coords_primals,
        d_out_coords_tangents
    );

    // // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // // auto finish = std::chrono::high_resolution_clock::now();
    // // std::chrono::duration<double> elapsed = finish - start;
    // // std::cout << "CentroidRestraint Elapsed time: " << elapsed.count() << " s\n";

};

template class CentroidRestraint<double>;
template class CentroidRestraint<float>;

} // namespace timemachine