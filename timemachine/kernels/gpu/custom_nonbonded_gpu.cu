#include <stdexcept>

#include "potential.hpp"
#include "custom_nonbonded_gpu.hpp"
#include "k_lennard_jones.cuh"
#include "k_electrostatics.cuh"
#include "kernel_utils.cuh"

#include <chrono>  // for high_resolution_clock
#include <iostream>
namespace timemachine {

template <typename RealType>
LennardJones<RealType>::LennardJones(
    std::vector<RealType> scale_matrix,
    std::vector<int> param_idxs
) {

    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_scale_matrix_, scale_matrix.size()*sizeof(*d_scale_matrix_)));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_scale_matrix_, &scale_matrix[0], scale_matrix.size()*sizeof(*d_scale_matrix_), cudaMemcpyHostToDevice));

};

template <typename RealType>
LennardJones<RealType>::~LennardJones() {
    gpuErrchk(cudaFree(d_param_idxs_));
    gpuErrchk(cudaFree(d_scale_matrix_));
};


template <typename RealType>
void LennardJones<RealType>::derivatives_device(
    const int num_confs,
    const int num_atoms,
    const RealType *d_coords,
    const RealType *d_params,
    RealType *d_E,
    RealType *d_dE_dx,
    RealType *d_d2E_dx2,
    // parameter derivatives
    const int num_dp,
    const int *d_param_gather_idxs,
    RealType *d_dE_dp,
    RealType *d_d2E_dxdp) const {

    const auto C = num_confs;
    const auto N = num_atoms;

    int tpb = 32;
    int n_blocks = (num_atoms + tpb - 1) / tpb;
    int dim_y = 1;

    dim3 dimBlock(tpb);
    dim3 dimGrid(n_blocks, dim_y, C); // x, y, z dims

    auto start = std::chrono::high_resolution_clock::now();
    k_lennard_jones<<<dimGrid, dimBlock>>>(
        N,
        d_coords,
        d_params,
        d_scale_matrix_,
        d_param_idxs_,
        d_E,
        d_dE_dx,
        d_d2E_dx2,
        // parameter derivatives
        num_dp,
        d_param_gather_idxs,
        d_dE_dp,
        d_d2E_dxdp
    );
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Custom LennardJones Elapsed time: " << elapsed.count() << " s\n";

    gpuErrchk(cudaPeekAtLastError());

};

template class LennardJones<float>;
template class LennardJones<double>;


template <typename RealType>
Electrostatics<RealType>::Electrostatics(
    std::vector<RealType> scale_matrix,
    std::vector<int> param_idxs
) {

    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_scale_matrix_, scale_matrix.size()*sizeof(*d_scale_matrix_)));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_scale_matrix_, &scale_matrix[0], scale_matrix.size()*sizeof(*d_scale_matrix_), cudaMemcpyHostToDevice));

};

template <typename RealType>
Electrostatics<RealType>::~Electrostatics() {
    gpuErrchk(cudaFree(d_param_idxs_));
    gpuErrchk(cudaFree(d_scale_matrix_));
};


template <typename RealType>
void Electrostatics<RealType>::derivatives_device(
    const int num_confs,
    const int num_atoms,
    const RealType *d_coords,
    const RealType *d_params,
    RealType *d_E,
    RealType *d_dE_dx,
    RealType *d_d2E_dx2,
    // parameter derivatives
    const int num_dp,
    const int *d_param_gather_idxs,
    RealType *d_dE_dp,
    RealType *d_d2E_dxdp) const {

    const auto C = num_confs;
    const auto N = num_atoms;

    int tpb = 32;
    int n_blocks = (num_atoms + tpb - 1) / tpb;
    int dim_y = 1;

    dim3 dimBlock(tpb);
    dim3 dimGrid(n_blocks, dim_y, C); // x, y, z dims

    auto start = std::chrono::high_resolution_clock::now();
    k_electrostatics<<<dimGrid, dimBlock>>>(
        N,
        d_coords,
        d_params,
        d_scale_matrix_,
        d_param_idxs_,
        d_E,
        d_dE_dx,
        d_d2E_dx2,
        // parameter derivatives
        num_dp,
        d_param_gather_idxs,
        d_dE_dp,
        d_d2E_dxdp
    );
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Custom Electrostatics Elapsed time: " << elapsed.count() << " s\n";

    gpuErrchk(cudaPeekAtLastError());

};

template class Electrostatics<float>;
template class Electrostatics<double>;

}