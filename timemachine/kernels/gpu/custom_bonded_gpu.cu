#include <stdexcept>

#include "custom_bonded_gpu.hpp"
#include "k_harmonic_bond.cuh"
#include "k_harmonic_angle.cuh"
#include "kernel_utils.cuh"

#include <iostream>

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
void HarmonicBond<RealType>::derivatives_device(
        const int num_confs,
        const int num_atoms,
        const int num_params,
        const RealType *d_coords,
        const RealType *d_params,
        RealType *d_E,
        RealType *d_dE_dx,

        const RealType *d_dx_dp,
        const int *d_dp_idxs,
        const int num_dp_idxs,
        RealType *d_dE_dp,
        RealType *d_d2E_dxdp) const {

    const auto C = num_confs;
    const auto N = num_atoms;
    const auto P = num_params;
    const auto B = n_bonds_;

    int tpb = 32;
    int n_blocks = (B + tpb - 1) / tpb;
    int dim_y = 1;

    // zero dimension dim_ys are *not* allowed.
    if(num_dp_idxs == 0) {
        // inference mode
        dim_y = 1;
        if(d_dp_idxs != nullptr) {
            throw std::runtime_error("d_dp_idxs is not null but num_dp_idxs == 0");
        }
    } else {
        dim_y = num_dp_idxs;
    }

    dim3 dimBlock(tpb);
    dim3 dimGrid(n_blocks, dim_y, C); // x, y, z dims

    k_harmonic_bond_derivatives<<<dimGrid, dimBlock>>>(
        N,
        P,
        d_coords,
        d_params,
        B,
        d_bond_idxs_,
        d_param_idxs_,
        d_E,
        d_dE_dx,
        // parameter derivatives
        d_dx_dp,
        d_dp_idxs,
        d_dE_dp,
        d_d2E_dxdp
    );

    gpuErrchk(cudaPeekAtLastError());

};

template class HarmonicBond<float>;
template class HarmonicBond<double>;

template <typename RealType>
HarmonicAngle<RealType>::HarmonicAngle(
    std::vector<int> angle_idxs,
    std::vector<int> param_idxs
) : n_angles_(angle_idxs.size()/3) {

    gpuErrchk(cudaMalloc((void**)&d_param_idxs_, param_idxs.size()*sizeof(*d_param_idxs_)));
    gpuErrchk(cudaMalloc((void**)&d_angle_idxs_, angle_idxs.size()*sizeof(*d_angle_idxs_)));
    gpuErrchk(cudaMemcpy(d_param_idxs_, &param_idxs[0], param_idxs.size()*sizeof(*d_param_idxs_), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_angle_idxs_, &angle_idxs[0], angle_idxs.size()*sizeof(*d_angle_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType>
HarmonicAngle<RealType>::~HarmonicAngle() {
    gpuErrchk(cudaFree(d_angle_idxs_));
    gpuErrchk(cudaFree(d_param_idxs_));
};

template <typename RealType>
void HarmonicAngle<RealType>::derivatives_device(
        const int num_confs,
        const int num_atoms,
        const int num_params,
        const RealType *d_coords,
        const RealType *d_params,
        RealType *d_E,
        RealType *d_dE_dx,

        const RealType *d_dx_dp,
        const int *d_dp_idxs,
        const int num_dp_idxs,
        RealType *d_dE_dp,
        RealType *d_d2E_dxdp) const {

    const auto C = num_confs;
    const auto N = num_atoms;
    const auto P = num_params;

    int tpb = 32;
    int n_blocks = (n_angles_ + tpb - 1) / tpb;
    int dim_y = 1;

    // zero dimension dim_ys are *not* allowed.
    if(num_dp_idxs == 0) {
        // inference mode
        dim_y = 1;
        if(d_dp_idxs != nullptr) {
            throw std::runtime_error("d_dp_idxs is not null but num_dp_idxs == 0");
        }
    } else {
        dim_y = num_dp_idxs;
    }

    dim3 dimBlock(tpb);
    dim3 dimGrid(n_blocks, dim_y, C); // x, y, z

    k_harmonic_angle_derivatives<<<dimGrid, dimBlock>>>(
        N,
        P,
        d_coords,
        d_params,
        n_angles_,
        d_angle_idxs_,
        d_param_idxs_,
        d_E,
        d_dE_dx,
        // parameter derivatives
        d_dx_dp,
        d_dp_idxs,
        d_dE_dp,
        d_d2E_dxdp
    );

    gpuErrchk(cudaPeekAtLastError());

};


template class HarmonicAngle<float>;
template class HarmonicAngle<double>;

} // namespace timemachine