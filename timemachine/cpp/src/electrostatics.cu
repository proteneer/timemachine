#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "electrostatics.hpp"
#include "gpu_utils.cuh"

#include "k_electrostatics.cuh"
#include "k_electrostatics_jvp.cuh"

namespace timemachine {

template <typename RealType>
Electrostatics<RealType>::Electrostatics(
    const std::vector<double> &charge_params, // [N]
    const std::vector<int> &exclusion_idxs, // [E,2]
    const std::vector<double> &charge_scales, // [E]
    const std::vector<int> &lambda_plane_idxs, // [N]
    const std::vector<int> &lambda_offset_idxs, // [N]
    double cutoff
) :  N_(charge_params.size()),
    cutoff_(cutoff),
    E_(exclusion_idxs.size()/2),
    nblist_(charge_params.size(), 3) {

    if(lambda_plane_idxs.size() != N_) {
        throw std::runtime_error("lambda plane idxs need to have size N");
    }

    if(lambda_offset_idxs.size() != N_) {
        throw std::runtime_error("lambda offset idxs need to have size N");
    }

    if(charge_scales.size()*2 != exclusion_idxs.size()) {
        throw std::runtime_error("charge scale idxs size not half of exclusion size!");
    }

    gpuErrchk(cudaMalloc(&d_lambda_plane_idxs_, N_*sizeof(*d_lambda_plane_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_plane_idxs_, &lambda_plane_idxs[0], N_*sizeof(*d_lambda_plane_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_*sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_*sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_charge_scales_, E_*sizeof(*d_charge_scales_)));
    gpuErrchk(cudaMemcpy(d_charge_scales_, &charge_scales[0], E_*sizeof(*d_charge_scales_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_charge_params_, N_*sizeof(*d_charge_params_)));
    gpuErrchk(cudaMemcpy(d_charge_params_, &charge_params[0], N_*sizeof(*d_charge_params_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_du_dcharge_primals_, N_*sizeof(*d_du_dcharge_primals_)));
    gpuErrchk(cudaMemset(d_du_dcharge_primals_, 0, N_*sizeof(*d_du_dcharge_primals_)));

    gpuErrchk(cudaMalloc(&d_du_dcharge_tangents_, N_*sizeof(*d_du_dcharge_tangents_)));
    gpuErrchk(cudaMemset(d_du_dcharge_tangents_, 0, N_*sizeof(*d_du_dcharge_tangents_)));


};

template <typename RealType>
Electrostatics<RealType>::~Electrostatics() {
    gpuErrchk(cudaFree(d_charge_params_));
    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_charge_scales_));
    gpuErrchk(cudaFree(d_lambda_plane_idxs_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));

    gpuErrchk(cudaFree(d_du_dcharge_primals_));
    gpuErrchk(cudaFree(d_du_dcharge_tangents_));
};


template <typename RealType>
void Electrostatics<RealType>::get_du_dcharge_primals(double *buf) {
    gpuErrchk(cudaMemcpy(buf, d_du_dcharge_primals_, N_*sizeof(*d_du_dcharge_primals_), cudaMemcpyDeviceToHost));
}

template <typename RealType>
void Electrostatics<RealType>::get_du_dcharge_tangents(double *buf) {
    gpuErrchk(cudaMemcpy(buf, d_du_dcharge_tangents_, N_*sizeof(*d_du_dcharge_tangents_), cudaMemcpyDeviceToHost));
}

template <typename RealType>
void Electrostatics<RealType>::execute_lambda_inference_device(
    const int N,
    const double *d_coords_primals,
    const double lambda_primal,
    unsigned long long *d_out_coords_primals,
    double *d_out_lambda_primals,
    double *d_out_energy_primal,
    cudaStream_t stream) {

    if(N != N_) {
        throw std::runtime_error("N != N_");
    }

    const int tpb = 32;
    const int B = (N_+tpb-1)/tpb;
    const int D = 3;

    // its safe for us to build a neighborlist in a lower dimension.
    nblist_.compute_block_bounds(N_, D, d_coords_primals, stream);

    gpuErrchk(cudaPeekAtLastError());

    dim3 dimGrid(B, B, 1); // x, y, z dims
    dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

    auto start = std::chrono::high_resolution_clock::now();

    // these can be ran in two streams later on
    k_electrostatics_inference<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_coords_primals,

        lambda_primal,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        d_charge_params_,
        cutoff_,
        nblist_.get_block_bounds_ctr(),
        nblist_.get_block_bounds_ext(),
        d_out_coords_primals,
        d_out_lambda_primals,
        d_out_energy_primal
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if(E_ > 0) {
        k_electrostatics_exclusion_inference<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
            E_,
            d_coords_primals,
            lambda_primal,
            d_lambda_plane_idxs_,
            d_lambda_offset_idxs_,
            d_exclusion_idxs_,
            d_charge_scales_,
            d_charge_params_,
            cutoff_,
            d_out_coords_primals,
            d_out_lambda_primals,
            d_out_energy_primal
        );
        // cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }
}


template <typename RealType>
void Electrostatics<RealType>::execute_lambda_jvp_device(
    const int N,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double lambda_primal,
    const double lambda_tangent,
    double *d_out_coords_primals,
    double *d_out_coords_tangents,
    cudaStream_t stream) {

    if(N != N_) {
        throw std::runtime_error("N != N_");
    }

    const int tpb = 32;
    const int B = (N_+tpb-1)/tpb;
    const int D = 3;

    nblist_.compute_block_bounds(N_, D, d_coords_primals, stream);

    gpuErrchk(cudaPeekAtLastError());

    dim3 dimGrid(B, B, 1); // x, y, z dims
    dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

    auto start = std::chrono::high_resolution_clock::now();

    k_electrostatics_jvp<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_coords_primals,
        d_coords_tangents,
        lambda_primal,
        lambda_tangent,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        d_charge_params_,
        cutoff_,
        nblist_.get_block_bounds_ctr(),
        nblist_.get_block_bounds_ext(),
        d_out_coords_primals,
        d_out_coords_tangents,
        d_du_dcharge_primals_,
        d_du_dcharge_tangents_
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if(E_ > 0) {
        k_electrostatics_exclusion_jvp<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
            E_,
            d_coords_primals,
            d_coords_tangents,
            lambda_primal,
            lambda_tangent,
            d_lambda_plane_idxs_,
            d_lambda_offset_idxs_,
            d_exclusion_idxs_,
            d_charge_scales_,
            d_charge_params_,
            cutoff_,
            d_out_coords_primals,
            d_out_coords_tangents,
            d_du_dcharge_primals_,
            d_du_dcharge_tangents_
        );            

        // cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }



};

template class Electrostatics<double>;
template class Electrostatics<float>;

} // namespace timemachine