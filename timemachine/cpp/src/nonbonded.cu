#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "nonbonded.hpp"
#include "gpu_utils.cuh"

#include "k_nonbonded.cuh"
#include "k_nonbonded_jvp.cuh"

namespace timemachine {

template <typename RealType>
Nonbonded<RealType>::Nonbonded(
    const std::vector<int> &charge_param_idxs,
    const std::vector<int> &lj_param_idxs,
    const std::vector<int> &exclusion_idxs, // [E,2]
    const std::vector<int> &charge_scale_idxs, // [E]
    const std::vector<int> &lj_scale_idxs, // [E]
    const std::vector<int> &lambda_plane_idxs, // [N]
    const std::vector<int> &lambda_offset_idxs, // [N]
    double cutoff
) :  N_(charge_param_idxs.size()),
    cutoff_(cutoff),
    E_(charge_scale_idxs.size()),
    nblist_(charge_param_idxs.size(), 3) {

    if(lambda_plane_idxs.size() != N_) {
        throw std::runtime_error("lambda plane idxs need to have size N");
    }

    if(lambda_offset_idxs.size() != N_) {
        throw std::runtime_error("lambda offset idxs need to have size N");
    }

    if(charge_scale_idxs.size()*2 != exclusion_idxs.size()) {
        throw std::runtime_error("charge scale idxs size not half of exclusion size!");
    }

    if(charge_scale_idxs.size() != lj_scale_idxs.size()) {
        throw std::runtime_error("Charge scale idxs does not match LJ scale idxs!");
    }

    if(charge_param_idxs.size()*2 != lj_param_idxs.size()) {
        throw std::runtime_error("Charge param idxs not half of lj param idxs!");
    }

    gpuErrchk(cudaMalloc(&d_lambda_plane_idxs_, N_*sizeof(*d_lambda_plane_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_plane_idxs_, &lambda_plane_idxs[0], N_*sizeof(*d_lambda_plane_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_*sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_*sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_charge_scale_idxs_, E_*sizeof(*d_charge_scale_idxs_)));
    gpuErrchk(cudaMemcpy(d_charge_scale_idxs_, &charge_scale_idxs[0], E_*sizeof(*d_charge_scale_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lj_scale_idxs_, E_*sizeof(*d_lj_scale_idxs_)));
    gpuErrchk(cudaMemcpy(d_lj_scale_idxs_, &lj_scale_idxs[0], E_*sizeof(*d_lj_scale_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_charge_param_idxs_, N_*sizeof(*d_charge_param_idxs_)));
    gpuErrchk(cudaMemcpy(d_charge_param_idxs_, &charge_param_idxs[0], N_*sizeof(*d_charge_param_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_lj_param_idxs_, N_*2*sizeof(*d_lj_param_idxs_)));
    gpuErrchk(cudaMemcpy(d_lj_param_idxs_, &lj_param_idxs[0], N_*2*sizeof(*d_lj_param_idxs_), cudaMemcpyHostToDevice));

};

template <typename RealType>
Nonbonded<RealType>::~Nonbonded() {
    gpuErrchk(cudaFree(d_charge_param_idxs_));
    gpuErrchk(cudaFree(d_lj_param_idxs_));
    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_charge_scale_idxs_));
    gpuErrchk(cudaFree(d_lj_scale_idxs_));
    gpuErrchk(cudaFree(d_lambda_plane_idxs_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));
};


template <typename RealType>
void Nonbonded<RealType>::execute_lambda_inference_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_params_primals,
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
    k_nonbonded_inference<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_coords_primals,
        d_params_primals,
        lambda_primal,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        d_charge_param_idxs_,
        d_lj_param_idxs_,
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
        k_nonbonded_exclusion_inference<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
            E_,
            d_coords_primals,
            d_params_primals,
            lambda_primal,
            d_lambda_plane_idxs_,
            d_lambda_offset_idxs_,
            d_exclusion_idxs_,
            d_charge_scale_idxs_,
            d_lj_scale_idxs_,
            d_charge_param_idxs_,
            d_lj_param_idxs_,
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
void Nonbonded<RealType>::execute_lambda_jvp_device(
    const int N,
    const int P,
    const double *d_coords_primals,
    const double *d_coords_tangents,
    const double *d_params_primals,
    const double lambda_primal,
    const double lambda_tangent,
    double *d_out_coords_primals,
    double *d_out_coords_tangents,
    double *d_out_params_primals,
    double *d_out_params_tangents,
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

    k_nonbonded_jvp<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_coords_primals,
        d_coords_tangents,
        d_params_primals,
        lambda_primal,
        lambda_tangent,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        d_charge_param_idxs_,
        d_lj_param_idxs_,
        cutoff_,
        nblist_.get_block_bounds_ctr(),
        nblist_.get_block_bounds_ext(),
        d_out_coords_primals,
        d_out_coords_tangents,
        d_out_params_primals,
        d_out_params_tangents
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if(E_ > 0) {
        k_nonbonded_exclusion_jvp<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
            E_,
            d_coords_primals,
            d_coords_tangents,
            d_params_primals,
            lambda_primal,
            lambda_tangent,
            d_lambda_plane_idxs_,
            d_lambda_offset_idxs_,
            d_exclusion_idxs_,
            d_charge_scale_idxs_,
            d_lj_scale_idxs_,
            d_charge_param_idxs_,
            d_lj_param_idxs_,
            cutoff_,
            d_out_coords_primals,
            d_out_coords_tangents,
            d_out_params_primals,
            d_out_params_tangents
        );            

        // cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }



};

template class Nonbonded<double>;
template class Nonbonded<float>;

} // namespace timemachine