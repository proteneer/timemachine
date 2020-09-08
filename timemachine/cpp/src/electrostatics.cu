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
    const std::vector<int> &exclusion_idxs, // [E,2]
    const std::vector<double> &charge_scales, // [E]
    const std::vector<int> &lambda_plane_idxs, // [N]
    const std::vector<int> &lambda_offset_idxs, // [N]
    double beta,
    double cutoff
) :  N_(lambda_plane_idxs.size()),
    beta_(beta),
    cutoff_(cutoff),
    E_(exclusion_idxs.size()/2),
    nblist_(lambda_plane_idxs.size(), 3) {


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


};

template <typename RealType>
Electrostatics<RealType>::~Electrostatics() {
    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_charge_scales_));
    gpuErrchk(cudaFree(d_lambda_plane_idxs_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));
};

template <typename RealType>
void Electrostatics<RealType>::execute_device(
    const int N,
    const int P,
    const double *d_x,
    const double *d_p,
    const double *d_box,
    const double lambda,
    unsigned long long *d_du_dx,
    double *d_du_dp,
    double *d_du_dl,
    double *d_u,
    cudaStream_t stream) {

    if(N != N_) {
        throw std::runtime_error("N != N_");
    }

    const int tpb = 32;
    const int B = (N_+tpb-1)/tpb;
    const int D = 3;

    // its safe for us to build a neighborlist in a lower dimension.
    nblist_.compute_block_bounds(N_, D, d_x, stream);

    gpuErrchk(cudaPeekAtLastError());

    dim3 dimGrid(B, B, 1); // x, y, z dims
    dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

    auto start = std::chrono::high_resolution_clock::now();

    // these can be ran in two streams later on
    k_electrostatics<RealType><<<dimGrid, tpb, 0, stream>>>(
        N_,
        d_x,
        d_p,
        d_box,
        lambda,
        d_lambda_plane_idxs_,
        d_lambda_offset_idxs_,
        beta_,
        cutoff_,
        nblist_.get_block_bounds_ctr(),
        nblist_.get_block_bounds_ext(),
        d_du_dx,
        d_du_dp,
        d_du_dl,
        d_u
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // if(E_ > 0) {
    //     k_electrostatics_exclusion_inference<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
    //         E_,
    //         d_coords_primals,
    //         lambda_primal,
    //         d_lambda_plane_idxs_,
    //         d_lambda_offset_idxs_,
    //         d_exclusion_idxs_,
    //         d_charge_scales_,
    //         d_charge_params_,
    //         cutoff_,
    //         d_out_coords_primals,
    //         d_out_lambda_primals,
    //         d_out_energy_primal
    //     );
    //     // cudaDeviceSynchronize();
    //     gpuErrchk(cudaPeekAtLastError());
    // }
}

template class Electrostatics<double>;
template class Electrostatics<float>;

} // namespace timemachine