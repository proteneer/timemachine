#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
#include "nonbonded.hpp"
#include "gpu_utils.cuh"

#include "k_nonbonded.cuh"

namespace timemachine {

template <typename RealType>
Nonbonded<RealType>::Nonbonded(
    const std::vector<int> &exclusion_idxs, // [E,2]
    const std::vector<double> &scales, // [E, 2]
    const std::vector<int> &lambda_offset_idxs, // [N]
    double beta,
    double cutoff
) :  N_(lambda_offset_idxs.size()),
    cutoff_(cutoff),
    E_(exclusion_idxs.size()/2),
    nblist_(lambda_offset_idxs.size()),
    beta_(beta) {

    if(lambda_offset_idxs.size() != N_) {
        throw std::runtime_error("lambda offset idxs need to have size N");
    }

    if(scales.size()/2 != E_) {
        throw std::runtime_error("bad scales size!");
    }

    gpuErrchk(cudaMalloc(&d_lambda_offset_idxs_, N_*sizeof(*d_lambda_offset_idxs_)));
    gpuErrchk(cudaMemcpy(d_lambda_offset_idxs_, &lambda_offset_idxs[0], N_*sizeof(*d_lambda_offset_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_scales_, E_*2*sizeof(*d_scales_)));
    gpuErrchk(cudaMemcpy(d_scales_, &scales[0], E_*sizeof(*d_scales_), cudaMemcpyHostToDevice));

};

template <typename RealType>
Nonbonded<RealType>::~Nonbonded() {

    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_scales_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));

};

template <typename RealType>
void Nonbonded<RealType>::execute_device(
        const int N,
        const int P,
        const double *d_x,
        const double *d_p, // N*3
        const double *d_box,
        const double lambda,
        unsigned long long *d_du_dx,
        double *d_du_dp,
        double *d_du_dl,
        double *d_u,
        cudaStream_t stream) {

    assert(N == N_);
    assert(P == N_*3);

    if(N != N_) {
        throw std::runtime_error("N != N_");
    }

    nblist_.build_nblist_device(
        N,
        d_x,
        d_box,
        cutoff_,
        stream
    );

    unsigned int *d_ixn_count = nblist_.get_ixn_count();
    unsigned int h_ixn_count;
    gpuErrchk(cudaMemcpy(&h_ixn_count, d_ixn_count, 1*sizeof(*d_ixn_count), cudaMemcpyDeviceToHost));

    std::cout << "h_ixn_count: " << h_ixn_count << std::endl;

    k_nonbonded<RealType><<<h_ixn_count, 32, 0, stream>>>(
        N_,
        d_x,
        d_p,
        d_box,
        lambda,
        d_lambda_offset_idxs_,
        beta_,
        cutoff_,
        nblist_.get_ixn_tiles(),
        nblist_.get_ixn_atoms(),
        d_du_dx,
        d_du_dp,
        d_du_dl,
        d_u);

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // if(E_ > 0) {
    //     // dim3 dimGridExclusions((E_+tpb-1)/tpb, 1, 1);

    //     k_nonbonded_exclusion_inference<RealType><<<dimGridExclusions, tpb, 0, stream>>>(
    //         E_,
    //         d_coords_primals,
    //         // d_params_primals,
    //         lambda_primal,
    //         d_lambda_plane_idxs_,
    //         d_lambda_offset_idxs_,
    //         d_exclusion_idxs_,
    //         d_charge_scales_,
    //         d_lj_scales_,
    //         d_charge_params_,
    //         d_lj_params_,
    //         cutoff_,
    //         d_out_coords_primals,
    //         d_out_lambda_primals,
    //         d_out_energy_primal
    //     );
    //     // cudaDeviceSynchronize();
    //     gpuErrchk(cudaPeekAtLastError());
    // }
}

template class Nonbonded<double>;
template class Nonbonded<float>;

} // namespace timemachine