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

    gpuErrchk(cudaMalloc(&d_du_dl_buffer_, N_*sizeof(*d_du_dl_buffer_)));
    gpuErrchk(cudaMalloc(&d_u_buffer_, N_*sizeof(*d_u_buffer_)));

    gpuErrchk(cudaMalloc(&d_exclusion_idxs_, E_*2*sizeof(*d_exclusion_idxs_)));
    gpuErrchk(cudaMemcpy(d_exclusion_idxs_, &exclusion_idxs[0], E_*2*sizeof(*d_exclusion_idxs_), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_scales_, E_*2*sizeof(*d_scales_)));
    gpuErrchk(cudaMemcpy(d_scales_, &scales[0], E_*sizeof(*d_scales_), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMallocHost(&p_ixn_count_, 1*sizeof(*p_ixn_count_)));

};

template <typename RealType>
Nonbonded<RealType>::~Nonbonded() {

    gpuErrchk(cudaFree(d_exclusion_idxs_));
    gpuErrchk(cudaFree(d_scales_));
    gpuErrchk(cudaFree(d_lambda_offset_idxs_));

    gpuErrchk(cudaFree(d_du_dl_buffer_));
    gpuErrchk(cudaFree(d_u_buffer_));

    gpuErrchk(cudaFreeHost(p_ixn_count_));

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

    gpuErrchk(cudaMemcpyAsync(p_ixn_count_, nblist_.get_ixn_count(), 1*sizeof(*p_ixn_count_), cudaMemcpyDeviceToHost, stream));
    // this stream needs to be synchronized so we can be sure that p_ixn_count_ is properly set.
    gpuErrchk(cudaStreamSynchronize(stream));

    if(d_du_dl) {
        gpuErrchk(cudaMemsetAsync(d_du_dl_buffer_, 0, N*sizeof(*d_du_dl_buffer_), stream));        
    }
    if(d_u) {
        gpuErrchk(cudaMemsetAsync(d_u_buffer_, 0, N*sizeof(*d_du_dl_buffer_), stream));        
    }

    k_nonbonded<RealType><<<p_ixn_count_[0], 32, 0, stream>>>(
        N,
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
        d_du_dl ? d_du_dl_buffer_ : nullptr, // switch to nullptr if we don't request du_dl
        d_u ? d_u_buffer_ : nullptr // switch to nullptr if we don't request energies
    );

    // cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    const int B = (N+32-1)/32;

    // these are called periodically so we use a slow implementation to reduce

    if(d_du_dl) {
        k_reduce_buffer<<<B, 32, 0, stream>>>(N, d_du_dl_buffer_, d_du_dl);
        gpuErrchk(cudaPeekAtLastError());
    }


    if(d_u) {
        k_reduce_buffer<<<B, 32, 0, stream>>>(N, d_u_buffer_, d_u);
        gpuErrchk(cudaPeekAtLastError());
    }
    

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