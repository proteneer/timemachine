#include <iostream>
#include "context.hpp"
#include "gpu_utils.cuh"

namespace timemachine {

template<typename RealType>
Context<RealType>::Context(
    const std::vector<Potential<RealType>* > system,
    const Optimizer<RealType> *optimizer,
    const RealType *h_params,
    const RealType *h_x0,
    const RealType *h_v0,
    const int N,
    const int D,
    const int P,
    const int *h_gather_param_idxs,
    const int DP) : system_(system),
    optimizer_(optimizer),
    step_(0),
    N_(N),
    P_(P),
    D_(D),
    DP_(DP) {
    // if DP == 0 then this is null

    // 1. allocate
    gpuErrchk(cudaMalloc((void**)&d_params_, P*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_gather_param_idxs_, P*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_x_t_, N*D*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_v_t_, N*D*sizeof(RealType)));

    gpuErrchk(cudaMalloc((void**)&d_E_, sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_dE_dx_, N*D*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_dE_dp_, DP*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_d2E_dx2_, N*N*D*D*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_d2E_dxdp_, DP*N*D*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_dx_dp_t_, DP*N*D*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_dv_dp_t_, DP*N*D*sizeof(RealType)));

    // 2. memcpy and memset to initialize
    gpuErrchk(cudaMemcpy(d_params_, h_params, P*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_gather_param_idxs_, h_gather_param_idxs, P*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_x_t_, h_x0, N*D*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v_t_, h_v0, N*D*sizeof(RealType), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_dx_dp_t_, 0, DP*N*D*sizeof(RealType)));
    gpuErrchk(cudaMemset(d_dv_dp_t_, 0, DP*N*D*sizeof(RealType)));

}

template<typename RealType>
Context<RealType>::~Context() {
    gpuErrchk(cudaFree(d_params_));
    gpuErrchk(cudaFree(d_gather_param_idxs_));
    gpuErrchk(cudaFree(d_x_t_));
    gpuErrchk(cudaFree(d_v_t_));

    gpuErrchk(cudaFree(d_E_));
    gpuErrchk(cudaFree(d_dE_dx_));
    gpuErrchk(cudaFree(d_dE_dp_));
    gpuErrchk(cudaFree(d_d2E_dx2_));
    gpuErrchk(cudaFree(d_d2E_dxdp_));
    gpuErrchk(cudaFree(d_dx_dp_t_));
    gpuErrchk(cudaFree(d_dv_dp_t_));
}

template<typename RealType>
void Context<RealType>::step() {

    // reset force buffers
    gpuErrchk(cudaMemset(d_E_, 0, sizeof(RealType)));
    gpuErrchk(cudaMemset(d_dE_dx_, 0, N_*D_*sizeof(RealType)));
    gpuErrchk(cudaMemset(d_dE_dp_, 0, DP_*sizeof(RealType)));
    gpuErrchk(cudaMemset(d_d2E_dx2_, 0, N_*N_*D_*D_*sizeof(RealType)));
    gpuErrchk(cudaMemset(d_d2E_dxdp_, 0, DP_*N_*D_*sizeof(RealType)));

    for(auto nrg : system_) {
        nrg->derivatives_device(
            1, // one conformer when doing dynamics
            N_,
            D_, // FIX ME
            d_x_t_,
            d_params_,
            d_E_, // this is likely optional as well
            d_dE_dx_,
            d_d2E_dx2_,
            DP_,
            d_gather_param_idxs_,
            d_dE_dp_, // (ytz) we don't actually need to compute dE_dp, so can probably speed this up later
            d_d2E_dxdp_
        );
    }

    optimizer_->step(
        N_,
        D_,
        DP_,
        d_dE_dx_,
        d_d2E_dx2_,
        d_d2E_dxdp_,
        d_x_t_,
        d_v_t_,
        d_dx_dp_t_,
        d_dv_dp_t_
    );
    step_++;

}

// template<typename RealType>
// void Context<RealType>::debug_compute_dE_dx(
//         const RealType *h_new_x,
//         RealType *h_E,
//         RealType *h_dE_dx) {

//     gpuErrchk(cudaMemset(d_E_, 0, sizeof(RealType)));
//     gpuErrchk(cudaMemset(d_dE_dx_, 0, N_*3*sizeof(RealType)));

//     gpuErrchk(cudaMemcpy(d_x_t_, h_new_x, N_*3*sizeof(RealType), cudaMemcpyHostToDevice));
    
//     for(auto nrg : system_) {
//         nrg->derivatives_device(
//             1, // one conformer when doing dynamics
//             N_,
//             d_x_t_,
//             d_params_,
//             d_E_, // this is likely optional as well
//             d_dE_dx_,
//             d_d2E_dx2_,
//             DP_,
//             d_gather_param_idxs_,
//             d_dE_dp_, // (ytz) we don't actually need to compute dE_dp, so can probably speed this up later
//             d_d2E_dxdp_
//         );
//     }

//     gpuErrchk(cudaMemcpy(h_dE_dx, d_dE_dx_, N_*3*sizeof(RealType), cudaMemcpyDeviceToHost));
//     gpuErrchk(cudaMemcpy(h_E, d_E_, sizeof(RealType), cudaMemcpyDeviceToHost));


// }

template<typename RealType>
void Context<RealType>::get_x(RealType *buffer) const {
    gpuErrchk(cudaMemcpy(buffer, d_x_t_, N_*D_*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template<typename RealType>
void Context<RealType>::get_v(RealType *buffer) const {
    gpuErrchk(cudaMemcpy(buffer, d_v_t_, N_*D_*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template<typename RealType>
void Context<RealType>::get_E(RealType *buffer) const {
    gpuErrchk(cudaMemcpy(buffer, d_E_, sizeof(RealType), cudaMemcpyDeviceToHost));
}

template<typename RealType>
void Context<RealType>::get_dE_dx(RealType *buffer) const {
    gpuErrchk(cudaMemcpy(buffer, d_dE_dx_, N_*D_*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template<typename RealType>
void Context<RealType>::get_d2E_dx2(RealType *buffer) const {
    gpuErrchk(cudaMemcpy(buffer, d_d2E_dx2_, N_*D_*N_*D_*sizeof(RealType), cudaMemcpyDeviceToHost));
    // fill diagonals
    size_t K = N_*D_;
    for(size_t row=0; row < K; row++) {
        for(size_t col=0; col < row; col++) {
            buffer[col*K+row] = buffer[row*K+col];
        }
    }
}

template<typename RealType>
void Context<RealType>::get_d2E_dxdp(RealType *buffer) const {
    gpuErrchk(cudaMemcpy(buffer, d_d2E_dxdp_, DP_*N_*D_*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template<typename RealType>
void Context<RealType>::get_dE_dp(RealType *buffer) const {
    gpuErrchk(cudaMemcpy(buffer, d_dE_dp_, DP_*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template<typename RealType>
void Context<RealType>::get_dx_dp(RealType *buffer) const {
    gpuErrchk(cudaMemcpy(buffer, d_dx_dp_t_, DP_*N_*D_*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template<typename RealType>
void Context<RealType>::get_dv_dp(RealType *buffer) const {
    gpuErrchk(cudaMemcpy(buffer, d_dv_dp_t_, DP_*N_*D_*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template class Context<float>;
template class Context<double>;

}