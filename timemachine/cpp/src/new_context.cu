#include <iostream>

#include "new_context.hpp"
#include "integrator.hpp"
#include "assert.h"
#include "kernel_utils.cuh"
#include "stepper.hpp"

namespace timemachine {

// D == 3

template <typename RealType, int D>
ReversibleContext<RealType, D>::~ReversibleContext() {
    gpuErrchk(cudaFree(d_params_));
    gpuErrchk(cudaFree(d_params_grads_));

    gpuErrchk(cudaFree(d_coeff_cbs_));

    gpuErrchk(cudaFree(d_coords_));
    gpuErrchk(cudaFree(d_du_dls_));

    gpuErrchk(cudaFree(d_velocities_));
    gpuErrchk(cudaFree(d_forces_));

    gpuErrchk(cudaFree(d_x_t_tangent_));
    gpuErrchk(cudaFree(d_param_adjoint_accum_));
    gpuErrchk(cudaFree(d_x_t_adjoint_));
    gpuErrchk(cudaFree(d_v_t_adjoint_));

    gpuErrchk(cudaFree(d_dE_dx_jvp_));
    gpuErrchk(cudaFree(d_dE_dp_jvp_));
};

template <typename RealType, int D>
ReversibleContext<RealType, D>::ReversibleContext(
    Stepper<RealType> *stepper,
    const int N,
    const std::vector<RealType> &x0,
    const std::vector<RealType> &v0,
    const std::vector<RealType> &coeff_cas,
    const std::vector<RealType> &coeff_cbs,
    const std::vector<RealType> &step_sizes,
    const std::vector<RealType> &params) :
        N_(N),
        P_(params.size()),
        stepper_(stepper),
        coeff_cas_(coeff_cas),
        step_sizes_(step_sizes) {

    size_t T = step_sizes.size();
    size_t P = P_;
    size_t F = T+1; // number of frames is number of steps + 1

    assert(x0.size() == N*D);
    assert(v0.size() == N*D);
    assert(coeff_cas.size() == T);
    assert(coeff_cbs.size() == N);
    assert(step_sizes.size() == T);
    assert(params.size() == P);

    gpuErrchk(cudaMalloc(&d_coeff_cbs_, N*sizeof(RealType)));
    gpuErrchk(cudaMemcpy(d_coeff_cbs_, &coeff_cbs[0], N*sizeof(RealType), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_coords_, F*N*D*sizeof(RealType)));
    gpuErrchk(cudaMalloc(&d_du_dls_, F*sizeof(RealType)));
    cudaMemset(d_du_dls_, 0, F*sizeof(RealType));
    gpuErrchk(cudaMemcpy(d_coords_, &x0[0], N*D*sizeof(RealType), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_velocities_, N*D*sizeof(RealType)));
    gpuErrchk(cudaMemcpy(d_velocities_, &v0[0], N*D*sizeof(RealType), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_forces_, N*D*sizeof(*d_forces_)));

    gpuErrchk(cudaMalloc(&d_params_, P*sizeof(RealType)));
    gpuErrchk(cudaMemcpy(d_params_, &params[0], P*sizeof(RealType), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_x_t_tangent_, N*D*sizeof(RealType))); // [NxD]
    gpuErrchk(cudaMalloc(&d_param_adjoint_accum_, P*sizeof(RealType))); // [P]
    gpuErrchk(cudaMalloc(&d_x_t_adjoint_, N*D*sizeof(RealType))); // [NxD]
    gpuErrchk(cudaMalloc(&d_v_t_adjoint_, N*D*sizeof(RealType))); // [NxD]

    gpuErrchk(cudaMalloc(&d_dE_dx_jvp_, N*D*sizeof(RealType))); // [NxD]
    gpuErrchk(cudaMalloc(&d_dE_dp_jvp_, P*sizeof(RealType))); // [P]

    gpuErrchk(cudaMalloc(&d_params_grads_, P*sizeof(RealType))); // [P]
};

template <typename RealType, int D>
void ReversibleContext<RealType, D>::forward_mode() {

    for(int t=0; t < step_sizes_.size(); t++) {

        // compute gradients
        gpuErrchk(cudaMemset(d_forces_, 0, N_*D*sizeof(*d_forces_)));
        // gpuErrchk(cudaMemset(d_params_grads_, 0, P_*sizeof(RealType))); # not used

        stepper_->forward_step(
            N_,
            P_,
            d_coords_ + t*N_*D,
            d_params_,
            d_forces_
        );

        step_forward<RealType>(
            N_,
            D,
            coeff_cas_[t],
            d_coeff_cbs_,
            d_coords_ + t*N_*D,
            d_velocities_, 
            d_forces_,
            step_sizes_[t],
            d_coords_ + (t+1)*N_*D,
            d_velocities_
        );

    }

}

template <typename RealType>
__global__ void update_backward_1(
    const int N,
    const int D,
    const RealType *d_coeff_bs,
    const RealType *d_adjoint_x_t_new,
    RealType *d_adjoint_v_t_new,
    RealType *d_x_t_tangent,
    const RealType dt) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    int local_idx = atom_idx*D + d_idx;
    d_adjoint_v_t_new[local_idx] += dt*d_adjoint_x_t_new[local_idx];
    d_x_t_tangent[local_idx] = d_coeff_bs[atom_idx]*d_adjoint_v_t_new[local_idx];

};

template <typename RealType>
__global__ void update_backward_2(
    int P,
    const RealType *d_dE_dp_jvp,
    RealType *d_adjoint_params) {

    int p_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(p_idx >= P) {
        return;
    }
    d_adjoint_params[p_idx] += d_dE_dp_jvp[p_idx];
}

template <typename RealType>
__global__ void update_backward_3(
    const int N,
    const int D,
    const RealType ca,
    const RealType *d_adjoint_x_new,
    const RealType *d_adjoint_v_new,
    const RealType *d_x_t_jvp,
    RealType *d_adjoint_x_old,
    RealType *d_adjoint_v_old) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }
    int d_idx = blockIdx.y;
    int local_idx = atom_idx*D + d_idx;

    d_adjoint_x_old[local_idx] += d_x_t_jvp[local_idx];
    d_adjoint_v_old[local_idx] = ca*d_adjoint_v_new[local_idx];

};

template <typename RealType, int D>
void ReversibleContext<RealType, D>::set_x_t_adjoint(const RealType *buffer) {
    gpuErrchk(cudaMemcpy(d_x_t_adjoint_, buffer, N_*D*sizeof(*buffer), cudaMemcpyHostToDevice));
}

template <typename RealType, int D>
void ReversibleContext<RealType, D>::backward_mode() {

    // initialized
    // d_x_t adjoint has been set via set_x_t_adjoint() but we should make it possible
    // to set the rest of the initial adjoints as well.
    gpuErrchk(cudaMemset(d_v_t_adjoint_, 0, N_*D*sizeof(*d_v_t_adjoint_)));
    gpuErrchk(cudaMemset(d_param_adjoint_accum_, 0, P_*sizeof(RealType)));

    // compute derivatives
    for(int t = step_sizes_.size()-1; t >= 0; t--) {
        size_t tpb = 32;
        size_t n_blocks = (N_*D + tpb - 1) / tpb;
        dim3 dimGrid_dx(n_blocks, D);
        gpuErrchk(cudaMemset(d_x_t_tangent_, 0, N_*D*sizeof(*d_x_t_tangent_)));

        update_backward_1<<<dimGrid_dx, tpb>>>(
            N_,
            D,
            d_coeff_cbs_,
            d_x_t_adjoint_,
            d_v_t_adjoint_,
            d_x_t_tangent_,
            step_sizes_[t] // this is out of bounds
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        // important that we set the memory addresses to zero
        gpuErrchk(cudaMemset(d_dE_dx_jvp_, 0, N_*D*sizeof(*d_dE_dx_jvp_)));
        gpuErrchk(cudaMemset(d_dE_dp_jvp_, 0, P_*sizeof(*d_dE_dp_jvp_)));

        stepper_->backward_step(
            N_,
            P_,
            d_coords_ + t*N_*D,
            d_params_,
            d_x_t_tangent_,
            d_dE_dx_jvp_,
            d_dE_dp_jvp_
        );

        size_t n_block_params = (P_ + tpb - 1) / tpb;

        // we can probably *directly* atomic add into this (and into the d_adjoint_xol in the above func)
        // also would let us save a little bit more buffer room
        update_backward_2<<<n_block_params, tpb>>>(P_, d_dE_dp_jvp_, d_param_adjoint_accum_);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        update_backward_3<<<dimGrid_dx, tpb>>>(
            N_,
            D,
            coeff_cas_[t],
            d_x_t_adjoint_,
            d_v_t_adjoint_,
            d_dE_dx_jvp_,
            d_x_t_adjoint_,
            d_v_t_adjoint_  
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

    }

}

template <typename RealType, int D>
void ReversibleContext<RealType, D>::get_all_coords(RealType *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_coords_, (step_sizes_.size()+1)*N_*D*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template <typename RealType, int D>
void ReversibleContext<RealType, D>::get_param_adjoint_accum(RealType *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_param_adjoint_accum_, P_*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template <typename RealType, int D>
void ReversibleContext<RealType, D>::get_x_t_adjoint(RealType *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_x_t_adjoint_, N_*D*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template <typename RealType, int D>
void ReversibleContext<RealType, D>::get_v_t_adjoint(RealType *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_v_t_adjoint_, N_*D*sizeof(RealType), cudaMemcpyDeviceToHost));
}

template class ReversibleContext<double, 4>;
template class ReversibleContext<double, 3>;

};