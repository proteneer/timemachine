#include <iostream>
#include <chrono>

#include "context.hpp"
#include "integrator.hpp"
#include "assert.h"
#include "kernel_utils.cuh"
#include "gpu_utils.cuh"
#include "stepper.hpp"

namespace timemachine {

ReversibleContext::~ReversibleContext() {

    curandErrchk(curandDestroyGenerator(cr_rng_));

    gpuErrchk(cudaFree(d_coeff_cbs_));
    gpuErrchk(cudaFree(d_coeff_ccs_));
    gpuErrchk(cudaFree(d_noise_buffer_));

    gpuErrchk(cudaFree(d_coords_));

    gpuErrchk(cudaFree(d_velocities_));
    gpuErrchk(cudaFree(d_forces_));

    gpuErrchk(cudaFree(d_x_t_tangent_));
    gpuErrchk(cudaFree(d_x_t_adjoint_));
    gpuErrchk(cudaFree(d_v_t_adjoint_));

    gpuErrchk(cudaFree(d_dE_dx_jvp_primals_));
    gpuErrchk(cudaFree(d_dE_dx_jvp_tangents_));

};

int round_up_even(int count) {
    if(count % 2 == 1) {
        return count += 1;
    } else {
        return count;
    }
}

ReversibleContext::ReversibleContext(
    Stepper *stepper,
    const int N,
    const std::vector<double> &x0,
    const std::vector<double> &v0,
    const std::vector<double> &coeff_cas,
    const std::vector<double> &coeff_cbs,
    const std::vector<double> &coeff_ccs,
    const std::vector<double> &step_sizes,
    unsigned long long seed) :
        N_(N),
        stepper_(stepper),
        coeff_cas_(coeff_cas),
        step_sizes_(step_sizes) {

    size_t T = step_sizes.size();
    size_t F = T+1; // number of frames is number of steps + 1

    assert(x0.size() == N*D);
    assert(v0.size() == N*D);
    assert(coeff_cas.size() == T);
    assert(coeff_cbs.size() == N);
    assert(coeff_ccs.size() == N);
    assert(step_sizes.size() == T);

    gpuErrchk(cudaMalloc(&d_coeff_cbs_, N*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_coeff_cbs_, &coeff_cbs[0], N*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_coeff_ccs_, N*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_coeff_ccs_, &coeff_ccs[0], N*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_coords_, F*N*D*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_coords_, &x0[0], N*D*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_velocities_, N*D*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_velocities_, &v0[0], N*D*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_forces_, N*D*sizeof(*d_forces_)));


    gpuErrchk(cudaMalloc(&d_x_t_tangent_, N*D*sizeof(double))); // [NxD]
    gpuErrchk(cudaMalloc(&d_x_t_adjoint_, N*D*sizeof(double))); // [NxD]
    gpuErrchk(cudaMalloc(&d_v_t_adjoint_, N*D*sizeof(double))); // [NxD]

    gpuErrchk(cudaMalloc(&d_dE_dx_jvp_primals_, N*D*sizeof(double))); // [NxD]
    gpuErrchk(cudaMalloc(&d_dE_dx_jvp_tangents_, N*D*sizeof(double))); // [NxD]
    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));

    gpuErrchk(cudaMalloc((void**)&d_noise_buffer_, round_up_even(N*D)*sizeof(double)));

    curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));
};

void ReversibleContext::forward_mode() {

    for(int t=0; t < step_sizes_.size(); t++) {

        gpuErrchk(cudaMemset(d_forces_, 0, N_*D*sizeof(*d_forces_)));

    	auto start0 = std::chrono::high_resolution_clock::now();

        stepper_->forward_step(
            N_,
            d_coords_ + t*N_*D,
            d_forces_
        );
    	auto finish0 = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double> elapsed0 = finish0 - start0;

    	auto start = std::chrono::high_resolution_clock::now();

        curandErrchk(templateCurandNormal(cr_rng_, d_noise_buffer_, round_up_even(N_*D), 0.0, 1.0));

        step_forward<double>(
            N_,
            D,
            coeff_cas_[t],
            d_coeff_cbs_,
            d_coeff_ccs_,
            d_noise_buffer_,
            d_coords_ + t*N_*D,
            d_velocities_, 
            d_forces_,
            step_sizes_[t],
            d_coords_ + (t+1)*N_*D,
            d_velocities_
        );

    	auto finish = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double> elapsed = finish - start;

    }

}

__global__ void update_backward_1(
    const int N,
    const int D,
    const double *d_coeff_bs,
    const double *d_adjoint_x_t_new,
    double *d_adjoint_v_t_new,
    double *d_x_t_tangent,
    const double dt) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    int local_idx = atom_idx*D + d_idx;
    d_adjoint_v_t_new[local_idx] += dt*d_adjoint_x_t_new[local_idx];
    d_x_t_tangent[local_idx] = d_coeff_bs[atom_idx]*d_adjoint_v_t_new[local_idx];

};

__global__ void update_backward_3(
    const int N,
    const int D,
    const double ca,
    const double *d_adjoint_x_new,
    const double *d_adjoint_v_new,
    const double *d_x_t_jvp,
    double *d_adjoint_x_old,
    double *d_adjoint_v_old) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }
    int d_idx = blockIdx.y;
    int local_idx = atom_idx*D + d_idx;

    d_adjoint_x_old[local_idx] += d_x_t_jvp[local_idx];
    d_adjoint_v_old[local_idx] = ca*d_adjoint_v_new[local_idx];

};

void ReversibleContext::set_x_t_adjoint(const double *buffer) {
    gpuErrchk(cudaMemcpy(d_x_t_adjoint_, buffer, N_*D*sizeof(*buffer), cudaMemcpyHostToDevice));
}

void ReversibleContext::set_v_t_adjoint(const double *buffer) {
    gpuErrchk(cudaMemcpy(d_v_t_adjoint_, buffer, N_*D*sizeof(*buffer), cudaMemcpyHostToDevice));
}

void ReversibleContext::backward_mode() {

    // initialized
    // d_x_t adjoint has been set via set_x_t_adjoint() but we should make it possible
    // to set the rest of the initial adjoints as well.
    gpuErrchk(cudaMemset(d_v_t_adjoint_, 0, N_*D*sizeof(*d_v_t_adjoint_)));

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
        gpuErrchk(cudaMemset(d_dE_dx_jvp_primals_, 0, N_*D*sizeof(*d_dE_dx_jvp_primals_)));
        gpuErrchk(cudaMemset(d_dE_dx_jvp_tangents_, 0, N_*D*sizeof(*d_dE_dx_jvp_tangents_)));

        stepper_->backward_step(
            N_,
            d_coords_ + t*N_*D,
            d_x_t_tangent_,
            d_dE_dx_jvp_primals_,
            d_dE_dx_jvp_tangents_
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        update_backward_3<<<dimGrid_dx, tpb>>>(
            N_,
            D,
            coeff_cas_[t],
            d_x_t_adjoint_,
            d_v_t_adjoint_,
            d_dE_dx_jvp_tangents_,
            d_x_t_adjoint_,
            d_v_t_adjoint_  
        );

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

    }

}

void ReversibleContext::get_last_coords(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_coords_ + step_sizes_.size()*N_*D, N_*D*sizeof(double), cudaMemcpyDeviceToHost));
}

void ReversibleContext::get_last_velocities(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_velocities_ + step_sizes_.size()*N_*D, N_*D*sizeof(double), cudaMemcpyDeviceToHost));
}

void ReversibleContext::get_all_coords(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_coords_, (step_sizes_.size()+1)*N_*D*sizeof(double), cudaMemcpyDeviceToHost));
}

void ReversibleContext::get_x_t_adjoint(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_x_t_adjoint_, N_*D*sizeof(double), cudaMemcpyDeviceToHost));
}

void ReversibleContext::get_v_t_adjoint(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_v_t_adjoint_, N_*D*sizeof(double), cudaMemcpyDeviceToHost));
}

};
