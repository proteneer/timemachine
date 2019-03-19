#include "cublas_v2.h"
#include "curand.h"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdio>

#include "integrator.hpp"
#include "gpu_utils.cuh"

template <typename NumericType>
__global__ void reduce_velocities(
    const NumericType *noise,
    const NumericType coeff_a,
    const NumericType *coeff_bs, // N x 3, not P x N x 3, but we could just pass in the first index
    const NumericType *coeff_cs,
    const NumericType *grads,
    const NumericType d_t,
    NumericType *x_t,
    NumericType *v_t,
    int N3) {

    int local_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(local_idx >= N3) {
        return;
    }

    v_t[local_idx] = coeff_a*v_t[local_idx] - coeff_bs[local_idx]*grads[local_idx] + coeff_cs[local_idx]*noise[local_idx];
    x_t[local_idx] += v_t[local_idx]*d_t;
}


template<typename NumericType>
__global__ void update_derivatives(
    NumericType coeff_a,
    const NumericType *coeff_bs,
    const NumericType *hmp,
    NumericType *dxdp_t,
    NumericType *dvdp_t,
    NumericType dt,
    int PN3 // num_params * num_atoms * 3
) {

    int local_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(local_idx >= PN3) {
        return;
    }

    NumericType tmp = coeff_a*dvdp_t[local_idx] - coeff_bs[local_idx]*hmp[local_idx];
    dvdp_t[local_idx] = tmp;
    dxdp_t[local_idx] += dt*tmp;

}


namespace timemachine {


template<typename NumericType> 
Integrator<NumericType>::Integrator(
    NumericType dt,
    int W,
    int N,
    int P,
    const NumericType coeff_a,
    const std::vector<NumericType> &coeff_bs,
    const std::vector<NumericType> &coeff_cs) :
    dt_(dt),
    W_(W),
    N_(N),
    P_(P),
    step_(0),
    coeff_a_(coeff_a) {

    if(coeff_bs.size() != N) {
        throw(std::runtime_error("Expected coeffbs to be PxNx3 shape"));
    }
    if(coeff_cs.size() != N) {
        throw(std::runtime_error("Expected coeffbs to be PxNx3 shape"));
    }

    std::vector<NumericType> expanded_coeff_bs(P*N*3);
    for(size_t p=0; p < P; p++) {
        for(size_t n=0; n < N; n++) {
            for(size_t d=0; d < 3; d++) {
                expanded_coeff_bs[p*N*3+n*3+d] = coeff_bs[n];
            }
        }
    }

    std::vector<NumericType> expanded_coeff_cs(N*3);
    for(size_t n=0; n < N; n++) {
        for(size_t d=0; d < 3; d++) {
            expanded_coeff_cs[n*3+d] = coeff_cs[n];
        }
    }

    // 1. Allocate memory on the GPU
    gpuErrchk(cudaMalloc((void**)&d_x_t_, N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_v_t_, N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_dxdp_t_, P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_dvdp_t_, P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_coeff_bs_, P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_coeff_cs_, N_*3*sizeof(NumericType)));

    // 2. Per-step buffers
    gpuErrchk(cudaMalloc((void**)&d_energy_, sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_grads_, N_*3*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_hessians_, N_*3*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_mixed_partials_, P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_rng_buffer_, N_*3*sizeof(NumericType)));

    // 3. Memset
    gpuErrchk(cudaMemset(d_x_t_, 0.0, N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_v_t_, 0.0, N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_dxdp_t_, 0.0, P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_dvdp_t_, 0.0, P_*N_*3*sizeof(NumericType)));

    gpuErrchk(cudaMemcpy(d_coeff_bs_, &expanded_coeff_bs[0], P_*N_*3*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeff_cs_, &expanded_coeff_cs[0], N_*3*sizeof(NumericType), cudaMemcpyHostToDevice));

    cublasErrchk(cublasCreate(&cb_handle_));
    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_PHILOX4_32_10));

    // (ytz): looks like by default cuRand always sets the default seed to 0.
    curandSetPseudoRandomGeneratorSeed(cr_rng_, time(NULL));

}


template <typename NumericType>
void Integrator<NumericType>::reset() {
    step_ = 0;
    gpuErrchk(cudaMemset(d_x_t_, 0.0, N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_v_t_, 0.0, N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_dxdp_t_, 0.0, P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_dvdp_t_, 0.0, P_*N_*3*sizeof(NumericType)));


    gpuErrchk(cudaMemset(d_energy_, 0, sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_grads_, 0, N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_hessians_, 0, N_*3*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_mixed_partials_, 0, P_*N_*3*sizeof(NumericType)));


    cudaDeviceSynchronize();
}


template<typename NumericType> 
Integrator<NumericType>::~Integrator() {
    gpuErrchk(cudaFree(d_x_t_));
    gpuErrchk(cudaFree(d_v_t_));
    gpuErrchk(cudaFree(d_dxdp_t_));
    gpuErrchk(cudaFree(d_dvdp_t_));
    gpuErrchk(cudaFree(d_coeff_bs_));
    gpuErrchk(cudaFree(d_coeff_cs_));

    gpuErrchk(cudaFree(d_grads_));
    gpuErrchk(cudaFree(d_hessians_));
    gpuErrchk(cudaFree(d_mixed_partials_));
    gpuErrchk(cudaFree(d_rng_buffer_));

    cublasErrchk(cublasDestroy(cb_handle_));
    curandErrchk(curandDestroyGenerator(cr_rng_));
}

template<typename NumericType> 
std::vector<NumericType> Integrator<NumericType>::get_dxdp() const {
    std::vector<NumericType> buf(P_*N_*3);
    gpuErrchk(cudaMemcpy(&buf[0], d_dxdp_t_, P_*N_*3*sizeof(NumericType), cudaMemcpyDeviceToHost));
    return buf;
}

template<typename NumericType> 
std::vector<NumericType> Integrator<NumericType>::get_noise() const {
    std::vector<NumericType> buf(N_*3);
    gpuErrchk(cudaMemcpy(&buf[0], d_rng_buffer_, N_*3*sizeof(NumericType), cudaMemcpyDeviceToHost));
    return buf;
};

template<typename NumericType> 
std::vector<NumericType> Integrator<NumericType>::get_coordinates() const {
    std::vector<NumericType> buf(N_*3);
    gpuErrchk(cudaMemcpy(&buf[0], d_x_t_, N_*3*sizeof(NumericType), cudaMemcpyDeviceToHost));
    return buf;
};

template<typename NumericType> 
std::vector<NumericType> Integrator<NumericType>::get_velocities() const {
    std::vector<NumericType> buf(N_*3);
    gpuErrchk(cudaMemcpy(&buf[0], d_v_t_, N_*3*sizeof(NumericType), cudaMemcpyDeviceToHost));
    return buf;
};

template<typename NumericType> 
void Integrator<NumericType>::set_coordinates(std::vector<NumericType> x) {
    for(size_t i=0; i < x.size(); i++) {
        // std::cout << "SC: " << x[i] << std::endl;
    }
    gpuErrchk(cudaMemcpy(d_x_t_, &x[0], N_*3*sizeof(NumericType), cudaMemcpyHostToDevice));
};

template<typename NumericType> 
void Integrator<NumericType>::set_velocities(std::vector<NumericType> v) {
    for(size_t i=0; i < v.size(); i++) {
        // std::cout << "SV: " << v[i] << std::endl;
    }
    gpuErrchk(cudaMemcpy(d_v_t_, &v[0], N_*3*sizeof(NumericType), cudaMemcpyHostToDevice));
};

// dangerous! not exception safe.
template<typename NumericType> 
void Integrator<NumericType>::step_cpu(
    const NumericType *h_grads,
    const NumericType *h_hessians,
    const NumericType *h_mixed_partials) {

    gpuErrchk(cudaMemcpy(d_grads_, h_grads, N_*3*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_hessians_, h_hessians, N_*3*N_*3*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mixed_partials_, h_mixed_partials, P_*N_*3*sizeof(NumericType), cudaMemcpyHostToDevice));

    step_gpu(d_grads_, d_hessians_, d_mixed_partials_);

    cudaDeviceSynchronize();
}

template<typename NumericType> 
void Integrator<NumericType>::step_gpu(
    const NumericType *d_grads,
    const NumericType *d_hessians,
    NumericType *d_mixed_partials) {

    size_t tpb = 32;

    if(d_hessians != nullptr && d_mixed_partials != nullptr) {
        hessian_vector_product(d_hessians_, d_dxdp_t_, d_mixed_partials);
        size_t n_blocks = (P_*N_*3 + tpb - 1) / tpb;

        update_derivatives<NumericType><<<n_blocks, tpb>>>(
            coeff_a_,
            d_coeff_bs_,
            d_mixed_partials_,
            d_dxdp_t_,
            d_dvdp_t_,
            dt_,
            P_*N_*3
        );
        gpuErrchk(cudaPeekAtLastError());
    }

    size_t n_blocks = (N_*3 + tpb - 1) / tpb;

    // generate new random numbers
    curandErrchk(templateCurandNormal(cr_rng_, d_rng_buffer_, N_*3, 0.0, 1.0));
    reduce_velocities<NumericType><<<n_blocks, tpb>>>(
        d_rng_buffer_,
        coeff_a_,
        d_coeff_bs_,
        d_coeff_cs_,
        d_grads_,
        dt_,
        d_x_t_,
        d_v_t_,
        N_*3);


    gpuErrchk(cudaPeekAtLastError());

    step_ += 1;

}

template<typename NumericType> 
void Integrator<NumericType>::hessian_vector_product(
    const NumericType *d_A,
    NumericType *d_B,
    NumericType *d_C) {

    NumericType alpha = 1.0;
    NumericType beta  = 1.0;
 
    const size_t N3 = N_*3;

    // this is set to UPPER because of fortran ordering
    cublasErrchk(templateSymm(cb_handle_,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
        N3, P_,
        &alpha,
        d_A, N3,
        d_B, N3,
        &beta,
        d_C, N3));

}

}

template class timemachine::Integrator<double>;
template class timemachine::Integrator<float>;