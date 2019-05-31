#include "cublas_v2.h"
#include "curand.h"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdio>

#include "integrator.hpp"
#include "gpu_utils.cuh"

template <typename RealType>
__global__ void reduce_velocities(
    const RealType *noise,
    const RealType coeff_a,
    const RealType *coeff_bs, // N x 3, not P x N x 3, but we could just pass in the first index
    const RealType *coeff_cs,
    const RealType *grads,
    const RealType d_t,
    RealType *x_t,
    RealType *v_t,
    int N3) {

    int local_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(local_idx >= N3) {
        return;
    }

    v_t[local_idx] = coeff_a*v_t[local_idx] - coeff_bs[local_idx]*grads[local_idx] + coeff_cs[local_idx]*noise[local_idx];
    x_t[local_idx] += v_t[local_idx]*d_t;
}


// REWRITE when P is implicit in gridDim.y

template<typename RealType>
__global__ void update_derivatives(
    RealType coeff_a,
    const RealType *coeff_bs,
    const RealType *hmp,
    RealType *dxdp_t,
    RealType *dvdp_t,
    RealType dt,
    int N3) {

    int local_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(local_idx >= N3) {
        return;
    }

    int p_idx = blockIdx.y;

    RealType tmp = coeff_a*dvdp_t[p_idx*N3 + local_idx] - coeff_bs[local_idx]*hmp[p*N3 + local_idx];
    dvdp_t[p_idx*N3 + local_idx] = tmp;
    dxdp_t[p_idx*N3 + local_idx] += dt*tmp;

}


namespace timemachine {


template<typename RealType> 
LangevinOptimizer<RealType>::LangevinOptimizer(
    RealType dt,
    const RealType coeff_a,
    const std::vector<RealType> &coeff_bs,
    const std::vector<RealType> &coeff_cs) :
    dt_(dt),
    step_(0),
    coeff_a_(coeff_a) {

    // if(coeff_bs.size() != N) {
    //     throw(std::runtime_error("Expected coeffbs to be PxNx3 shape"));
    // }
    // if(coeff_cs.size() != N) {
    //     throw(std::runtime_error("Expected coeffbs to be PxNx3 shape"));
    // }

    // std::vector<RealType> expanded_coeff_bs(); // NOT PN3, but N3
    // for(size_t p=0; p < P; p++) {
        // for(size_t n=0; n < N; n++) {
            // for(size_t d=0; d < 3; d++) {
                // expanded_coeff_bs[p*N*3+n*3+d] = coeff_bs[n];
    //         }
    //     }
    // }

    // std::vector<RealType> expanded_coeff_cs(N*3);
    // for(size_t n=0; n < N; n++) {
    //     for(size_t d=0; d < 3; d++) {
    //         expanded_coeff_cs[n*3+d] = coeff_cs[n];
    //     }
    // }

    // 1. Allocate memory on the GPU
    // gpuErrchk(cudaMalloc((void**)&d_x_t_, N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMalloc((void**)&d_v_t_, N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMalloc((void**)&d_dxdp_t_, P_*N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMalloc((void**)&d_dvdp_t_, P_*N_*3*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_coeff_bs_, coeff_bs.size()*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_coeff_cs_, coeff_cs.size()*sizeof(RealType)));

    // 2. Per-step buffers
    // gpuErrchk(cudaMalloc((void**)&d_energy_, sizeof(RealType)));
    // gpuErrchk(cudaMalloc((void**)&d_grads_, N_*3*N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMalloc((void**)&d_hessians_, N_*3*N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMalloc((void**)&d_mixed_partials_, P_*N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMalloc((void**)&d_rng_buffer_, N_*3*sizeof(RealType)));

    // 3. Memset
    // gpuErrchk(cudaMemset(d_x_t_, 0.0, N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMemset(d_v_t_, 0.0, N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMemset(d_dxdp_t_, 0.0, P_*N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMemset(d_dvdp_t_, 0.0, P_*N_*3*sizeof(RealType)));

    gpuErrchk(cudaMemcpy(d_coeff_bs_, &coeff_bs[0], coeff_bs.size()*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeff_cs_, &coeff_cs[0], coeff_cs.size()*sizeof(RealType), cudaMemcpyHostToDevice));

    cublasErrchk(cublasCreate(&cb_handle_));
    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_PHILOX4_32_10));

    // (ytz): looks like by default cuRand always sets the default seed to 0.
    // allow extra arg for this to be set to zero
    curandSetPseudoRandomGeneratorSeed(cr_rng_, time(NULL));

}


template <typename RealType>
void Integrator<RealType>::opt_init(
    const int N,
    const RealType *h_x0,
    const RealType *h_v0,
    const std::vector<int> &dp_idxs) {
    step_ = 0;
    curandSetPseudoRandomGeneratorSeed(cr_rng_, time(NULL));

    const int P = dp_idxs.size();

    gpuErrchk(cudaMemset(d_x_t_, 0.0, N*3*sizeof(RealType)));
    gpuErrchk(cudaMemset(d_v_t_, 0.0, N*3*sizeof(RealType)));
    gpuErrchk(cudaMemset(d_dxdp_t_, 0.0, P*N*3*sizeof(RealType)));
    gpuErrchk(cudaMemset(d_dvdp_t_, 0.0, P*N*3*sizeof(RealType)));


    // gpuErrchk(cudaMemset(d_energy_, 0, sizeof(RealType)));
    // gpuErrchk(cudaMemset(d_grads_, 0, N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMemset(d_hessians_, 0, N_*3*N_*3*sizeof(RealType)));
    // gpuErrchk(cudaMemset(d_mixed_partials_, 0, P_*N_*3*sizeof(RealType)));


    cudaDeviceSynchronize();
}


template<typename RealType> 
Integrator<RealType>::~Integrator() {
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

template<typename RealType> 
std::vector<RealType> Integrator<RealType>::get_dxdp() const {
    std::vector<RealType> buf(P_*N_*3);
    gpuErrchk(cudaMemcpy(&buf[0], d_dxdp_t_, P_*N_*3*sizeof(RealType), cudaMemcpyDeviceToHost));
    return buf;
}

template<typename RealType> 
std::vector<RealType> Integrator<RealType>::get_noise() const {
    std::vector<RealType> buf(N_*3);
    gpuErrchk(cudaMemcpy(&buf[0], d_rng_buffer_, N_*3*sizeof(RealType), cudaMemcpyDeviceToHost));
    return buf;
};

template<typename RealType> 
std::vector<RealType> Integrator<RealType>::get_coordinates() const {
    std::vector<RealType> buf(N_*3);
    gpuErrchk(cudaMemcpy(&buf[0], d_x_t_, N_*3*sizeof(RealType), cudaMemcpyDeviceToHost));
    return buf;
};

template<typename RealType> 
std::vector<RealType> Integrator<RealType>::get_velocities() const {
    std::vector<RealType> buf(N_*3);
    gpuErrchk(cudaMemcpy(&buf[0], d_v_t_, N_*3*sizeof(RealType), cudaMemcpyDeviceToHost));
    return buf;
};

template<typename RealType> 
void Integrator<RealType>::set_coordinates(std::vector<RealType> x) {
    for(size_t i=0; i < x.size(); i++) {
        // std::cout << "SC: " << x[i] << std::endl;
    }
    gpuErrchk(cudaMemcpy(d_x_t_, &x[0], N_*3*sizeof(RealType), cudaMemcpyHostToDevice));
};

template<typename RealType> 
void Integrator<RealType>::set_velocities(std::vector<RealType> v) {
    for(size_t i=0; i < v.size(); i++) {
        // std::cout << "SV: " << v[i] << std::endl;
    }
    gpuErrchk(cudaMemcpy(d_v_t_, &v[0], N_*3*sizeof(RealType), cudaMemcpyHostToDevice));
};

// dangerous! not exception safe.
// template<typename RealType> 
// void Integrator<RealType>::step_cpu(
//     const RealType *h_grads,
//     const RealType *h_hessians,
//     const RealType *h_mixed_partials) {

//     gpuErrchk(cudaMemcpy(d_grads_, h_grads, N_*3*sizeof(RealType), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(d_hessians_, h_hessians, N_*3*N_*3*sizeof(RealType), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(d_mixed_partials_, h_mixed_partials, P_*N_*3*sizeof(RealType), cudaMemcpyHostToDevice));

//     step_gpu(d_grads_, d_hessians_, d_mixed_partials_);

//     cudaDeviceSynchronize();
// }


template<typename RealType> 
void LangevinOptimizer<RealType>::step(
    const int N,
    const int P,
    const RealType *dE_dx,
    const RealType *d2E_dx2,
    RealType *d2E_dxdp, // this is modified in place
    RealType *d_x_t,
    RealType *d_v_t,
    RealType *d_dx_dp_t,
    RealType *d_dv_dp_t) const {

    size_t tpb = 32;
    size_t n_blocks = (N_*3 + tpb - 1) / tpb;
    if(d_hessians != nullptr && d_mixed_partials != nullptr) {
        hessian_vector_product(d2E_dx2, d_dx_dp_t, d2E_dxdp);

        dim3 dimGrid(n_blocks, P, C); // x, y, z dims
        update_derivatives<RealType><<<n_blocks, tpb>>>(
            coeff_a_,
            d_coeff_bs_,
            d2E_dxdp,
            d_dxdp_t,
            d_dvdp_t,
            dt_,
            N*3
        );
        gpuErrchk(cudaPeekAtLastError());
    }
    curandErrchk(templateCurandNormal(cr_rng_, d_rng_buffer_, N_*3, 0.0, 1.0));
    reduce_velocities<RealType><<<n_blocks, tpb>>>(
        d_rng_buffer_,
        coeff_a_,
        d_coeff_bs_,
        d_coeff_cs_,
        d_grads,
        dt,
        d_x_t,
        d_v_t,
        N*3);
}




template<typename RealType> 
void Integrator<RealType>::step_gpu(
    const RealType *d_grads,
    const RealType *d_hessians,
    RealType *d_mixed_partials) {

    size_t tpb = 32;

    if(d_hessians != nullptr && d_mixed_partials != nullptr) {
        hessian_vector_product(d_hessians_, d_dxdp_t_, d_mixed_partials);
        size_t n_blocks = (P_*N_*3 + tpb - 1) / tpb;

        update_derivatives<RealType><<<n_blocks, tpb>>>(
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
    reduce_velocities<RealType><<<n_blocks, tpb>>>(
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

template<typename RealType> 
void Integrator<RealType>::hessian_vector_product(
    const RealType *d_A,
    RealType *d_B,
    RealType *d_C) {

    RealType alpha = 1.0;
    RealType beta  = 1.0;
 
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