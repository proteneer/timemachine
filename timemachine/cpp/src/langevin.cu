#include "cublas_v2.h"
#include "curand.h"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdio>

#include "langevin.hpp"
#include "gpu_utils.cuh"


template <typename RealType>
__global__ void update_positions(
    const RealType *noise,
    const RealType coeff_a,
    const RealType *coeff_bs, // N x 3, not P x N x 3, but we could just pass in the first index
    const RealType *coeff_cs,
    const RealType *dE_dx,
    const RealType d_t,
    const int N,
    const int D,
    RealType *x_t,
    RealType *v_t) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;

    // only integrate first three dimensions
    if(d_idx >= 3) {
        return;
    }

    int local_idx = atom_idx*D + d_idx;

    v_t[local_idx] = coeff_a*v_t[local_idx] - coeff_bs[atom_idx]*dE_dx[local_idx] + coeff_cs[atom_idx]*noise[local_idx];
    x_t[local_idx] += v_t[local_idx]*d_t;

}


template<typename RealType>
__global__ void update_derivatives(
    const RealType coeff_a,
    const RealType *coeff_bs, // shape N
    const RealType *d2E_dxdp, 
    const RealType dt,
    const int N,
    const int D,
    RealType *dx_dp_t,
    RealType *dv_dp_t) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    // only integrate first three dimensions
    int d_idx = blockIdx.y;
    if(d_idx >= 3) {
        return;
    }
    int p_idx = blockIdx.z;
    int local_idx = p_idx*N*D + atom_idx*D + d_idx;

    // derivative of the above equation
    RealType tmp = coeff_a*dv_dp_t[local_idx] - coeff_bs[atom_idx]*d2E_dxdp[local_idx];
    dv_dp_t[local_idx] = tmp;
    dx_dp_t[local_idx] += dt*tmp;

}


namespace timemachine {


template<typename RealType> 
LangevinOptimizer<RealType>::LangevinOptimizer(
    RealType dt,
    const int num_dims,
    const RealType coeff_a,
    const std::vector<RealType> &coeff_bs,
    const std::vector<RealType> &coeff_cs) :
    dt_(dt),
    coeff_a_(coeff_a),
    d_rng_buffer_(nullptr) {

    gpuErrchk(cudaMalloc((void**)&d_coeff_bs_, coeff_bs.size()*sizeof(RealType)));
    gpuErrchk(cudaMalloc((void**)&d_coeff_cs_, coeff_cs.size()*sizeof(RealType)));

    gpuErrchk(cudaMemcpy(d_coeff_bs_, &coeff_bs[0], coeff_bs.size()*sizeof(RealType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeff_cs_, &coeff_cs[0], coeff_cs.size()*sizeof(RealType), cudaMemcpyHostToDevice));

    cublasErrchk(cublasCreate(&cb_handle_));
    curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_PHILOX4_32_10));

    gpuErrchk(cudaMalloc((void**)&d_rng_buffer_, coeff_bs.size()*num_dims*sizeof(RealType)));

    curandSetPseudoRandomGeneratorSeed(cr_rng_, time(NULL));

}


template<typename RealType> 
LangevinOptimizer<RealType>::~LangevinOptimizer() {
    gpuErrchk(cudaFree(d_coeff_bs_));
    gpuErrchk(cudaFree(d_coeff_cs_));
    gpuErrchk(cudaFree(d_rng_buffer_));

    cublasErrchk(cublasDestroy(cb_handle_));
    curandErrchk(curandDestroyGenerator(cr_rng_));
}

template<typename RealType> 
void LangevinOptimizer<RealType>::step(
    const int N,
    const int D,
    const int DP,
    const RealType *dE_dx,
    const RealType *d2E_dx2,
    RealType *d2E_dxdp, // this is modified in place
    RealType *d_x_t,
    RealType *d_v_t,
    RealType *d_dx_dp_t,
    RealType *d_dv_dp_t,
    const RealType *d_input_noise_buffer) const {

    size_t tpb = 32;
    size_t n_blocks = (N*D + tpb - 1) / tpb;
    if(d2E_dx2 != nullptr && d2E_dxdp != nullptr) {
        hessian_vector_product(N, D, DP, d2E_dx2, d_dx_dp_t, d2E_dxdp);

        dim3 dimGrid_dxdp(n_blocks, D, DP); // x, y, z dims
        update_derivatives<RealType><<<dimGrid_dxdp, tpb>>>(
            coeff_a_,
            d_coeff_bs_,
            d2E_dxdp,
            dt_,
            N,
            D,
            d_dx_dp_t,
            d_dv_dp_t
        );
        gpuErrchk(cudaPeekAtLastError());
    }

    const RealType* d_noise_buf = nullptr;

    if(d_input_noise_buffer == nullptr) {
        curandErrchk(templateCurandNormal(cr_rng_, d_rng_buffer_, N*D, 0.0, 1.0));
        d_noise_buf = d_rng_buffer_;
    } else {
        d_noise_buf = d_input_noise_buffer;
    }

    dim3 dimGrid_dx(n_blocks, D);
    update_positions<RealType><<<dimGrid_dx, tpb>>>(
        d_noise_buf,
        coeff_a_,
        d_coeff_bs_,
        d_coeff_cs_,
        dE_dx,
        dt_,
        N,
        D,
        d_x_t,
        d_v_t
    );

    gpuErrchk(cudaPeekAtLastError());

}

template<typename RealType> 
void LangevinOptimizer<RealType>::hessian_vector_product(
    const int N,
    const int D,
    const int DP,
    const RealType *d_A,
    RealType *d_B,
    RealType *d_C) const {

    RealType alpha = 1.0;
    RealType beta  = 1.0;
 
    const size_t ND = N*D;

    // this is set to UPPER because of fortran ordering
    cublasErrchk(templateSymm(cb_handle_,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
        ND, DP,
        &alpha,
        d_A, ND,
        d_B, ND,
        &beta,
        d_C, ND));

}

template<typename RealType>
void LangevinOptimizer<RealType>::set_coeff_a(RealType a) {
    coeff_a_ = a;
}

template<typename RealType>
void LangevinOptimizer<RealType>::set_coeff_b(int num_atoms, const RealType *cb) {
    gpuErrchk(cudaMemcpy(d_coeff_bs_, cb, num_atoms*sizeof(RealType), cudaMemcpyHostToDevice));
}

template<typename RealType>
void LangevinOptimizer<RealType>::set_coeff_c(int num_atoms, const RealType *cc) {
    gpuErrchk(cudaMemcpy(d_coeff_cs_, cc, num_atoms*sizeof(RealType), cudaMemcpyHostToDevice));
}

template<typename RealType>
void LangevinOptimizer<RealType>::set_dt(RealType ndt) {
    dt_ = ndt;
}

}

template class timemachine::LangevinOptimizer<double>;
template class timemachine::LangevinOptimizer<float>;