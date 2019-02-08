
#include "cublas_v2.h"

// #include <ctime>
#include <vector>
#include <stdexcept>
#include <cstdio>

#include "integrator.hpp"

#include "gpu_utils.cuh"


/*

Buffer operations:

Let W be the number of windows

1. Compute the total derivative Dx_t, using dxdp_t, hessians, and mixed partials
2. Add window_t into the converged window sums
3. Replace window_t with Dx_t
4. Reduce over all the windows and the converged buffer.
5. Update dx/dp_t to t+1

Each thread processes 1 out of [P,N,3] elements.

*/

template<typename NumericType>
__global__ void reduce_total(
    NumericType coeff_a,
    const NumericType *coeff_bs,
    const NumericType *Dx_t,
    NumericType *total_buffer,
    NumericType *converged_buffer,        
    NumericType *dxdp_t,
    int t, // starting window slot
    int W, // number of windows
    int PN3 // PN3
) {

    // 1. Done by SGEMM call
    int local_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(local_idx >= PN3) {
        return;
    }

    // 2. Add total_buffer[t] into converged buffer
    int window_idx = t * PN3 + blockIdx.x * blockDim.x + threadIdx.x;
    converged_buffer[local_idx] += total_buffer[window_idx];

    // 3. Replace window_t with Dx_t
    total_buffer[window_idx] = Dx_t[local_idx];

    // 4. Reduce over all the windows.
    NumericType prefactor = 0.0;
    NumericType a_n = 1.0;
    NumericType accum = 0.0;

    //      iter i
    // k=0  0 3 2 1
    // k=1  1 0 3 2
    // k=2  1 0 3 2
    // k=3  2 1 0 3
    for(int i=0; i < W; i++) {
        int slot = t - i < 0 ? t - i + W : t - i;
        int slot_idx = slot*PN3 + blockIdx.x*blockDim.x + threadIdx.x;
        prefactor += a_n;
        a_n *= coeff_a;
        // printf("%d %d %f\n", i, blockIdx.x*blockDim.x + threadIdx.x, total_buffer[slot_idx]);
        // printf("%d %d %f\n", i, blockIdx.x*blockDim.x + threadIdx.x, total_buffer[slot_idx]);

        if(local_idx == 0) {
            printf("w: %d, pref: %f, tot_buf: %f\n", i, prefactor, total_buffer[slot_idx]);
        }

        accum += prefactor*total_buffer[slot_idx];
    }

    // 5. Compute new dxdp_t
    // (ytz). coeff_b's can be optimized into smaller chunks.
    dxdp_t[local_idx] = -coeff_bs[local_idx] * (accum + prefactor * converged_buffer[local_idx]);

}

#include <iostream>
namespace timemachine {


template<typename NumericType> 
std::vector<NumericType> Integrator<NumericType>::get_dxdp() const {
    std::vector<NumericType> cpu_dxdp(P_*N_*3);
    gpuErrchk(cudaMemcpy(&cpu_dxdp[0], d_dxdp_t_, P_*N_*3*sizeof(NumericType), cudaMemcpyDeviceToHost));
    return cpu_dxdp;
}

template<typename NumericType> 
Integrator<NumericType>::Integrator(
    NumericType coeff_a,
    int W,
    int N,
    int P,
    const std::vector<NumericType> &coeff_bs) :
    W_(W),
    N_(N),
    P_(P),
    step_(0),
    coeff_a_(coeff_a) {

    if(coeff_bs.size() != N) {
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

    // 1. Allocate memory on the GPU
    gpuErrchk(cudaMalloc((void**)&d_dxdp_t_, P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_total_buffer_, W_*P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_converged_buffer_, P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_coeff_bs_, P_*N_*3*sizeof(NumericType)));

    gpuErrchk(cudaMalloc((void**)&d_hessians_, N_*3*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMalloc((void**)&d_mixed_partials_, P_*N_*3*sizeof(NumericType)));
    // 2. Memset
    gpuErrchk(cudaMemset(d_dxdp_t_, 0.0, P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_total_buffer_, 0.0, W_*P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemset(d_converged_buffer_, 0.0, P_*N_*3*sizeof(NumericType)));
    gpuErrchk(cudaMemcpy(d_coeff_bs_, &expanded_coeff_bs[0], P_*N_*3*sizeof(NumericType), cudaMemcpyHostToDevice));

    cublasErrchk(cublasCreate(&cb_handle_));

}


template<typename NumericType> 
Integrator<NumericType>::~Integrator() {
    gpuErrchk(cudaFree(d_dxdp_t_));
    gpuErrchk(cudaFree(d_total_buffer_));
    gpuErrchk(cudaFree(d_converged_buffer_));
    gpuErrchk(cudaFree(d_coeff_bs_));

    gpuErrchk(cudaFree(d_hessians_));
    gpuErrchk(cudaFree(d_mixed_partials_));

    cublasErrchk(cublasDestroy(cb_handle_));
}


// dangerous! not exception safe.
template<typename NumericType> 
void Integrator<NumericType>::step_cpu(
    const NumericType *h_hessians,
    const NumericType *h_mixed_partials) {

    gpuErrchk(cudaMemcpy(d_hessians_, h_hessians, N_*3*N_*3*sizeof(NumericType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mixed_partials_, h_mixed_partials, P_*N_*3*sizeof(NumericType), cudaMemcpyHostToDevice));

    step_gpu(d_hessians_, d_mixed_partials_);
}

template<typename NumericType> 
void Integrator<NumericType>::step_gpu(
    const NumericType *d_hessians,
    NumericType *d_mixed_partials) {

    hessian_vector_product(d_hessians_, d_dxdp_t_, d_mixed_partials);

    // std::vector<NumericType> debug(P_*N_*3);
    // gpuErrchk(cudaMemcpy(&debug[0], d_mixed_partials, sizeof(float)*P_*N_*3, cudaMemcpyDeviceToHost));
    // for(size_t i=0; i < P_*N_*3; i++) {
    //     std::cout << "hvp:" << debug[i] << " " << std::endl;
    // }

    std::cout << "STEPPING INTO: " << step_ % W_ << std::endl;

    reduce_buffers(d_mixed_partials_, step_ % W_);

    step_ += 1;

}

template<typename NumericType> 
void Integrator<NumericType>::reduce_buffers(const NumericType *d_Dx_t, int window_k) {

    size_t tpb = 32;
    const size_t tot = P_*N_*3;
    size_t n_blocks = (tot + tpb - 1) / tpb;

    reduce_total<NumericType><<<n_blocks, tpb>>>(
        coeff_a_,
        d_coeff_bs_,
        d_Dx_t,
        d_total_buffer_,
        d_converged_buffer_,
        d_dxdp_t_,
        window_k,
        W_,
        P_*N_*3
    );

    gpuErrchk(cudaPeekAtLastError());

};

template<typename NumericType> 
void Integrator<NumericType>::hessian_vector_product(
    const NumericType *d_A,
    NumericType *d_B,
    NumericType *d_C) {

    NumericType alpha = 1.0;
    NumericType beta  = 1.0;
 
    const size_t N3 = N_*3;

    // replace with SGEMM later
    cublasErrchk(cublasDgemm(cb_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N, // whether or not we transpose A
        N3, P_, N3,
        &alpha,
        d_A, N3,
        d_B, N3,
        &beta,
        d_C, N3));
}

}

// template class timemachine::Integrator<float>;
template class timemachine::Integrator<double>;