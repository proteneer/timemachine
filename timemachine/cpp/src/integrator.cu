#include "integrator.hpp"
#include "fixed_point.hpp"
#include "kernel_utils.cuh"
#include <cstdio>

namespace timemachine {

template <typename RealType>
__global__ void update_forward(
    const int N,
    const int D,
    const RealType coeff_a,
    const RealType *d_coeff_bs, // N x 3, not P x N x 3, but we could just pass in the first index
    const RealType *d_x_t_old,
    const RealType *d_v_t_old,
    const unsigned long long *d_dE_dx_old,
    const RealType dt,
    RealType *d_x_t_new,
    RealType *d_v_t_new) {

    int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(atom_idx >= N) {
        return;
    }

    int d_idx = blockIdx.y;
    int local_idx = atom_idx*D + d_idx;

    auto force = static_cast<RealType>(static_cast<long long>(d_dE_dx_old[local_idx]))/FIXED_EXPONENT;
    auto new_v_t = coeff_a*d_v_t_old[local_idx] + d_coeff_bs[atom_idx]*force;
    d_v_t_new[local_idx] = new_v_t; 
    d_x_t_new[local_idx] = d_x_t_old[local_idx] + new_v_t*dt;

};

template<typename RealType>
void step_forward(
    int N,
    int D,
    const RealType ca,
    const RealType *d_coeff_bs,
    const RealType *d_x_old,
    const RealType *d_v_old,
    const unsigned long long *d_dE_dx_old,
    const RealType dt,
    RealType *d_x_new,
    RealType *d_v_new) {

    size_t tpb = 32;
    size_t n_blocks = (N*D + tpb - 1) / tpb;
    dim3 dimGrid_dx(n_blocks, D);

    update_forward<<<dimGrid_dx, tpb>>>(
        N,
        D,
        ca,
        d_coeff_bs,
        d_x_old,
        d_v_old,
        d_dE_dx_old,
        dt,
        d_x_new,
        d_v_new
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

};

template void step_forward<double>(
    int N,
    int D,
    const double ca,
    const double *d_coeff_bs,
    const double *d_x_old,
    const double *d_v_old,
    const unsigned long long *d_dE_dx_old,
    const double dt,
    double *d_x_new,
    double *d_v_new);

}