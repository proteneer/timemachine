#pragma once

#include "k_fixed_point.cuh"

template<typename RealType>
void __global__ k_calc_centroid(
    const double * __restrict__ d_coords,  // [n, 3]
    const int * __restrict__ d_group_a_idxs,
    const int * __restrict__ d_group_b_idxs,
    const int N_A,
    const int N_B,
    unsigned long long * d_centroid_a, // [3]
    unsigned long long * d_centroid_b // [3]
    ){
    int t_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (N_A + N_B <= t_idx) {
        return;
    }
    int group_idx = t_idx < N_A ? t_idx : t_idx - N_A;
    unsigned long long * centroid = t_idx < N_A ? d_centroid_a : d_centroid_b;
    const int * cur_array = t_idx < N_A ? d_group_a_idxs : d_group_b_idxs;

    #pragma unroll
    for(int d=0; d < 3; d++) {
        atomicAdd(centroid + d, FLOAT_TO_FIXED<RealType>(d_coords[cur_array[group_idx]*3+d]));
    }
 }

template<typename RealType>
void __global__ k_centroid_restraint(
    // const int N,     // number of bonds, ignore for now
    const double * __restrict__ d_coords,  // [n, 3]
    const int * __restrict__ d_group_a_idxs,
    const int * __restrict__ d_group_b_idxs,
    const int N_A,
    const int N_B,
    const unsigned long long * __restrict__ d_centroid_a, // [3]
    const unsigned long long * __restrict__ d_centroid_b, // [3]
    // const double *d_masses, // ignore d_masses for now
    const double kb,
    const double b0,
    unsigned long long *d_du_dx,
    unsigned long long *d_u) {

    const int t_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (N_A + N_B <= t_idx) {
        return;
    }
    int group_idx = t_idx < N_A ? t_idx : t_idx - N_A;
    int count = t_idx < N_A ? N_A : N_B;
    const int * cur_array = t_idx < N_A ? d_group_a_idxs : d_group_b_idxs;
    RealType sign = t_idx < N_A ? 1.0 : -1.0;
    RealType dij = 0;
    RealType deltas[3];
    #pragma unroll
    for (int d=0; d < 3; d++) {
        deltas[d] = FIXED_TO_FLOAT<RealType>(d_centroid_a[d]) / N_A - FIXED_TO_FLOAT<RealType>(d_centroid_b[d]) / N_B;
        dij += deltas[d]*deltas[d];
    }
    dij = sqrt(dij);

    if(t_idx == 0 && d_u) {
        RealType nrg = kb*(dij-b0)*(dij-b0);
        d_u[0] = FLOAT_TO_FIXED<RealType>(nrg);
    }

    // grads
    if(d_du_dx) {
        #pragma unroll
        for(int d=0; d < 3; d++) {
            if(b0 != 0) {
                RealType du_ddij = 2*kb*(dij-b0);
                RealType ddij_dxi = deltas[d]/dij;
                RealType delta = sign*du_ddij*ddij_dxi/count;
                atomicAdd(d_du_dx + cur_array[group_idx]*3 + d, FLOAT_TO_FIXED<RealType>(delta));
            } else {
                RealType delta = sign*2*kb*deltas[d]/count;
                atomicAdd(d_du_dx + cur_array[group_idx]*3 + d, FLOAT_TO_FIXED<RealType>(delta));
            }
        }
    }
}
