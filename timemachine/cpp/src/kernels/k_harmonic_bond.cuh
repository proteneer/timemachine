#include "../fixed_point.hpp"

template<typename RealType>
void __global__ k_harmonic_bond(
    const int B,           // number of bonds
    const double *coords,
    const double *params,  // [p, 2]
    const int *bond_idxs,  // [b, 2]
    unsigned long long *du_dx,
    double *du_dp,
    double *u) {

    const auto b_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(b_idx >= B) {
        return;
    }

    int src_idx = bond_idxs[b_idx*2+0];
    int dst_idx = bond_idxs[b_idx*2+1];

    RealType dx[3];
    RealType d2ij = 0;
    for(int d=0; d < 3; d++) {
        RealType delta = coords[src_idx*3+d] - coords[dst_idx*3+d];
        dx[d] = delta;
        d2ij += delta*delta;
    }

    int kb_idx = b_idx*2+0;
    int b0_idx = b_idx*2+1;

    RealType kb = params[kb_idx];
    RealType b0 = params[b0_idx];

    RealType dij = sqrt(d2ij);
    RealType db = dij - b0;

    if(du_dx) {
        for(int d=0; d < 3; d++) {
            RealType grad_delta = kb*db*dx[d]/dij;
            atomicAdd(du_dx + src_idx*3 + d, static_cast<unsigned long long>((long long) (grad_delta*FIXED_EXPONENT)));
            atomicAdd(du_dx + dst_idx*3 + d, static_cast<unsigned long long>((long long) (-grad_delta*FIXED_EXPONENT)));
        }
    }

    if(du_dp) {
        atomicAdd(du_dp + kb_idx, 0.5*db*db);
        atomicAdd(du_dp + b0_idx, -kb*db);        
    }

    if(u) {
        atomicAdd(u, kb/2*db*db);
    }

}