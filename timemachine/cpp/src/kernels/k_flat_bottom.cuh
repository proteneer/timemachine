
#include "surreal.cuh"
#include "../fixed_point.hpp"


__device__ const double PI = 3.14159265358979323846;

template<typename RealType>
void __global__ k_flat_bottom_inference(
    const int B,     // number of bonds
    const double *coords,  // [n, 3]
    const double *params,  // [p,]
    const double lambda,
    const int *bond_idxs,    // [b, 2]
    const int *param_idxs,   // [b, 2]
    const int *lambda_flags,
    unsigned long long *grad_coords,
    double *du_dl,
    double *energy) {

    // tbd flat bottom

    const auto b_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(b_idx >= B) {
        return;
    }

    double f_lambda = sin(lambda*PI/2);
    f_lambda = f_lambda*f_lambda;
    double df = PI*sin((PI*lambda)/2)*cos((PI*lambda)/2);

    // zero out otherwise
    if(lambda_flags[b_idx] == 0) {
        f_lambda = 1;
        df = 0;
    }

    int src_idx = bond_idxs[b_idx*2+0];
    int dst_idx = bond_idxs[b_idx*2+1];

    RealType dx[3];
    RealType d2ij = 0; // initialize your summed variables!
    for(int d=0; d < 3; d++) {
        RealType delta = coords[src_idx*3+d] - coords[dst_idx*3+d];
        dx[d] = delta;
        d2ij += delta*delta;
    }

    int kb_idx = param_idxs[b_idx*2+0];
    int b0_idx = param_idxs[b_idx*2+1];

    RealType kb = params[kb_idx];
    RealType b0 = params[b0_idx];

    RealType dij = sqrt(d2ij);
    RealType db = dij - b0;

    for(int d=0; d < 3; d++) {
        RealType grad_delta = f_lambda*kb*db*dx[d]/dij;
        atomicAdd(grad_coords + src_idx*3 + d, static_cast<unsigned long long>((long long) (grad_delta*FIXED_EXPONENT)));
        atomicAdd(grad_coords + dst_idx*3 + d, static_cast<unsigned long long>((long long) (-grad_delta*FIXED_EXPONENT)));
    }

    atomicAdd(du_dl, df*kb/2*db*db);
    atomicAdd(energy, f_lambda*kb/2*db*db);

}