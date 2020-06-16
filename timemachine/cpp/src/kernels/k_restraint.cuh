
#include "surreal.cuh"
#include "../fixed_point.hpp"


__device__ const double PI = 3.14159265358979323846;

template<typename RealType>
void __global__ k_restraint_inference(
    const int B,     // number of bonds
    const double *coords,  // [n, 3]
    const double *params,  // [p,]
    const double lambda,
    const int *bond_idxs,    // [b, 2]
    const int *param_idxs,   // [b, 3]
    const int *lambda_flags,
    unsigned long long *grad_coords,
    double *du_dl,
    double *energy) {

    // tbd flat bottom

    const auto b_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(b_idx >= B) {
        return;
    }

    // double f_lambda = sin(lambda*PI/2);
    // f_lambda = f_lambda*f_lambda;
    // double df = PI*sin((PI*lambda)/2)*cos((PI*lambda)/2);

    double f_lambda = lambda;
    double df = 1;

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

    int kb_idx = param_idxs[b_idx*3+0];
    int b0_idx = param_idxs[b_idx*3+1];
    int a0_idx = param_idxs[b_idx*3+2];

    RealType kb = params[kb_idx];
    RealType b0 = params[b0_idx];
    RealType a0 = params[a0_idx];

    RealType dij = sqrt(d2ij);
    RealType db = dij - b0;

    RealType prefactor = f_lambda*2*a0*exp(-a0*(-b0+dij))*(1-exp(-a0*(-b0+dij)))*kb;

    for(int d=0; d < 3; d++) {
        RealType grad_delta = prefactor*dx[d]/dij;
        atomicAdd(grad_coords + src_idx*3 + d, static_cast<unsigned long long>((long long) (grad_delta*FIXED_EXPONENT)));
        atomicAdd(grad_coords + dst_idx*3 + d, static_cast<unsigned long long>((long long) (-grad_delta*FIXED_EXPONENT)));
    }

    double term = (1-exp(-a0*db));
    double u = kb*term*term;

    atomicAdd(du_dl, df*u);
    atomicAdd(energy, f_lambda*u);

}


// template<typename RealType>
// void __global__ k_flat_bottom_jvp(
//     const int B,     // number of bonds
//     const double *coords,  
//     const double *coords_tangent,  
//     const double *params,  // [p,]
//     const double lambda_primal,
//     const double lambda_tangent,
//     const int *bond_idxs,    // [b, 2]
//     const int *param_idxs,   // [b, 2]
//     const int *lambda_flags,
//     double *grad_coords_primals,
//     double *grad_coords_tangents,
//     double *grad_params_primals,
//     double *grad_params_tangents) {

//     const auto b_idx = blockDim.x*blockIdx.x + threadIdx.x;

//     if(b_idx >= B) {
//         return;
//     }


//     Surreal<RealType> lambda(lambda_primal, lambda_tangent);
//     Surreal<RealType> f_lambda = sin(lambda*PI/2);
//     f_lambda = f_lambda*f_lambda;
//     Surreal<RealType> df = PI*sin((PI*lambda)/2)*cos((PI*lambda)/2);

//     if(lambda_flags[b_idx] == 0) {
//         f_lambda.real = 1;
//         f_lambda.imag = 0;
//         df.real = 0;
//         df.imag = 0;
//     }

//     int src_idx = bond_idxs[b_idx*2+0];
//     int dst_idx = bond_idxs[b_idx*2+1];

//     Surreal<RealType> dx[3];
//     Surreal<RealType> d2ij(0.0, 0.0); // initialize your summed variables!
//     for(int d=0; d < 3; d++) {
//         Surreal<RealType> delta;
//         delta.real = coords[src_idx*3+d] - coords[dst_idx*3+d];
//         delta.imag = coords_tangent[src_idx*3+d] - coords_tangent[dst_idx*3+d];
//         dx[d] = delta;
//         d2ij += delta*delta;
//     }

//     int kb_idx = param_idxs[b_idx*2+0];
//     int b0_idx = param_idxs[b_idx*2+1];

//     RealType kb = params[kb_idx];
//     RealType b0 = params[b0_idx];

//     Surreal<RealType> dij = sqrt(d2ij);
//     Surreal<RealType> db = dij - b0;

//     for(int d=0; d < 3; d++) {
//         Surreal<RealType> grad_delta = f_lambda*kb*db*dx[d]/dij;
//         atomicAdd(grad_coords_primals + src_idx*3 + d, grad_delta.real);
//         atomicAdd(grad_coords_primals + dst_idx*3 + d, -grad_delta.real);

//         atomicAdd(grad_coords_tangents + src_idx*3 + d, grad_delta.imag);
//         atomicAdd(grad_coords_tangents + dst_idx*3 + d, -grad_delta.imag);
//     }

//     // avoid writing out to the real parts if possible
//     atomicAdd(grad_params_primals + kb_idx, (f_lambda*0.5*db*db).real);
//     atomicAdd(grad_params_tangents + kb_idx, (f_lambda*0.5*db*db).imag);

//     atomicAdd(grad_params_primals + b0_idx, (-kb*db*f_lambda).real);
//     atomicAdd(grad_params_tangents + b0_idx, (-kb*db*f_lambda).imag);

// }
