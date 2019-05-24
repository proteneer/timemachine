#pragma once

#include "surreal.cuh"
// #include <cstdio>
/*

Consider an energy function function E(x,p): R^(3N), R^P -> R, this kernel computes
a four way kernel that conditionally computes in one pass. It is assumed that dp/dx
is zero but dx/dp may not necessarily be zero:

Notation:
---------
E: the energy (always computed)
dE_dx: derivative of the energy with respect to the coordinates (always computed)
dE_dp: derivative of the energy with respect to the parameters
d2E_dxdp: derivative of the gradient with respect to the parameters
dx_dp: derivatives of the geometry with respect to the parameters

Commonly used patterns
---------
E + dE_dp: re-parameterization-gradient (this is not asymptotically efficient if dx_dp == 0)
E + dE_dx: inference-mode when needed to run a simple simulation
E + dE_dx + d2E_dxdp: timemachine time travelling (this is not asymptotically efficient if dx_dp == 0)

*/

template<typename RealType>
void __global__ k_harmonic_bond_derivatives(
    const int num_atoms,    // n
    const int num_params,   // p
    const RealType *coords, // [n, 3]
    const RealType *params, // [p,]
    const int num_bonds,    // b
    const int *bond_idxs,   // [b, 2]
    const int *param_idxs,  // [b, 2]
    RealType *E,            // [,] never null
    RealType *dE_dx,        // [n,3] or null
    // parameters used for computing derivatives
    const RealType *dx_dp, // [dp, n, 3]
    const int *dp_idxs,     // of shape [dp] or null, if null then we don't compute parameter derivatives
    RealType *dE_dp,        // [dp,] or null
    RealType *d2E_dxdp      // [dp, n, 3] or null
) {

    // note that dp == gridDim.y
    const bool compute_dp = (dp_idxs != nullptr);
    auto b_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(b_idx >= num_bonds) {
        return;
    }

    int src_idx = bond_idxs[b_idx*2 + 0];
    int dst_idx = bond_idxs[b_idx*2 + 1];

    RealType rx0 = coords[src_idx*3+0];
    RealType ry0 = coords[src_idx*3+1];
    RealType rz0 = coords[src_idx*3+2];

    RealType rx1 = coords[dst_idx*3+0];
    RealType ry1 = coords[dst_idx*3+1];
    RealType rz1 = coords[dst_idx*3+2];

    RealType ix0 = 0;
    RealType iy0 = 0;
    RealType iz0 = 0;

    RealType ix1 = 0;
    RealType iy1 = 0;
    RealType iz1 = 0;

    RealType rkb = params[param_idxs[b_idx*2+0]];
    RealType rb0 = params[param_idxs[b_idx*2+1]];

    RealType ikb = 0;
    RealType ib0 = 0;

    const auto dp_idx = blockIdx.y;

    // (ytz): this a complex step size, not a standard finite difference step size.
    // the error decays quadratically as opposed to linearly w.r.t. step size.
    const RealType step_size = 1e-7;

    if(compute_dp) {

        // complex-step coordinates
        if(dx_dp != nullptr) {
            ix0 = dx_dp[dp_idx*num_atoms*3+src_idx*3+0]*step_size;
            iy0 = dx_dp[dp_idx*num_atoms*3+src_idx*3+1]*step_size;
            iz0 = dx_dp[dp_idx*num_atoms*3+src_idx*3+2]*step_size;

            ix1 = dx_dp[dp_idx*num_atoms*3+dst_idx*3+0]*step_size;
            iy1 = dx_dp[dp_idx*num_atoms*3+dst_idx*3+1]*step_size;
            iz1 = dx_dp[dp_idx*num_atoms*3+dst_idx*3+2]*step_size;
        }

        // complex-step parameters 
        if(param_idxs[b_idx*2+0] == dp_idxs[dp_idx]){
            ikb = step_size;
        }
        if(param_idxs[b_idx*2+1] == dp_idxs[dp_idx]){
            ib0 = step_size;
        }   
    }

    Surreal<RealType> x0(rx0, ix0);
    Surreal<RealType> y0(ry0, iy0);
    Surreal<RealType> z0(rz0, iz0);

    Surreal<RealType> x1(rx1, ix1);
    Surreal<RealType> y1(ry1, iy1);
    Surreal<RealType> z1(rz1, iz1);

    Surreal<RealType> dx = x0 - x1;
    Surreal<RealType> dy = y0 - y1;
    Surreal<RealType> dz = z0 - z1;

    Surreal<RealType> kb(rkb, ikb);
    Surreal<RealType> b0(rb0, ib0);

    Surreal<RealType> d2x = dx*dx;
    Surreal<RealType> d2y = dy*dy;
    Surreal<RealType> d2z = dz*dz;

    Surreal<RealType> d2ij = d2x + d2y + d2z;
    Surreal<RealType> dij = sqrt(d2ij);
    Surreal<RealType> db = dij - b0;

    Surreal<RealType> src_grad_dx = kb*db*dx/dij;
    Surreal<RealType> src_grad_dy = kb*db*dy/dij;
    Surreal<RealType> src_grad_dz = kb*db*dz/dij;

    Surreal<RealType> dst_grad_dx = -src_grad_dx;
    Surreal<RealType> dst_grad_dy = -src_grad_dy;
    Surreal<RealType> dst_grad_dz = -src_grad_dz;

    Surreal<RealType> energy = 0.5*kb*db*db;

    // E is never null
    if(blockIdx.y == 0) {
        atomicAdd(E, energy.real);
    }

    if(blockIdx.y == 0 && dE_dx != nullptr) {
        atomicAdd(dE_dx + src_idx*3 + 0, src_grad_dx.real);
        atomicAdd(dE_dx + src_idx*3 + 1, src_grad_dy.real);
        atomicAdd(dE_dx + src_idx*3 + 2, src_grad_dz.real);

        atomicAdd(dE_dx + dst_idx*3 + 0, dst_grad_dx.real);
        atomicAdd(dE_dx + dst_idx*3 + 1, dst_grad_dy.real);
        atomicAdd(dE_dx + dst_idx*3 + 2, dst_grad_dz.real);
    }

    if(compute_dp && dE_dp != nullptr) {
        atomicAdd(dE_dp + dp_idx, energy.imag/step_size);
    }

    if(compute_dp && d2E_dxdp != nullptr) {
        atomicAdd(d2E_dxdp + dp_idx*num_atoms*3 + src_idx*3 + 0, src_grad_dx.imag/step_size);
        atomicAdd(d2E_dxdp + dp_idx*num_atoms*3 + src_idx*3 + 1, src_grad_dy.imag/step_size);
        atomicAdd(d2E_dxdp + dp_idx*num_atoms*3 + src_idx*3 + 2, src_grad_dz.imag/step_size);

        atomicAdd(d2E_dxdp + dp_idx*num_atoms*3 + dst_idx*3 + 0, dst_grad_dx.imag/step_size);
        atomicAdd(d2E_dxdp + dp_idx*num_atoms*3 + dst_idx*3 + 1, dst_grad_dy.imag/step_size);
        atomicAdd(d2E_dxdp + dp_idx*num_atoms*3 + dst_idx*3 + 2, dst_grad_dz.imag/step_size);
    }

}
