#pragma once

#include "surreal.cuh"
#include "stdio.h"

/*

Consider an energy function function E(x,p): R^(3N), R^P -> R, this kernel computes
a four way kernel that conditionally computes in one pass. It is assumed that dp/dx
is zero but dx/dp may not necessarily be zero:

E: the energy
dE_dp: derivative of the energy with respect to the parameters
dE_dx: derivative of the energy with respect to the coordinates
d2E_dxdp: derivative of the gradient with respect to the parameters

*/
template<typename RealType>
void __global__ harmonic_bond_derivatives(
    const int num_atoms,
    const int num_params,
    const RealType *coords, // [n, 3]
    const RealType *params, // [p,]
    const RealType *dxdps, // [p, n, 3]
    const int num_bonds,
    const int *bond_idxs, // [b, 2]
    const int *param_idxs, // [b, 2]
    RealType *E, // [,]
    RealType *dE_dp, // [p,] 
    RealType *dE_dx, // [n,3]
    RealType *d2E_dxdp // [p, n, 3]
) {

    // ytz: this a complex step size, not a standard finite difference step size.
    // the error decays quadratically as opposed to linearly w.r.t. step size.
    const RealType step_size = 1e-7;

    // blockDim.y indexes into the parameter we're computing
    // blockDim.x indexes into the particular bond we're indexing

    auto p_idx = blockIdx.y;
    auto b_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(p_idx >= num_params) {
        return;
    }

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

    if(dxdps != nullptr) {
        ix0 = dxdps[p_idx*num_atoms*3+src_idx*3+0]*step_size;
        iy0 = dxdps[p_idx*num_atoms*3+src_idx*3+1]*step_size;
        iz0 = dxdps[p_idx*num_atoms*3+src_idx*3+2]*step_size;

        ix1 = dxdps[p_idx*num_atoms*3+dst_idx*3+0]*step_size;
        iy1 = dxdps[p_idx*num_atoms*3+dst_idx*3+1]*step_size;
        iz1 = dxdps[p_idx*num_atoms*3+dst_idx*3+2]*step_size;
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

    RealType rkb = params[param_idxs[b_idx*2+0]];
    RealType rb0 = params[param_idxs[b_idx*2+1]];

    RealType ikb = 0;
    RealType ib0 = 0;

    // apply complex step.
    if(param_idxs[b_idx*2+0] == p_idx){
        ikb = step_size;
    }
    if(param_idxs[b_idx*2+1] == p_idx){
        ib0 = step_size;
    }

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

    if(E != nullptr && p_idx == 0) {
        atomicAdd(E, energy.real);
    }

    if(dE_dx != nullptr && p_idx == 0){
        atomicAdd(dE_dx + src_idx*3 + 0, src_grad_dx.real);
        atomicAdd(dE_dx + src_idx*3 + 1, src_grad_dy.real);
        atomicAdd(dE_dx + src_idx*3 + 2, src_grad_dz.real);

        atomicAdd(dE_dx + dst_idx*3 + 0, dst_grad_dx.real);
        atomicAdd(dE_dx + dst_idx*3 + 1, dst_grad_dy.real);
        atomicAdd(dE_dx + dst_idx*3 + 2, dst_grad_dz.real);
    }

    if(dE_dp != nullptr) {
        atomicAdd(dE_dp + p_idx, energy.imag/step_size);
    }

    if(d2E_dxdp != nullptr) {
        atomicAdd(d2E_dxdp + p_idx*num_atoms*3 + src_idx*3 + 0, src_grad_dx.imag/step_size);
        atomicAdd(d2E_dxdp + p_idx*num_atoms*3 + src_idx*3 + 1, src_grad_dy.imag/step_size);
        atomicAdd(d2E_dxdp + p_idx*num_atoms*3 + src_idx*3 + 2, src_grad_dz.imag/step_size);

        atomicAdd(d2E_dxdp + p_idx*num_atoms*3 + dst_idx*3 + 0, dst_grad_dx.imag/step_size);
        atomicAdd(d2E_dxdp + p_idx*num_atoms*3 + dst_idx*3 + 1, dst_grad_dy.imag/step_size);
        atomicAdd(d2E_dxdp + p_idx*num_atoms*3 + dst_idx*3 + 2, dst_grad_dz.imag/step_size);
    }

}
