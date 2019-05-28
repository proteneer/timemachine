#pragma once

#include "surreal.cuh"
#include "kernel_utils.cuh"
// #include <cstdio>
/*

Consider an energy function function E(x,p): R^(3N), R^P -> R, this kernel computes
the four terms in one pass.

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

Grid Dimensions
---------------
gridDim.x = main axis
gridDim.y = number of parameters we're differentiating with respect to
gridDim.z = number of conformations we want to compute gradients for

*/

#include <cstdio>

template<typename RealType>
void __global__ k_harmonic_bond_derivatives(
    const int num_atoms,    // n
    const int num_params,   // p
    const RealType *coords, // [C, n, 3]
    const RealType *params, // [p,]
    const int num_bonds,    // b
    const int *bond_idxs,   // [b, 2]
    const int *param_idxs,  // [b, 2]
    RealType *E,            // [,] never null
    RealType *dE_dx,        // [n,3] or null
    // parameters used for computing derivatives
    // const RealType *dx_dp, // [C, dp, n, 3] or null
    // const int *dp_idxs,     // of shape [dp] or null, if null then we don't ocompute parameter derivatives
    const int *param_gather_idxs, // [p,] if -1, then we discard
    RealType *dE_dp,        // [C, dp,] or null
    RealType *d2E_dx2,      // [C, n, 3, n, 3] or null, hessian
    RealType *d2E_dxdp      // [C, dp, n, 3] or null
) {

    // num conformations is indexed via gridDim.z
    auto conf_idx = blockIdx.z;

    // note that num_dp == gridDim.y
    const bool compute_dp = (dp_idxs != nullptr);
    auto b_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(bond_idx < num_bonds) {
        size_t src_idx = bond_idxs[bond_idx*2+0];
        size_t dst_idx = bond_idxs[bond_idx*2+1];

        RealType x0 = coords[src_idx*3+0];
        RealType y0 = coords[src_idx*3+1];
        RealType z0 = coords[src_idx*3+2];
        RealType x1 = coords[dst_idx*3+0];
        RealType y1 = coords[dst_idx*3+1];
        RealType z1 = coords[dst_idx*3+2];

        RealType dx = x0 - x1;
        RealType dy = y0 - y1;
        RealType dz = z0 - z1;

        RealType kb = params[param_idxs[bond_idx*2+0]];
        RealType b0 = params[param_idxs[bond_idx*2+1]];

        RealType d2x = dx*dx;
        RealType d2y = dy*dy;
        RealType d2z = dz*dz;

        RealType d2ij = d2x + d2y + d2z;
        RealType dij = sqrt(d2ij);

        RealType d3ij = dij*dij*dij;
        RealType db = dij - b0;

        RealType src_grad_dx = kb*db*dx/dij;
        RealType src_grad_dy = kb*db*dy/dij;
        RealType src_grad_dz = kb*db*dz/dij;
        RealType dst_grad_dx = -src_grad_dx;
        RealType dst_grad_dy = -src_grad_dy;
        RealType dst_grad_dz = -src_grad_dz;

        if(E) {
            atomicAdd(energy_out, kb/2.0*db*db);            
        }

        if(dE_dp) {
            
        }

        if(dE_dx) {
            atomicAdd(dE_dx + src_idx*3 + 0, src_grad_dx);
            atomicAdd(dE_dx + src_idx*3 + 1, src_grad_dy);
            atomicAdd(dE_dx + src_idx*3 + 2, src_grad_dz);
            atomicAdd(dE_dx + dst_idx*3 + 0, dst_grad_dx);
            atomicAdd(dE_dx + dst_idx*3 + 1, dst_grad_dy);
            atomicAdd(dE_dx + dst_idx*3 + 2, dst_grad_dz);
        }

        if(d2E_dx2) {
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 0*N*3 + src_idx*3 + 0, kb*(db*-dx*dx/d3ij + db/dij + d2x/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 0*N*3 + src_idx*3 + 1, kb*(db*dx*-dy/d3ij + dx*dy/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 0*N*3 + src_idx*3 + 2, kb*(db*dx*-dz/d3ij + dx*dz/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 0*N*3 + dst_idx*3 + 0, kb*(db*d2x/d3ij - db/dij + -dx*dx/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 0*N*3 + dst_idx*3 + 1, kb*(db*dx*dy/d3ij + dx*-dy/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 0*N*3 + dst_idx*3 + 2, kb*(db*dx*dz/d3ij + dx*-dz/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 1*N*3 + src_idx*3 + 0, kb*(db*-dx*dy/d3ij + dx*dy/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 1*N*3 + src_idx*3 + 1, kb*(db*-dy*dy/d3ij + db/dij + d2y/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 1*N*3 + src_idx*3 + 2, kb*(db*dy*-dz/d3ij + dy*dz/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 1*N*3 + dst_idx*3 + 0, kb*(db*dx*dy/d3ij + -dx*dy/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 1*N*3 + dst_idx*3 + 1, kb*(db*d2y/d3ij - db/dij + -dy*dy/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 1*N*3 + dst_idx*3 + 2, kb*(db*dy*dz/d3ij + dy*-dz/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 2*N*3 + src_idx*3 + 0, kb*(db*-dx*dz/d3ij + dx*dz/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 2*N*3 + src_idx*3 + 1, kb*(db*-dy*dz/d3ij + dy*dz/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 2*N*3 + src_idx*3 + 2, kb*(db*-dz*dz/d3ij + db/dij + d2z/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 2*N*3 + dst_idx*3 + 0, kb*(db*dx*dz/d3ij + -dx*dz/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 2*N*3 + dst_idx*3 + 1, kb*(db*dy*dz/d3ij + -dy*dz/d2ij));
            atomicAdd(d2E_dx2 + src_idx*3*N*3 + 2*N*3 + dst_idx*3 + 2, kb*(db*d2z/d3ij - db/dij + -dz*dz/d2ij));

            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 0*N*3 + src_idx*3 + 0, kb*(db*d2x/d3ij - db/dij + -dx*dx/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 0*N*3 + src_idx*3 + 1, kb*(db*-dx*-dy/d3ij + -dx*dy/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 0*N*3 + src_idx*3 + 2, kb*(db*-dx*-dz/d3ij + -dx*dz/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 0*N*3 + dst_idx*3 + 0, kb*(db*-dx*dx/d3ij + db/dij + d2x/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 0*N*3 + dst_idx*3 + 1, kb*(db*-dx*dy/d3ij + -dx*-dy/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 0*N*3 + dst_idx*3 + 2, kb*(db*-dx*dz/d3ij + -dx*-dz/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 1*N*3 + src_idx*3 + 0, kb*(db*-dx*-dy/d3ij + dx*-dy/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 1*N*3 + src_idx*3 + 1, kb*(db*d2y/d3ij - db/dij + -dy*dy/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 1*N*3 + src_idx*3 + 2, kb*(db*-dy*-dz/d3ij + -dy*dz/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 1*N*3 + dst_idx*3 + 0, kb*(db*dx*-dy/d3ij + -dx*-dy/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 1*N*3 + dst_idx*3 + 1, kb*(db*-dy*dy/d3ij + db/dij + d2y/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 1*N*3 + dst_idx*3 + 2, kb*(db*-dy*dz/d3ij + -dy*-dz/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 2*N*3 + src_idx*3 + 0, kb*(db*-dx*-dz/d3ij + dx*-dz/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 2*N*3 + src_idx*3 + 1, kb*(db*-dy*-dz/d3ij + dy*-dz/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 2*N*3 + src_idx*3 + 2, kb*(db*d2z/d3ij - db/dij + -dz*dz/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 2*N*3 + dst_idx*3 + 0, kb*(db*dx*-dz/d3ij + -dx*-dz/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 2*N*3 + dst_idx*3 + 1, kb*(db*dy*-dz/d3ij + -dy*-dz/d2ij));
            atomicAdd(d2E_dx2 + dst_idx*3*N*3 + 2*N*3 + dst_idx*3 + 2, kb*(db*-dz*dz/d3ij + db/dij + d2z/d2ij));            
        }

        if(d2E_dxdp) {
            int b0_gather_idx = param_gather_idxs[param_idxs[bond_idx*2+0]];

            if(b0_gather_idx >= 0) {
                atomicAdd(d2E_dxdp + b0_gather_idx*N*3 + src_idx*3 + 0, -kb*(x0 - x1)/dij);
                atomicAdd(d2E_dxdp + b0_gather_idx*N*3 + src_idx*3 + 1, -kb*(y0 - y1)/dij);
                atomicAdd(d2E_dxdp + b0_gather_idx*N*3 + src_idx*3 + 2, -kb*(z0 - z1)/dij);
                atomicAdd(d2E_dxdp + b0_gather_idx*N*3 + dst_idx*3 + 0, -kb*(-x0 + x1)/dij);
                atomicAdd(d2E_dxdp + b0_gather_idx*N*3 + dst_idx*3 + 1, -kb*(-y0 + y1)/dij);
                atomicAdd(d2E_dxdp + b0_gather_idx*N*3 + dst_idx*3 + 2, -kb*(-z0 + z1)/dij);
            }

            int kb_gather_idx = param_gather_idxs[param_idxs[bond_idx*2+1]];

            if(kb_gather_idx >= 0) {
                atomicAdd(d2E_dxdp + kb_gather_idx*N*3 + src_idx*3 + 0, db*(x0 - x1)/dij);
                atomicAdd(d2E_dxdp + kb_gather_idx*N*3 + src_idx*3 + 1, db*(y0 - y1)/dij);
                atomicAdd(d2E_dxdp + kb_gather_idx*N*3 + src_idx*3 + 2, db*(z0 - z1)/dij);
                atomicAdd(d2E_dxdp + kb_gather_idx*N*3 + dst_idx*3 + 0, db*(-x0 + x1)/dij);
                atomicAdd(d2E_dxdp + kb_gather_idx*N*3 + dst_idx*3 + 1, db*(-y0 + y1)/dij);
                atomicAdd(d2E_dxdp + kb_gather_idx*N*3 + dst_idx*3 + 2, db*(-z0 + z1)/dij);
            }

        }


    }

    // if(b_idx >= num_bonds) {
    //     return;
    // }

    // int src_idx = bond_idxs[b_idx*2 + 0];
    // int dst_idx = bond_idxs[b_idx*2 + 1];

    // RealType rx0 = coords[conf_idx*num_atoms*3+src_idx*3+0];
    // RealType ry0 = coords[conf_idx*num_atoms*3+src_idx*3+1];
    // RealType rz0 = coords[conf_idx*num_atoms*3+src_idx*3+2];

    // RealType rx1 = coords[conf_idx*num_atoms*3+dst_idx*3+0];
    // RealType ry1 = coords[conf_idx*num_atoms*3+dst_idx*3+1];
    // RealType rz1 = coords[conf_idx*num_atoms*3+dst_idx*3+2];

    // RealType ix0 = 0;
    // RealType iy0 = 0;
    // RealType iz0 = 0;

    // RealType ix1 = 0;
    // RealType iy1 = 0;
    // RealType iz1 = 0;

    // RealType rkb = params[param_idxs[b_idx*2+0]];
    // RealType rb0 = params[param_idxs[b_idx*2+1]];

    // RealType ikb = 0;
    // RealType ib0 = 0;

    // const auto dp_idx = blockIdx.y;
    // const auto num_dp = gridDim.y;

    // // (ytz): this a complex step size, not a standard finite difference step size.
    // // the error decays quadratically as opposed to linearly w.r.t. step size.
    // const RealType step_size = 1e-7;

    // if(compute_dp) {

    //     // complex-step coordinates
    //     if(dx_dp != nullptr) {
    //         ix0 = dx_dp[conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + src_idx*3 + 0]*step_size;
    //         iy0 = dx_dp[conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + src_idx*3 + 1]*step_size;
    //         iz0 = dx_dp[conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + src_idx*3 + 2]*step_size;

    //         ix1 = dx_dp[conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + dst_idx*3 + 0]*step_size;
    //         iy1 = dx_dp[conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + dst_idx*3 + 1]*step_size;
    //         iz1 = dx_dp[conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + dst_idx*3 + 2]*step_size;
    //     }

    //     // complex-step parameters 
    //     if(param_idxs[b_idx*2+0] == dp_idxs[dp_idx]){
    //         ikb = step_size;
    //     }
    //     if(param_idxs[b_idx*2+1] == dp_idxs[dp_idx]){
    //         ib0 = step_size;
    //     }   
    // }

    // Surreal<RealType> x0(rx0, ix0);
    // Surreal<RealType> y0(ry0, iy0);
    // Surreal<RealType> z0(rz0, iz0);

    // Surreal<RealType> x1(rx1, ix1);
    // Surreal<RealType> y1(ry1, iy1);
    // Surreal<RealType> z1(rz1, iz1);

    // Surreal<RealType> dx = x0 - x1;
    // Surreal<RealType> dy = y0 - y1;
    // Surreal<RealType> dz = z0 - z1;

    // Surreal<RealType> kb(rkb, ikb);
    // Surreal<RealType> b0(rb0, ib0);

    // Surreal<RealType> d2x = dx*dx;
    // Surreal<RealType> d2y = dy*dy;
    // Surreal<RealType> d2z = dz*dz;

    // Surreal<RealType> d2ij = d2x + d2y + d2z;
    // Surreal<RealType> dij = sqrt(d2ij);
    // Surreal<RealType> db = dij - b0;

    // Surreal<RealType> src_grad_dx = kb*db*dx/dij;
    // Surreal<RealType> src_grad_dy = kb*db*dy/dij;
    // Surreal<RealType> src_grad_dz = kb*db*dz/dij;

    // Surreal<RealType> dst_grad_dx = -src_grad_dx;
    // Surreal<RealType> dst_grad_dy = -src_grad_dy;
    // Surreal<RealType> dst_grad_dz = -src_grad_dz;

    // Surreal<RealType> energy = 0.5*kb*db*db;

    // // E is never null
    // if(blockIdx.y == 0) {
    //     atomicAdd(E + conf_idx, energy.real);
    // }

    // if(blockIdx.y == 0 && dE_dx != nullptr) {
    //     atomicAdd(dE_dx + conf_idx*num_atoms*3 + src_idx*3 + 0, src_grad_dx.real);
    //     atomicAdd(dE_dx + conf_idx*num_atoms*3 + src_idx*3 + 1, src_grad_dy.real);
    //     atomicAdd(dE_dx + conf_idx*num_atoms*3 + src_idx*3 + 2, src_grad_dz.real);

    //     atomicAdd(dE_dx + conf_idx*num_atoms*3 + dst_idx*3 + 0, dst_grad_dx.real);
    //     atomicAdd(dE_dx + conf_idx*num_atoms*3 + dst_idx*3 + 1, dst_grad_dy.real);
    //     atomicAdd(dE_dx + conf_idx*num_atoms*3 + dst_idx*3 + 2, dst_grad_dz.real);
    // }

    // if(compute_dp && dE_dp != nullptr) {
    //     atomicAdd(dE_dp + conf_idx * num_dp + dp_idx, energy.imag/step_size);
    // }

    // if(compute_dp && d2E_dxdp != nullptr) {
    //     atomicAdd(d2E_dxdp + conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + src_idx*3 + 0, src_grad_dx.imag/step_size);
    //     atomicAdd(d2E_dxdp + conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + src_idx*3 + 1, src_grad_dy.imag/step_size);
    //     atomicAdd(d2E_dxdp + conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + src_idx*3 + 2, src_grad_dz.imag/step_size);

    //     atomicAdd(d2E_dxdp + conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + dst_idx*3 + 0, dst_grad_dx.imag/step_size);
    //     atomicAdd(d2E_dxdp + conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + dst_idx*3 + 1, dst_grad_dy.imag/step_size);
    //     atomicAdd(d2E_dxdp + conf_idx*num_dp*num_atoms*3 + dp_idx*num_atoms*3 + dst_idx*3 + 2, dst_grad_dz.imag/step_size);
    // }

}
