#pragma once

#include "surreal.cuh"
#include "kernel_utils.cuh"

template<typename RealType>
void __global__ k_harmonic_bond_derivatives(
    const int num_atoms,     // n, number of atoms
    const RealType *coords,  // [C, n, 3]
    const RealType *params,  // [p,]
    const int num_bonds,     // b
    const int *bond_idxs,    // [b, 2]
    const int *param_idxs,   // [b, 2]
    RealType *E,             // [,] or null
    RealType *dE_dx,         // [n,3] or null
    RealType *d2E_dx2,       // [C, n, 3, n, 3] or null, hessian
    // parameters used for computing derivatives
    const int num_dp,        // dp, number of parameters we're differentiating w.r.t. 
    const int *param_gather_idxs, // [p,] if -1, then we discard
    RealType *dE_dp,         // [C, dp,] or null
    RealType *d2E_dxdp       // [C, dp, n, 3] or null
) {

    const auto conf_idx = blockIdx.z;
    const int N = num_atoms;
    const int DP = num_dp;
    const auto b_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(b_idx >= num_bonds) {
        return;
    }

    int src_idx = bond_idxs[b_idx*2+0];
    int dst_idx = bond_idxs[b_idx*2+1];

    RealType x0 = coords[conf_idx*num_atoms*3+src_idx*3+0];
    RealType y0 = coords[conf_idx*num_atoms*3+src_idx*3+1];
    RealType z0 = coords[conf_idx*num_atoms*3+src_idx*3+2];
    RealType x1 = coords[conf_idx*num_atoms*3+dst_idx*3+0];
    RealType y1 = coords[conf_idx*num_atoms*3+dst_idx*3+1];
    RealType z1 = coords[conf_idx*num_atoms*3+dst_idx*3+2];

    RealType dx = x0 - x1;
    RealType dy = y0 - y1;
    RealType dz = z0 - z1;

    RealType kb = params[param_idxs[b_idx*2+0]];
    RealType b0 = params[param_idxs[b_idx*2+1]];

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

    // energy
    if(E) {
        atomicAdd(E + conf_idx, 0.5*kb*db*db);            
    }

    // force
    if(dE_dx) {
        atomicAdd(dE_dx + conf_idx*N*3 + src_idx*3 + 0, src_grad_dx);
        atomicAdd(dE_dx + conf_idx*N*3 + src_idx*3 + 1, src_grad_dy);
        atomicAdd(dE_dx + conf_idx*N*3 + src_idx*3 + 2, src_grad_dz);
        atomicAdd(dE_dx + conf_idx*N*3 + dst_idx*3 + 0, dst_grad_dx);
        atomicAdd(dE_dx + conf_idx*N*3 + dst_idx*3 + 1, dst_grad_dy);
        atomicAdd(dE_dx + conf_idx*N*3 + dst_idx*3 + 2, dst_grad_dz);
    }

    // hessian
    if(d2E_dx2) {
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 0*N*3 + src_idx*3 + 0, kb*(db*-dx*dx/d3ij + db/dij + d2x/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 0*N*3 + src_idx*3 + 1, kb*(db*dx*-dy/d3ij + dx*dy/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 0*N*3 + src_idx*3 + 2, kb*(db*dx*-dz/d3ij + dx*dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 0*N*3 + dst_idx*3 + 0, kb*(db*d2x/d3ij - db/dij + -dx*dx/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 0*N*3 + dst_idx*3 + 1, kb*(db*dx*dy/d3ij + dx*-dy/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 0*N*3 + dst_idx*3 + 2, kb*(db*dx*dz/d3ij + dx*-dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 1*N*3 + src_idx*3 + 0, kb*(db*-dx*dy/d3ij + dx*dy/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 1*N*3 + src_idx*3 + 1, kb*(db*-dy*dy/d3ij + db/dij + d2y/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 1*N*3 + src_idx*3 + 2, kb*(db*dy*-dz/d3ij + dy*dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 1*N*3 + dst_idx*3 + 0, kb*(db*dx*dy/d3ij + -dx*dy/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 1*N*3 + dst_idx*3 + 1, kb*(db*d2y/d3ij - db/dij + -dy*dy/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 1*N*3 + dst_idx*3 + 2, kb*(db*dy*dz/d3ij + dy*-dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 2*N*3 + src_idx*3 + 0, kb*(db*-dx*dz/d3ij + dx*dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 2*N*3 + src_idx*3 + 1, kb*(db*-dy*dz/d3ij + dy*dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 2*N*3 + src_idx*3 + 2, kb*(db*-dz*dz/d3ij + db/dij + d2z/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 2*N*3 + dst_idx*3 + 0, kb*(db*dx*dz/d3ij + -dx*dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 2*N*3 + dst_idx*3 + 1, kb*(db*dy*dz/d3ij + -dy*dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + src_idx*3*N*3 + 2*N*3 + dst_idx*3 + 2, kb*(db*d2z/d3ij - db/dij + -dz*dz/d2ij));

        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 0*N*3 + src_idx*3 + 0, kb*(db*d2x/d3ij - db/dij + -dx*dx/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 0*N*3 + src_idx*3 + 1, kb*(db*-dx*-dy/d3ij + -dx*dy/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 0*N*3 + src_idx*3 + 2, kb*(db*-dx*-dz/d3ij + -dx*dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 0*N*3 + dst_idx*3 + 0, kb*(db*-dx*dx/d3ij + db/dij + d2x/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 0*N*3 + dst_idx*3 + 1, kb*(db*-dx*dy/d3ij + -dx*-dy/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 0*N*3 + dst_idx*3 + 2, kb*(db*-dx*dz/d3ij + -dx*-dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 1*N*3 + src_idx*3 + 0, kb*(db*-dx*-dy/d3ij + dx*-dy/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 1*N*3 + src_idx*3 + 1, kb*(db*d2y/d3ij - db/dij + -dy*dy/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 1*N*3 + src_idx*3 + 2, kb*(db*-dy*-dz/d3ij + -dy*dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 1*N*3 + dst_idx*3 + 0, kb*(db*dx*-dy/d3ij + -dx*-dy/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 1*N*3 + dst_idx*3 + 1, kb*(db*-dy*dy/d3ij + db/dij + d2y/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 1*N*3 + dst_idx*3 + 2, kb*(db*-dy*dz/d3ij + -dy*-dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 2*N*3 + src_idx*3 + 0, kb*(db*-dx*-dz/d3ij + dx*-dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 2*N*3 + src_idx*3 + 1, kb*(db*-dy*-dz/d3ij + dy*-dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 2*N*3 + src_idx*3 + 2, kb*(db*d2z/d3ij - db/dij + -dz*dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 2*N*3 + dst_idx*3 + 0, kb*(db*dx*-dz/d3ij + -dx*-dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 2*N*3 + dst_idx*3 + 1, kb*(db*dy*-dz/d3ij + -dy*-dz/d2ij));
        atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + dst_idx*3*N*3 + 2*N*3 + dst_idx*3 + 2, kb*(db*-dz*dz/d3ij + db/dij + d2z/d2ij));            
    }

    int kb_gather_idx = param_gather_idxs[param_idxs[b_idx*2+0]];
    int b0_gather_idx = param_gather_idxs[param_idxs[b_idx*2+1]];

    // parameter derivatives
    if(dE_dp) {
        if(kb_gather_idx >= 0) {
            atomicAdd(dE_dp + conf_idx*DP + kb_gather_idx, 0.5*db*db);
        }
        if(b0_gather_idx >= 0) {
            atomicAdd(dE_dp + conf_idx*DP + b0_gather_idx, -kb*db);
        }
    }

    // mixed partial
    if(d2E_dxdp) {
        if(b0_gather_idx >= 0) {
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + b0_gather_idx*N*3 + src_idx*3 + 0, -kb*(x0 - x1)/dij);
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + b0_gather_idx*N*3 + src_idx*3 + 1, -kb*(y0 - y1)/dij);
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + b0_gather_idx*N*3 + src_idx*3 + 2, -kb*(z0 - z1)/dij);
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + b0_gather_idx*N*3 + dst_idx*3 + 0, -kb*(-x0 + x1)/dij);
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + b0_gather_idx*N*3 + dst_idx*3 + 1, -kb*(-y0 + y1)/dij);
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + b0_gather_idx*N*3 + dst_idx*3 + 2, -kb*(-z0 + z1)/dij);
        }
        if(kb_gather_idx >= 0) {
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + kb_gather_idx*N*3 + src_idx*3 + 0, db*(x0 - x1)/dij);
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + kb_gather_idx*N*3 + src_idx*3 + 1, db*(y0 - y1)/dij);
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + kb_gather_idx*N*3 + src_idx*3 + 2, db*(z0 - z1)/dij);
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + kb_gather_idx*N*3 + dst_idx*3 + 0, db*(-x0 + x1)/dij);
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + kb_gather_idx*N*3 + dst_idx*3 + 1, db*(-y0 + y1)/dij);
            atomicAdd(d2E_dxdp + conf_idx*DP*N*3 + kb_gather_idx*N*3 + dst_idx*3 + 2, db*(-z0 + z1)/dij);
        }
    }
}
