#pragma once

#include <cstdio>
#include "surreal.cuh"


// adds support for __shfling complex numbers
template<typename RealType>
__device__ inline Surreal<RealType> __shfl_sync(unsigned mask, Surreal<RealType> var, int srcLane, int width=warpSize) {
    var.real = __shfl_sync(mask, var.real, srcLane, width);
    var.imag = __shfl_sync(mask, var.imag, srcLane, width);
    return var;
}


template<typename RealType>
void __global__ k_lennard_jones(
    const int num_atoms,    // n
    const int num_params,   // p
    const RealType *coords, // [n, 3]
    const RealType *params, // [p,]
    const RealType *scale_matrix,
    const int *param_idxs,  // [n, 2] sig eps
    RealType *E,            // [,] never null
    RealType *dE_dx,        // [n,3] or null
    // parameters used for computing derivatives
    const RealType *dx_dp, // [dp, n, 3]
    const int *dp_idxs,     // of shape [dp] or null, if null then we don't compute parameter derivatives
    RealType *dE_dp,        // [dp,] or null
    RealType *d2E_dxdp      // [dp, n, 3] or null
) {
    auto conf_idx = blockIdx.z;
    auto i_idx = blockDim.x*blockIdx.x + threadIdx.x;

    Surreal<RealType> x0(0, 0);
    Surreal<RealType> y0(0, 0);
    Surreal<RealType> z0(0, 0);
    Surreal<RealType> sig0(0, 0);
    Surreal<RealType> eps0(0, 0);

    Surreal<RealType> grad_dx(0, 0);
    Surreal<RealType> grad_dy(0, 0);
    Surreal<RealType> grad_dz(0, 0);

    Surreal<RealType> energy(0, 0);

    const auto dp_idx = blockIdx.y;
    const auto num_dp = gridDim.y;
    const RealType step_size = 1e-7;
    const bool compute_dp = (dp_idxs != nullptr);

    if(i_idx < num_atoms) {
        RealType rx0 = coords[conf_idx*num_atoms*3 + i_idx*3 + 0];
        RealType ry0 = coords[conf_idx*num_atoms*3 + i_idx*3 + 1];
        RealType rz0 = coords[conf_idx*num_atoms*3 + i_idx*3 + 2];
        RealType rsig0 = params[param_idxs[i_idx*2+0]];
        RealType reps0 = params[param_idxs[i_idx*2+1]];

        RealType ix0 = 0.0;
        RealType iy0 = 0.0;
        RealType iz0 = 0.0;
        RealType isig0 = 0.0;
        RealType ieps0 = 0.0;

        if(compute_dp) {
            if(dx_dp != nullptr) {
                ix0 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3+0]*step_size;
                iy0 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3+1]*step_size;
                iz0 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3+2]*step_size;  
            }

            if(param_idxs[i_idx*2+0] == dp_idxs[dp_idx]) {
                isig0 = step_size;
            }

            if(param_idxs[i_idx*2+1] == dp_idxs[dp_idx]) {
                ieps0 = step_size;
            }
        }

        x0 = Surreal<RealType>(rx0, ix0);
        y0 = Surreal<RealType>(ry0, iy0);
        z0 = Surreal<RealType>(rz0, iz0);
        sig0 = Surreal<RealType>(rsig0, isig0);
        eps0 = Surreal<RealType>(reps0, ieps0);
    }

    int num_y_tiles = blockIdx.x + 1;

    // (ytz): compute the upper right triangular contribution only.
    for(int tile_y_idx = 0; tile_y_idx < num_y_tiles; tile_y_idx++) {

        Surreal<RealType> x1(0, 0);
        Surreal<RealType> y1(0, 0);
        Surreal<RealType> z1(0, 0);
        Surreal<RealType> sig1(0, 0);
        Surreal<RealType> eps1(0, 0);

        Surreal<RealType> shfl_grad_dx(0, 0);
        Surreal<RealType> shfl_grad_dy(0, 0);
        Surreal<RealType> shfl_grad_dz(0, 0);

        int j_idx = tile_y_idx*WARP_SIZE + threadIdx.x;

        if(j_idx < num_atoms) {
            RealType rx1 = coords[conf_idx*num_atoms*3 + j_idx*3 + 0];
            RealType ry1 = coords[conf_idx*num_atoms*3 + j_idx*3 + 1];
            RealType rz1 = coords[conf_idx*num_atoms*3 + j_idx*3 + 2];
            RealType rsig1 = params[param_idxs[j_idx*2+0]];
            RealType reps1 = params[param_idxs[j_idx*2+1]];

            RealType ix1 = 0.0;
            RealType iy1 = 0.0;
            RealType iz1 = 0.0;
            RealType isig1 = 0.0;
            RealType ieps1 = 0.0;

            if(compute_dp) {
                if(dx_dp != nullptr) {
                    ix1 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+j_idx*3+0]*step_size;
                    iy1 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+j_idx*3+1]*step_size;
                    iz1 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+j_idx*3+2]*step_size;  
                }

                if(param_idxs[j_idx*2+0] == dp_idxs[dp_idx]) {
                    isig1 = step_size;
                }

                if(param_idxs[j_idx*2+1] == dp_idxs[dp_idx]) {
                    ieps1 = step_size;
                }
            }

            x1 = Surreal<RealType>(rx1, ix1);
            y1 = Surreal<RealType>(ry1, iy1);
            z1 = Surreal<RealType>(rz1, iz1);
            sig1 = Surreal<RealType>(rsig1, isig1);
            eps1 = Surreal<RealType>(reps1, ieps1);
        }

        for(int round=0; round < WARP_SIZE; round++) {

            j_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

            if(j_idx < i_idx && i_idx < num_atoms && j_idx < num_atoms) {

                Surreal<RealType> dx = x0 - x1;
                Surreal<RealType> dy = y0 - y1;
                Surreal<RealType> dz = z0 - z1;
                Surreal<RealType> d2x = dx*dx;
                Surreal<RealType> d2y = dy*dy;
                Surreal<RealType> d2z = dz*dz;

                Surreal<RealType> sij = scale_matrix[i_idx*num_atoms + j_idx];
                Surreal<RealType> d2ij = d2x + d2y + d2z;
                Surreal<RealType> dij = sqrt(d2ij);
                Surreal<RealType> d4ij = d2ij*d2ij;
                Surreal<RealType> d6ij = d4ij*d2ij;
                Surreal<RealType> d8ij = d4ij*d4ij;
                Surreal<RealType> d16ij = d8ij*d8ij;
                Surreal<RealType> inv_d16ij = 1.0/d16ij;

                Surreal<RealType> eps = sqrt(eps0*eps1);
                Surreal<RealType> sig = (sig0 + sig1)/2;

                Surreal<RealType> sig2 = sig*sig;
                Surreal<RealType> sig3 = sig2*sig;
                Surreal<RealType> sig5 = sig3*sig2;
                Surreal<RealType> sig6 = sig3*sig3;
                Surreal<RealType> sig12 = sig6*sig6;

                Surreal<RealType> rij = d2ij;
                Surreal<RealType> rij3 = d6ij;
                Surreal<RealType> rij4 = d8ij;
                Surreal<RealType> rij7 = rij4 * rij3;

                energy += sij*4*eps*(sig6/d6ij-1.0)*sig6/d6ij;

                // loss of precision?
                Surreal<RealType> sig1rij1 = sig/rij;
                Surreal<RealType> sig3rij3 = sig1rij1*sig1rij1*sig1rij1;
                Surreal<RealType> sig6rij3 = sig3*sig3rij3;
                Surreal<RealType> sig4rij4 = sig3rij3*sig1rij1;
                Surreal<RealType> sig5rij4 = sig*sig4rij4;
                Surreal<RealType> sig6rij4 = sig*sig5rij4;

                Surreal<RealType> sig12rij7 = sig12/rij7;

                Surreal<RealType> dEdx = 24*eps*dx*(sig12rij7*2 - sig6rij4);
                Surreal<RealType> dEdy = 24*eps*dy*(sig12rij7*2 - sig6rij4);
                Surreal<RealType> dEdz = 24*eps*dz*(sig12rij7*2 - sig6rij4);

                // TODO: multiply by sij when atomic adding (But this is trivial relative to cost
                // of hessians.
                grad_dx -= sij*dEdx;
                grad_dy -= sij*dEdy;
                grad_dz -= sij*dEdz;

                shfl_grad_dx += sij*dEdx;
                shfl_grad_dy += sij*dEdy;
                shfl_grad_dz += sij*dEdz;
            }

            int srcLane = (threadIdx.x + 1) % WARP_SIZE;

            x1 = __shfl_sync(0xffffffff, x1, srcLane);
            y1 = __shfl_sync(0xffffffff, y1, srcLane);
            z1 = __shfl_sync(0xffffffff, z1, srcLane);
            sig1 = __shfl_sync(0xffffffff, sig1, srcLane);
            eps1 = __shfl_sync(0xffffffff, eps1, srcLane);

            shfl_grad_dx = __shfl_sync(0xffffffff, shfl_grad_dx, srcLane);
            shfl_grad_dy = __shfl_sync(0xffffffff, shfl_grad_dy, srcLane);
            shfl_grad_dz = __shfl_sync(0xffffffff, shfl_grad_dz, srcLane);

            j_idx += 1;

        }

        int target_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

        if(target_idx < num_atoms) {

            if(blockIdx.y == 0 && dE_dx != nullptr) {
                atomicAdd(dE_dx + conf_idx*num_atoms*3 + target_idx*3 + 0, shfl_grad_dx.real);
                atomicAdd(dE_dx + conf_idx*num_atoms*3 + target_idx*3 + 1, shfl_grad_dy.real);
                atomicAdd(dE_dx + conf_idx*num_atoms*3 + target_idx*3 + 2, shfl_grad_dz.real);
            }

            if(compute_dp && d2E_dxdp != nullptr) {
                atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+target_idx*3 + 0, shfl_grad_dx.imag/step_size);
                atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+target_idx*3 + 1, shfl_grad_dy.imag/step_size);
                atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+target_idx*3 + 2, shfl_grad_dz.imag/step_size);
            }
        }
    }


    if(blockIdx.y == 0) {
        atomicAdd(E + conf_idx, energy.real);
    }

    if(compute_dp && dE_dp != nullptr) {
        atomicAdd(dE_dp+conf_idx*num_dp+dp_idx, energy.imag/step_size);
    }

    if(i_idx < num_atoms) {
        if(blockIdx.y == 0 && dE_dx != nullptr) {
            atomicAdd(dE_dx + conf_idx*num_atoms*3 + i_idx*3 + 0, grad_dx.real);
            atomicAdd(dE_dx + conf_idx*num_atoms*3 + i_idx*3 + 1, grad_dy.real);
            atomicAdd(dE_dx + conf_idx*num_atoms*3 + i_idx*3 + 2, grad_dz.real);
        }

        if(compute_dp && d2E_dxdp != nullptr) {
            atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3 + 0, grad_dx.imag/step_size);
            atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3 + 1, grad_dy.imag/step_size);
            atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3 + 2, grad_dz.imag/step_size);
        }
    }

}