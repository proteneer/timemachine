#pragma once

#include "kernel_utils.cuh"
/*

This kernel is very similar to electrostatics.cuh, please read from that for additional info.

*/

template<typename NumericType>
__global__ void lennard_jones_total_derivative(
    const NumericType *coords,
    const NumericType *params, // change to int later?
    const int *global_param_idxs, // change to int later?
    const int *param_idxs,
    const NumericType *scale_matrix,
    NumericType *energy_out,
    NumericType *grad_out,
    NumericType *hessian_out,
    NumericType *mp_out,
    int N) {

    const int n_atoms = N;

    auto i_idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    NumericType *mp_out_sig0;
    NumericType *mp_out_eps0;

    NumericType x0, y0, z0, sig0, eps0;

    if(i_idx >= n_atoms) {
        x0 = 0.0;
        y0 = 0.0;
        z0 = 0.0;
        sig0 = 0.0;
        eps0 = 0.0;
    } else {
        x0 = coords[i_idx*3+0];
        y0 = coords[i_idx*3+1];
        z0 = coords[i_idx*3+2];
        sig0 = params[param_idxs[i_idx*2+0]];
        eps0 = params[param_idxs[i_idx*2+1]];

        mp_out_sig0 = mp_out + global_param_idxs[param_idxs[i_idx*2+0]]*n_atoms*3;
        mp_out_eps0 = mp_out + global_param_idxs[param_idxs[i_idx*2+1]]*n_atoms*3;

    }

    NumericType grad_dx = 0;
    NumericType grad_dy = 0;
    NumericType grad_dz = 0;

    NumericType hess_xx = 0;
    NumericType hess_yx = 0;
    NumericType hess_yy = 0;
    NumericType hess_zx = 0;
    NumericType hess_zy = 0;
    NumericType hess_zz = 0;

    int num_y_tiles = blockIdx.x + 1;

    for(int tile_y_idx = 0; tile_y_idx < num_y_tiles; tile_y_idx++) {

        NumericType x1, y1, z1, sig1, eps1;
        NumericType shfl_grad_dx = 0;
        NumericType shfl_grad_dy = 0;
        NumericType shfl_grad_dz = 0;

        NumericType shfl_hess_xx = 0;
        NumericType shfl_hess_yx = 0;
        NumericType shfl_hess_yy = 0;
        NumericType shfl_hess_zx = 0;
        NumericType shfl_hess_zy = 0;
        NumericType shfl_hess_zz = 0;

        // load diagonal elements exactly once, shuffle the rest
        int j_idx = tile_y_idx*WARP_SIZE + threadIdx.x;

        if(j_idx >= n_atoms) {
            x1 = 0.0;
            y1 = 0.0;
            z1 = 0.0;
            sig1 = 0.0;
            eps1 = 0.0;
        } else {
            x1 = coords[j_idx*3+0];
            y1 = coords[j_idx*3+1];
            z1 = coords[j_idx*3+2];
            sig1 = params[param_idxs[j_idx*2+0]];
            eps1 = params[param_idxs[j_idx*2+1]];
        }

        // off diagonal
        // iterate over a block of i's because we improve locality of writes to off diagonal elements
        #pragma unroll 4
        for(int round=0; round < WARP_SIZE; round++) {
            NumericType xi = __shfl_sync(0xffffffff, x0, round);
            NumericType yi = __shfl_sync(0xffffffff, y0, round);
            NumericType zi = __shfl_sync(0xffffffff, z0, round);
            NumericType sigi = __shfl_sync(0xffffffff, sig0, round);
            NumericType epsi = __shfl_sync(0xffffffff, eps0, round);

            int h_i_idx = blockIdx.x*WARP_SIZE + round;
            int h_j_idx = j_idx;


            if(h_j_idx < h_i_idx && h_i_idx < n_atoms && h_j_idx < n_atoms) {

                NumericType dx = xi - x1;
                NumericType dy = yi - y1;
                NumericType dz = zi - z1;
                NumericType d2x = dx*dx;
                NumericType d2y = dy*dy;
                NumericType d2z = dz*dz;

                NumericType sij = scale_matrix[h_i_idx*n_atoms + h_j_idx];

                NumericType eps = sqrt(epsi * eps1);
                NumericType sig = (sigi + sig1)/2;

                NumericType d2ij = d2x + d2y + d2z;
                NumericType dij = gpuSqrt(d2ij);
                NumericType d4ij = d2ij*d2ij;
                NumericType d6ij = d4ij*d2ij;
                NumericType d8ij = d4ij*d4ij;
                NumericType d16ij = d8ij*d8ij;
                NumericType inv_d16ij = 1.0/d16ij;

                NumericType sig2 = sig*sig;
                NumericType sig3 = sig2*sig;
                NumericType sig6 = sig3*sig3;
                NumericType prefactor = sij*eps*sig6;

                hessian_out[HESS_IDX(h_i_idx, h_j_idx, N, 0, 0)] += prefactor*-24*(d8ij - 8*d6ij*pow(dx, 2) - 2*d2ij*sig6 + 28*pow(dx, 2)*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, N, 0, 1)] += prefactor*96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, N, 0, 2)] += prefactor*96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;

                hessian_out[HESS_IDX(h_i_idx, h_j_idx, N, 1, 0)] += prefactor*96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, N, 1, 1)] += prefactor*-24*(d8ij - 8*d6ij*pow(dy, 2) - 2*d2ij*sig6 + 28*pow(dy, 2)*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, N, 1, 2)] += prefactor*96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;

                hessian_out[HESS_IDX(h_i_idx, h_j_idx, N, 2, 0)] += prefactor*96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, N, 2, 1)] += prefactor*96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_i_idx, h_j_idx, N, 2, 2)] += prefactor*-24*(d8ij - 8*d6ij*pow(dz, 2) - 2*d2ij*sig6 + 28*pow(dz, 2)*sig6)*inv_d16ij;

                hessian_out[HESS_IDX(h_j_idx, h_i_idx, N, 0, 0)] += prefactor*-24*(d8ij - 8*d6ij*pow(dx, 2) - 2*d2ij*sig6 + 28*pow(dx, 2)*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_j_idx, h_i_idx, N, 0, 1)] += prefactor*96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_j_idx, h_i_idx, N, 0, 2)] += prefactor*96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;

                hessian_out[HESS_IDX(h_j_idx, h_i_idx, N, 1, 0)] += prefactor*96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_j_idx, h_i_idx, N, 1, 1)] += prefactor*-24*(d8ij - 8*d6ij*pow(dy, 2) - 2*d2ij*sig6 + 28*pow(dy, 2)*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_j_idx, h_i_idx, N, 1, 2)] += prefactor*96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;

                hessian_out[HESS_IDX(h_j_idx, h_i_idx, N, 2, 0)] += prefactor*96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_j_idx, h_i_idx, N, 2, 1)] += prefactor*96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hessian_out[HESS_IDX(h_j_idx, h_i_idx, N, 2, 2)] += prefactor*-24*(d8ij - 8*d6ij*pow(dz, 2) - 2*d2ij*sig6 + 28*pow(dz, 2)*sig6)*inv_d16ij;

            }

        }

        // diagonal elements and mixed partials
        for(int round=0; round < WARP_SIZE; round++) {

            j_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

            if(j_idx < i_idx && i_idx < n_atoms && j_idx < n_atoms) {

                NumericType dx = x0 - x1;
                NumericType dy = y0 - y1;
                NumericType dz = z0 - z1;
                NumericType d2x = dx*dx;
                NumericType d2y = dy*dy;
                NumericType d2z = dz*dz;

                NumericType sij = scale_matrix[i_idx*n_atoms + j_idx];
                NumericType d2ij = d2x + d2y + d2z;
                NumericType dij = gpuSqrt(d2ij);
                NumericType d4ij = d2ij*d2ij;
                NumericType d6ij = d4ij*d2ij;
                NumericType d8ij = d4ij*d4ij;
                NumericType d16ij = d8ij*d8ij;
                NumericType inv_d16ij = 1.0/d16ij;

                NumericType eps = gpuSqrt(eps0*eps1);
                NumericType sig = (sig0 + sig1)/2;

                NumericType sig2 = sig*sig;
                NumericType sig3 = sig2*sig;
                NumericType sig5 = sig3*sig2;
                NumericType sig6 = sig3*sig3;
                NumericType sig11 = sig6*sig3*sig2;
                NumericType sig12 = sig6*sig6;

                // rij = d2ij, avoids a square root.
                NumericType rij = dx*dx + dy*dy + dz*dz;
                // printf("i_idx %d j_idx %d eps %f sig %f\n", i_idx, j_idx, eps, sig);
                NumericType rij3 = rij * rij * rij;
                NumericType rij4 = rij3 * rij;
                NumericType rij7 = rij4 * rij3;

                // (ytz): 99 % sure this loses precision so we need to refactor
                NumericType sig1rij1 = sig/rij;
                NumericType sig3rij3 = sig1rij1*sig1rij1*sig1rij1;
                NumericType sig6rij3 = sig3*sig3rij3;
                NumericType sig4rij4 = sig3rij3*sig1rij1;
                NumericType sig5rij4 = sig*sig4rij4;
                NumericType sig6rij4 = sig*sig5rij4;


                NumericType sig12rij7 = sig12/rij7;
                NumericType sig11rij7 = sig11/rij7;

                // NumericType sig6rij4 = sig6/rij4;
                // NumericType sig5rij4 = sig5/rij4;

                NumericType dEdx = 24*eps*dx*(sig12rij7*2 - sig6rij4);
                NumericType dEdy = 24*eps*dy*(sig12rij7*2 - sig6rij4);
                NumericType dEdz = 24*eps*dz*(sig12rij7*2 - sig6rij4);

                // TODO: multiply by sij when atomic adding (But this is trivial relative to cost
                // of hessians.
                grad_dx -= sij*dEdx;
                grad_dy -= sij*dEdy;
                grad_dz -= sij*dEdz;

                shfl_grad_dx += sij*dEdx;
                shfl_grad_dy += sij*dEdy;
                shfl_grad_dz += sij*dEdz;

                NumericType *mp_out_sig1 = mp_out + global_param_idxs[param_idxs[j_idx*2+0]]*n_atoms*3;
                NumericType *mp_out_eps1 = mp_out + global_param_idxs[param_idxs[j_idx*2+1]]*n_atoms*3;

                printf("sig6/rij4 %f sig12 %f, rij4 %f\n", sig6rij4, sig6, rij4);

                atomicAdd(mp_out_eps0 + i_idx*3 + 0, -sij*12*(eps1/eps)*dx*(sig6rij4)*(2*sig6rij3 - 1));
                atomicAdd(mp_out_eps0 + i_idx*3 + 1, -sij*12*(eps1/eps)*dy*(sig6rij4)*(2*sig6rij3 - 1));
                atomicAdd(mp_out_eps0 + i_idx*3 + 2, -sij*12*(eps1/eps)*dz*(sig6rij4)*(2*sig6rij3 - 1));

                atomicAdd(mp_out_eps1 + i_idx*3 + 0, -sij*12*(eps0/eps)*dx*(sig6rij4)*(2*sig6rij3 - 1));
                atomicAdd(mp_out_eps1 + i_idx*3 + 1, -sij*12*(eps0/eps)*dy*(sig6rij4)*(2*sig6rij3 - 1));
                atomicAdd(mp_out_eps1 + i_idx*3 + 2, -sij*12*(eps0/eps)*dz*(sig6rij4)*(2*sig6rij3 - 1));

                atomicAdd(mp_out_eps0 + j_idx*3 + 0,  sij*12*(eps1/eps)*dx*(sig6rij4)*(2*sig6rij3 - 1));
                atomicAdd(mp_out_eps0 + j_idx*3 + 1,  sij*12*(eps1/eps)*dy*(sig6rij4)*(2*sig6rij3 - 1));
                atomicAdd(mp_out_eps0 + j_idx*3 + 2,  sij*12*(eps1/eps)*dz*(sig6rij4)*(2*sig6rij3 - 1));

                atomicAdd(mp_out_eps1 + j_idx*3 + 0,  sij*12*(eps0/eps)*dx*(sig6rij4)*(2*sig6rij3 - 1));
                atomicAdd(mp_out_eps1 + j_idx*3 + 1,  sij*12*(eps0/eps)*dy*(sig6rij4)*(2*sig6rij3 - 1));
                atomicAdd(mp_out_eps1 + j_idx*3 + 2,  sij*12*(eps0/eps)*dz*(sig6rij4)*(2*sig6rij3 - 1));

                atomicAdd(mp_out_sig0 + i_idx*3 + 0, -sij*24*eps*dx*(sig5/rij4)*(12*sig6rij3 - 3));
                atomicAdd(mp_out_sig0 + i_idx*3 + 1, -sij*24*eps*dy*(sig5/rij4)*(12*sig6rij3 - 3));
                atomicAdd(mp_out_sig0 + i_idx*3 + 2, -sij*24*eps*dz*(sig5/rij4)*(12*sig6rij3 - 3));

                atomicAdd(mp_out_sig1 + i_idx*3 + 0, -sij*24*eps*dx*(sig5/rij4)*(12*sig6rij3 - 3));
                atomicAdd(mp_out_sig1 + i_idx*3 + 1, -sij*24*eps*dy*(sig5/rij4)*(12*sig6rij3 - 3));
                atomicAdd(mp_out_sig1 + i_idx*3 + 2, -sij*24*eps*dz*(sig5/rij4)*(12*sig6rij3 - 3));

                atomicAdd(mp_out_sig0 + j_idx*3 + 0,  sij*24*eps*dx*(sig5/rij4)*(12*sig6rij3 - 3));
                atomicAdd(mp_out_sig0 + j_idx*3 + 1,  sij*24*eps*dy*(sig5/rij4)*(12*sig6rij3 - 3));
                atomicAdd(mp_out_sig0 + j_idx*3 + 2,  sij*24*eps*dz*(sig5/rij4)*(12*sig6rij3 - 3));

                atomicAdd(mp_out_sig1 + j_idx*3 + 0,  sij*24*eps*dx*(sig5/rij4)*(12*sig6rij3 - 3));
                atomicAdd(mp_out_sig1 + j_idx*3 + 1,  sij*24*eps*dy*(sig5/rij4)*(12*sig6rij3 - 3));
                atomicAdd(mp_out_sig1 + j_idx*3 + 2,  sij*24*eps*dz*(sig5/rij4)*(12*sig6rij3 - 3));




                // atomicAdd(mp_out_eps0 + i_idx*3 + 0, -sij*12*(eps1/eps)*dx*(sig12rij7*2 - sig6rij4));
                // atomicAdd(mp_out_eps0 + i_idx*3 + 1, -sij*12*(eps1/eps)*dy*(sig12rij7*2 - sig6rij4));
                // atomicAdd(mp_out_eps0 + i_idx*3 + 2, -sij*12*(eps1/eps)*dz*(sig12rij7*2 - sig6rij4));

                // atomicAdd(mp_out_eps1 + i_idx*3 + 0, -sij*12*(eps0/eps)*dx*(sig12rij7*2 - sig6rij4));
                // atomicAdd(mp_out_eps1 + i_idx*3 + 1, -sij*12*(eps0/eps)*dy*(sig12rij7*2 - sig6rij4));
                // atomicAdd(mp_out_eps1 + i_idx*3 + 2, -sij*12*(eps0/eps)*dz*(sig12rij7*2 - sig6rij4));

                // atomicAdd(mp_out_eps0 + j_idx*3 + 0,  sij*12*(eps1/eps)*dx*(sig12rij7*2 - sig6rij4));
                // atomicAdd(mp_out_eps0 + j_idx*3 + 1,  sij*12*(eps1/eps)*dy*(sig12rij7*2 - sig6rij4));
                // atomicAdd(mp_out_eps0 + j_idx*3 + 2,  sij*12*(eps1/eps)*dz*(sig12rij7*2 - sig6rij4));

                // atomicAdd(mp_out_eps1 + j_idx*3 + 0,  sij*12*(eps0/eps)*dx*(sig12rij7*2 - sig6rij4));
                // atomicAdd(mp_out_eps1 + j_idx*3 + 1,  sij*12*(eps0/eps)*dy*(sig12rij7*2 - sig6rij4));
                // atomicAdd(mp_out_eps1 + j_idx*3 + 2,  sij*12*(eps0/eps)*dz*(sig12rij7*2 - sig6rij4));

                // atomicAdd(mp_out_sig0 + i_idx*3 + 0, -sij*24*eps*dx*(12*sig11rij7 - 3*sig5rij4));
                // atomicAdd(mp_out_sig0 + i_idx*3 + 1, -sij*24*eps*dy*(12*sig11rij7 - 3*sig5rij4));
                // atomicAdd(mp_out_sig0 + i_idx*3 + 2, -sij*24*eps*dz*(12*sig11rij7 - 3*sig5rij4));

                // atomicAdd(mp_out_sig1 + i_idx*3 + 0, -sij*24*eps*dx*(12*sig11rij7 - 3*sig5rij4));
                // atomicAdd(mp_out_sig1 + i_idx*3 + 1, -sij*24*eps*dy*(12*sig11rij7 - 3*sig5rij4));
                // atomicAdd(mp_out_sig1 + i_idx*3 + 2, -sij*24*eps*dz*(12*sig11rij7 - 3*sig5rij4));

                // atomicAdd(mp_out_sig0 + j_idx*3 + 0,  sij*24*eps*dx*(12*sig11rij7 - 3*sig5rij4));
                // atomicAdd(mp_out_sig0 + j_idx*3 + 1,  sij*24*eps*dy*(12*sig11rij7 - 3*sig5rij4));
                // atomicAdd(mp_out_sig0 + j_idx*3 + 2,  sij*24*eps*dz*(12*sig11rij7 - 3*sig5rij4));

                // atomicAdd(mp_out_sig1 + j_idx*3 + 0,  sij*24*eps*dx*(12*sig11rij7 - 3*sig5rij4));
                // atomicAdd(mp_out_sig1 + j_idx*3 + 1,  sij*24*eps*dy*(12*sig11rij7 - 3*sig5rij4));
                // atomicAdd(mp_out_sig1 + j_idx*3 + 2,  sij*24*eps*dz*(12*sig11rij7 - 3*sig5rij4));

                NumericType prefactor = sij*eps*sig6;

                hess_xx += prefactor*24*(d8ij - 8*d6ij*pow(dx, 2) - 2*d2ij*sig6 + 28*pow(dx, 2)*sig6)*inv_d16ij;
                hess_yx += prefactor*-96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                hess_yy += prefactor*24*(d8ij - 8*d6ij*pow(dy, 2) - 2*d2ij*sig6 + 28*pow(dy, 2)*sig6)*inv_d16ij;                
                hess_zx += prefactor*-96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hess_zy += prefactor*-96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                hess_zz += prefactor*24*(d8ij - 8*d6ij*pow(dz, 2) - 2*d2ij*sig6 + 28*pow(dz, 2)*sig6)*inv_d16ij;
                
                shfl_hess_xx += prefactor*24*(d8ij - 8*d6ij*pow(dx, 2) - 2*d2ij*sig6 + 28*pow(dx, 2)*sig6)*inv_d16ij;
                shfl_hess_yx += prefactor*-96*dx*dy*(2*d6ij - 7*sig6)*inv_d16ij;
                shfl_hess_yy += prefactor*24*(d8ij - 8*d6ij*pow(dy, 2) - 2*d2ij*sig6 + 28*pow(dy, 2)*sig6)*inv_d16ij;
                shfl_hess_zx += prefactor*-96*dx*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                shfl_hess_zy += prefactor*-96*dy*dz*(2*d6ij - 7*sig6)*inv_d16ij;
                shfl_hess_zz += prefactor*24*(d8ij - 8*d6ij*pow(dz, 2) - 2*d2ij*sig6 + 28*pow(dz, 2)*sig6)*inv_d16ij;

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

            shfl_hess_xx = __shfl_sync(0xffffffff, shfl_hess_xx, srcLane);
            shfl_hess_yx = __shfl_sync(0xffffffff, shfl_hess_yx, srcLane);
            shfl_hess_yy = __shfl_sync(0xffffffff, shfl_hess_yy, srcLane);
            shfl_hess_zx = __shfl_sync(0xffffffff, shfl_hess_zx, srcLane);
            shfl_hess_zy = __shfl_sync(0xffffffff, shfl_hess_zy, srcLane);
            shfl_hess_zz = __shfl_sync(0xffffffff, shfl_hess_zz, srcLane);

            j_idx += 1;

        }

        int target_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

        if(target_idx < n_atoms) {
            atomicAdd(grad_out + target_idx*3 + 0, shfl_grad_dx);
            atomicAdd(grad_out + target_idx*3 + 1, shfl_grad_dy);
            atomicAdd(grad_out + target_idx*3 + 2, shfl_grad_dz);

            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 0, 0), shfl_hess_xx);
            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 1, 0), shfl_hess_yx);
            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 1, 1), shfl_hess_yy);
            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 2, 0), shfl_hess_zx);
            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 2, 1), shfl_hess_zy);
            atomicAdd(hessian_out + HESS_IDX(target_idx, target_idx, n_atoms, 2, 2), shfl_hess_zz);
        }

    }

    if(i_idx < n_atoms) {
        atomicAdd(grad_out + i_idx*3 + 0, grad_dx);
        atomicAdd(grad_out + i_idx*3 + 1, grad_dy);
        atomicAdd(grad_out + i_idx*3 + 2, grad_dz);

        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 0, 0), hess_xx);
        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 1, 0), hess_yx);
        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 1, 1), hess_yy);
        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 2, 0), hess_zx);
        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 2, 1), hess_zy);
        atomicAdd(hessian_out + HESS_IDX(i_idx, i_idx, n_atoms, 2, 2), hess_zz);
    }

};
