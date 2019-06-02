#pragma once

#include "kernel_utils.cuh"

template<typename RealType>
void __global__ k_electrostatics(
    const int num_atoms,    // n
    const RealType *coords, // [n, 3]
    const RealType *params, // [p,]
    const RealType *scale_matrix, // [n, n]
    const int *param_idxs,  // [n, 1] charge
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

    auto i_idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    RealType x0, y0, z0, q0;
    int q0_g_idx;

    if(i_idx >= N) {
        x0 = 0.0;
        y0 = 0.0;
        z0 = 0.0;
        q0 = 0.0;
        q0_g_idx = 0;
    } else {
        x0 = coords[conf_idx*N*3+i_idx*3+0];
        y0 = coords[conf_idx*N*3+i_idx*3+1];
        z0 = coords[conf_idx*N*3+i_idx*3+2];
        q0 = params[param_idxs[i_idx]];
        q0_g_idx = param_gather_idxs[param_idxs[i_idx]];

    }

    RealType grad_dx = 0;
    RealType grad_dy = 0;
    RealType grad_dz = 0;

    RealType dE_dp_q = 0;

    RealType mixed_dx = 0;
    RealType mixed_dy = 0;
    RealType mixed_dz = 0;

    RealType hess_xx = 0;
    RealType hess_yx = 0;
    RealType hess_yy = 0;
    RealType hess_zx = 0;
    RealType hess_zy = 0;
    RealType hess_zz = 0;

    RealType energy = 0;

    int num_y_tiles = blockIdx.x + 1;

    for(int tile_y_idx = 0; tile_y_idx < num_y_tiles; tile_y_idx++) {

        RealType x1, y1, z1, q1;
        int q1_g_idx;

        RealType shfl_grad_dx = 0;
        RealType shfl_grad_dy = 0;
        RealType shfl_grad_dz = 0;

        RealType shfl_dE_dp_q = 0;

        RealType shfl_mixed_dx = 0;
        RealType shfl_mixed_dy = 0;
        RealType shfl_mixed_dz = 0;

        RealType shfl_hess_xx = 0;
        RealType shfl_hess_yx = 0;
        RealType shfl_hess_yy = 0;
        RealType shfl_hess_zx = 0;
        RealType shfl_hess_zy = 0;
        RealType shfl_hess_zz = 0;

        // load diagonal elements exactly once, shuffle the rest
        int j_idx = tile_y_idx*WARP_SIZE + threadIdx.x;

        if(j_idx >= N) {
            x1 = 0.0;
            y1 = 0.0;
            z1 = 0.0;
            q1 = 0.0;
            q1_g_idx = 0;
        } else {
            x1 = coords[conf_idx*N*3+j_idx*3+0];
            y1 = coords[conf_idx*N*3+j_idx*3+1];
            z1 = coords[conf_idx*N*3+j_idx*3+2];
            q1 = params[param_idxs[j_idx]];
            q1_g_idx = param_gather_idxs[param_idxs[j_idx]];
        }

        // off diagonal
        // iterate over a block of i's because we improve locality of writes to off diagonal elements
        // add a conditional for inference mode. (if d2E_dx2 and d2E_dxdp)
        for(int round=0; round < WARP_SIZE; round++) {
            RealType xi = __shfl_sync(0xffffffff, x0, round);
            RealType yi = __shfl_sync(0xffffffff, y0, round);
            RealType zi = __shfl_sync(0xffffffff, z0, round);
            RealType qi = __shfl_sync(0xffffffff, q0, round);
            int qi_g_idx = __shfl_sync(0xffffffff, q0_g_idx, round);

            int h_i_idx = blockIdx.x*WARP_SIZE + round;
            int h_j_idx = j_idx;

            if(h_j_idx < h_i_idx && h_i_idx < N && h_j_idx < N) {

                RealType dx = xi - x1;
                RealType dy = yi - y1;
                RealType dz = zi - z1;
                RealType d2x = dx*dx;
                RealType d2y = dy*dy;
                RealType d2z = dz*dz;

                RealType d2ij = d2x + d2y + d2z;
                RealType dij = sqrt(d2ij);
                RealType d3ij = d2ij*dij;
                RealType inv_d3ij = 1/d3ij;
                RealType d5ij = d3ij*d2ij;

                RealType sij = scale_matrix[h_i_idx*N + h_j_idx];
                RealType so4eq01 = sij*ONE_4PI_EPS0*qi*q1;
                RealType hess_prefactor = so4eq01/d5ij;

                if(d2E_dx2) {
                    // don't need atomic adds because these are unique diagonals
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 0, 0)] += hess_prefactor*(d2ij - 3*d2x);
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 0, 1)] += -3*hess_prefactor*dx*dy;
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 0, 2)] += -3*hess_prefactor*dx*dz;

                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 1, 0)] += -3*hess_prefactor*dx*dy;
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 1, 1)] += hess_prefactor*(d2ij - 3*d2y);
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 1, 2)] += -3*hess_prefactor*dy*dz;

                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 2, 0)] += -3*hess_prefactor*dx*dz;
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 2, 1)] += -3*hess_prefactor*dy*dz;
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 2, 2)] += hess_prefactor*(d2ij - 3*d2z);
                }

                if(d2E_dxdp) {

                    RealType mp_prefactor = sij*ONE_4PI_EPS0*inv_d3ij;

                    RealType PREFACTOR_QI_GRAD = mp_prefactor*q1;
                    RealType PREFACTOR_QJ_GRAD = mp_prefactor*qi;

                    if(qi_g_idx >= 0) {
                        RealType *mp_out_q_h_i = d2E_dxdp + conf_idx*DP*N*3 + qi_g_idx*N*3;
                        atomicAdd(mp_out_q_h_i + h_j_idx*3 + 0, PREFACTOR_QI_GRAD * (dx));
                        atomicAdd(mp_out_q_h_i + h_j_idx*3 + 1, PREFACTOR_QI_GRAD * (dy));
                        atomicAdd(mp_out_q_h_i + h_j_idx*3 + 2, PREFACTOR_QI_GRAD * (dz));
                    }

                    if(q1_g_idx >= 0) {
                        RealType *mp_out_q_h_j = d2E_dxdp + conf_idx*DP*N*3 + q1_g_idx*N*3;
                        atomicAdd(mp_out_q_h_j + h_i_idx*3 + 0, PREFACTOR_QJ_GRAD * (-dx));
                        atomicAdd(mp_out_q_h_j + h_i_idx*3 + 1, PREFACTOR_QJ_GRAD * (-dy));
                        atomicAdd(mp_out_q_h_j + h_i_idx*3 + 2, PREFACTOR_QJ_GRAD * (-dz));; 
                    }

                }
            }
        }

        // diagonal elements and mixed partials
        for(int round=0; round < WARP_SIZE; round++) {

            j_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

            if(j_idx < i_idx && i_idx < N && j_idx < N) {

                RealType dx = x0 - x1;
                RealType dy = y0 - y1;
                RealType dz = z0 - z1;
                RealType d2x = dx*dx;
                RealType d2y = dy*dy;
                RealType d2z = dz*dz;

                RealType d2ij = d2x + d2y + d2z;
                RealType dij = sqrt(d2ij);
                RealType d3ij = d2ij*dij;
                RealType inv_d3ij = 1/d3ij;
                RealType d5ij = d3ij*d2ij;

                RealType sij = scale_matrix[i_idx*N + j_idx];
                RealType so4eq01 = sij*ONE_4PI_EPS0*q0*q1;
                RealType grad_prefactor = so4eq01*inv_d3ij;
                RealType hess_prefactor = so4eq01/d5ij;

                if(E) {
                    energy += (sij*ONE_4PI_EPS0*q0*q1)/dij;
                }

                if(dE_dp) {
                    dE_dp_q += (sij*ONE_4PI_EPS0*q1)/dij;
                    shfl_dE_dp_q += (sij*ONE_4PI_EPS0*q0)/dij;
                }

                grad_dx -= grad_prefactor*dx;
                grad_dy -= grad_prefactor*dy;
                grad_dz -= grad_prefactor*dz;

                shfl_grad_dx += grad_prefactor*dx;
                shfl_grad_dy += grad_prefactor*dy;
                shfl_grad_dz += grad_prefactor*dz;

                // (ytz) todo: optimize for individual dxdps
                if(d2E_dxdp) {

                    RealType mp_prefactor = sij*ONE_4PI_EPS0*inv_d3ij;

                    RealType PREFACTOR_QI_GRAD = mp_prefactor*q1;
                    RealType PREFACTOR_QJ_GRAD = mp_prefactor*q0;

                    // (ytz): We can further optimize the off-diagonal elements if desired.
                    mixed_dx += PREFACTOR_QI_GRAD * (-dx);
                    mixed_dy += PREFACTOR_QI_GRAD * (-dy);
                    mixed_dz += PREFACTOR_QI_GRAD * (-dz);

                    shfl_mixed_dx += PREFACTOR_QJ_GRAD * dx;
                    shfl_mixed_dy += PREFACTOR_QJ_GRAD * dy;
                    shfl_mixed_dz += PREFACTOR_QJ_GRAD * dz;

                }

                // hessians
                if(d2E_dx2) {
                    hess_xx += hess_prefactor*(-d2ij + 3*d2x);
                    hess_yx += 3*hess_prefactor*dx*dy;
                    hess_yy += hess_prefactor*(-d2ij + 3*d2y);
                    hess_zx += 3*hess_prefactor*dx*dz;
                    hess_zy += 3*hess_prefactor*dy*dz;
                    hess_zz += hess_prefactor*(-d2ij + 3*d2z);

                    shfl_hess_xx += hess_prefactor*(-d2ij + 3*d2x);
                    shfl_hess_yx += 3*hess_prefactor*dx*dy;
                    shfl_hess_yy += hess_prefactor*(-d2ij + 3*d2y);
                    shfl_hess_zx += 3*hess_prefactor*dx*dz;
                    shfl_hess_zy += 3*hess_prefactor*dy*dz;
                    shfl_hess_zz += hess_prefactor*(-d2ij + 3*d2z);

                }

            }

            int srcLane = (threadIdx.x + 1) % WARP_SIZE;

            // we should shuffle no matter what
            x1 = __shfl_sync(0xffffffff, x1, srcLane);
            y1 = __shfl_sync(0xffffffff, y1, srcLane);
            z1 = __shfl_sync(0xffffffff, z1, srcLane);
            q1 = __shfl_sync(0xffffffff, q1, srcLane);

            // add conditionals depending on if we do certain ops
            q1_g_idx = __shfl_sync(0xffffffff, q1_g_idx, srcLane);
            // eps1_g_idx = __shfl_sync(0xffffffff, eps1_g_idx, srcLane);

            shfl_dE_dp_q = __shfl_sync(0xffffffff, shfl_dE_dp_q, srcLane);
            // shfl_dE_dp_sig = __shfl_sync(0xffffffff, shfl_dE_dp_sig, srcLane);
            // shfl_dE_dp_eps = __shfl_sync(0xffffffff, shfl_dE_dp_eps, srcLane);

            shfl_grad_dx = __shfl_sync(0xffffffff, shfl_grad_dx, srcLane);
            shfl_grad_dy = __shfl_sync(0xffffffff, shfl_grad_dy, srcLane);
            shfl_grad_dz = __shfl_sync(0xffffffff, shfl_grad_dz, srcLane);

            shfl_mixed_dx = __shfl_sync(0xffffffff, shfl_mixed_dx, srcLane);
            shfl_mixed_dy = __shfl_sync(0xffffffff, shfl_mixed_dy, srcLane);
            shfl_mixed_dz = __shfl_sync(0xffffffff, shfl_mixed_dz, srcLane);

            shfl_hess_xx = __shfl_sync(0xffffffff, shfl_hess_xx, srcLane);
            shfl_hess_yx = __shfl_sync(0xffffffff, shfl_hess_yx, srcLane);
            shfl_hess_yy = __shfl_sync(0xffffffff, shfl_hess_yy, srcLane);
            shfl_hess_zx = __shfl_sync(0xffffffff, shfl_hess_zx, srcLane);
            shfl_hess_zy = __shfl_sync(0xffffffff, shfl_hess_zy, srcLane);
            shfl_hess_zz = __shfl_sync(0xffffffff, shfl_hess_zz, srcLane);

            j_idx += 1;

        }

        int target_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

        if(target_idx < N) {

            if(dE_dx) {
                atomicAdd(dE_dx + conf_idx*N*3 + target_idx*3 + 0, shfl_grad_dx);
                atomicAdd(dE_dx + conf_idx*N*3 + target_idx*3 + 1, shfl_grad_dy);
                atomicAdd(dE_dx + conf_idx*N*3 + target_idx*3 + 2, shfl_grad_dz);                
            }

            if(d2E_dx2) {
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 0, 0), shfl_hess_xx);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 1, 0), shfl_hess_yx);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 1, 1), shfl_hess_yy);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 2, 0), shfl_hess_zx);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 2, 1), shfl_hess_zy);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 2, 2), shfl_hess_zz);
            }

            if(dE_dp) {
                if(q1_g_idx >= 0) {
                    atomicAdd(dE_dp + conf_idx*DP + q1_g_idx, shfl_dE_dp_q);
                }
            }

            if(d2E_dxdp) {
                // optimize for only parameters we care about
                if(q1_g_idx >= 0) {
                    RealType *mp_out_q1 = d2E_dxdp + conf_idx*DP*N*3 + q1_g_idx*N*3;
                    atomicAdd(mp_out_q1 + target_idx*3 + 0, shfl_mixed_dx);
                    atomicAdd(mp_out_q1 + target_idx*3 + 1, shfl_mixed_dy);
                    atomicAdd(mp_out_q1 + target_idx*3 + 2, shfl_mixed_dz);
                }
            }

        }

    }

    if(i_idx < N) {

        if(E) {
            atomicAdd(E + conf_idx, energy);   
        }

        if(dE_dx) {
            atomicAdd(dE_dx + conf_idx*N*3 + i_idx*3 + 0, grad_dx);
            atomicAdd(dE_dx + conf_idx*N*3 + i_idx*3 + 1, grad_dy);
            atomicAdd(dE_dx + conf_idx*N*3 + i_idx*3 + 2, grad_dz);            
        }

        if(d2E_dx2) {
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 0, 0), hess_xx);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 1, 0), hess_yx);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 1, 1), hess_yy);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 2, 0), hess_zx);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 2, 1), hess_zy);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 2, 2), hess_zz);            
        }

        if(dE_dp) {
            if(q0_g_idx >= 0) {
                atomicAdd(dE_dp + conf_idx*DP + q0_g_idx, dE_dp_q);
            }
        }

        if(d2E_dxdp) {
            if(q0_g_idx >= 0) {
                RealType *mp_out_q0 = d2E_dxdp + conf_idx*DP*N*3 + q0_g_idx*N*3;                
                atomicAdd(mp_out_q0 + i_idx*3 + 0, mixed_dx);
                atomicAdd(mp_out_q0 + i_idx*3 + 1, mixed_dy);
                atomicAdd(mp_out_q0 + i_idx*3 + 2, mixed_dz);
            }
        }
    }

}