#pragma once
#include "kernel_utils.cuh"

template<typename RealType, size_t NDIMS>
void __global__ k_electrostatics(
    const int num_atoms,    // n
    const RealType *coords, // [n, D]
    const RealType *params, // [p,]
    const RealType *scale_matrix, // [n, n]
    const int *param_idxs,  // [n, 1] charge
    RealType *E,             // [,] or null
    RealType *dE_dx,         // [n,D] or null
    RealType *d2E_dx2,       // [C, n, D, n, D] or null, hessian
    // parameters used for computing derivatives
    const int num_dp,        // dp, number of parameters we're differentiating w.r.t. 
    const int *param_gather_idxs, // [p,] if -1, then we discard
    RealType *dE_dp,         // [C, dp,] or null
    RealType *d2E_dxdp       // [C, dp, n, D] or null
) {

    const auto conf_idx = blockIdx.z;
    const int N = num_atoms;
    const int DP = num_dp;

    auto i_idx = blockDim.x*blockIdx.x + threadIdx.x;
    RealType q0;
    RealType X0[NDIMS] = {0};
    int q0_g_idx;

    if(i_idx >= N) {
        q0 = 0.0;
        q0_g_idx = 0;
    } else {
        #pragma unroll
        for(size_t d=0; d < NDIMS; d++) {
            X0[d] = coords[conf_idx*N*NDIMS+i_idx*NDIMS+d];
        }
        q0 = params[param_idxs[i_idx]];
        q0_g_idx = param_gather_idxs[param_idxs[i_idx]];

    }


    RealType grad_X[NDIMS] = {0};
    RealType mixed_X[NDIMS] = {0};
    RealType hess_X[NDIMS*(NDIMS-1)] = {0};

    RealType dE_dp_q = 0;
    RealType energy = 0;

    int num_y_tiles = blockIdx.x + 1;

    for(int tile_y_idx = 0; tile_y_idx < num_y_tiles; tile_y_idx++) {

        RealType X1[NDIMS] = {0};
        RealType q1;
        int q1_g_idx;

        RealType shfl_grad_X[NDIMS] = {0};
        RealType shfl_mixed_X[NDIMS] = {0};
        RealType shfl_hess_X[NDIMS*(NDIMS-1)] = {0};

        RealType shfl_dE_dp_q = 0;

        // load diagonal elements exactly once, shuffle the rest
        int j_idx = tile_y_idx*WARP_SIZE + threadIdx.x;

        if(j_idx >= N) {
            q1 = 0.0;
            q1_g_idx = 0;
        } else {
            #pragma unroll
            for(size_t d=0; d < NDIMS; d++) {
                X1[d] = coords[conf_idx*N*NDIMS+j_idx*NDIMS+d];
            }
            q1 = params[param_idxs[j_idx]];
            q1_g_idx = param_gather_idxs[param_idxs[j_idx]];
        }

        // off diagonal
        // iterate over a block of i's because we improve locality of writes to off diagonal elements
        // add a conditional for inference mode. (if d2E_dx2 and d2E_dxdp)
        for(int round=0; round < WARP_SIZE; round++) {
            RealType XI[NDIMS] = {0};
            #pragma unroll
            for(size_t d=0; d < NDIMS; d++) {
                XI[d] = __shfl_sync(0xffffffff, X0[d], round);
            }
            RealType qi = __shfl_sync(0xffffffff, q0, round);
            int qi_g_idx = __shfl_sync(0xffffffff, q0_g_idx, round);

            int h_i_idx = blockIdx.x*WARP_SIZE + round;
            int h_j_idx = j_idx;

            if(h_j_idx < h_i_idx && h_i_idx < N && h_j_idx < N) {

                RealType DX[NDIMS];
                RealType D2X[NDIMS];
                RealType d2ij = 0;
                #pragma unroll
                for(size_t d=0; d < NDIMS; d++) {
                    RealType dx = XI[d] - X1[d];
                    DX[d] = dx;
                    D2X[d] = dx*dx;
                    d2ij += dx*dx;
                }

                RealType dij = sqrt(d2ij);
                RealType d3ij = d2ij*dij;
                RealType inv_d3ij = 1/d3ij;
                RealType d5ij = d3ij*d2ij;

                RealType sij = scale_matrix[h_i_idx*N + h_j_idx];
                RealType so4eq01 = sij*ONE_4PI_EPS0*qi*q1;
                RealType hess_prefactor = so4eq01/d5ij;

                if(d2E_dx2) {
                    // don't need atomic adds because these are unique diagonals
                    // (ytz): this isn't necessary if we don't mind doing the diagonal twice.

                    for(size_t d0=0; d0 < NDIMS; d0++) {
                        for(size_t d1=0; d1 < NDIMS; d1++) {
                            if(d0 != d1) {
                                d2E_dx2[conf_idx*N*NDIMS*N*NDIMS+HESS_IDX_ND(h_i_idx, h_j_idx, N, d0, d1, NDIMS)] += -3*hess_prefactor*DX[d0]*DX[d1];
                            }
                        }
                        d2E_dx2[conf_idx*N*NDIMS*N*NDIMS+HESS_IDX_ND(h_i_idx, h_j_idx, N, d0, d0, NDIMS)] += hess_prefactor*(d2ij - 3*D2X[d0]);
                    }

                }

                if(d2E_dxdp) {

                    RealType mp_prefactor = sij*ONE_4PI_EPS0*inv_d3ij;

                    RealType PREFACTOR_QI_GRAD = mp_prefactor*q1;
                    RealType PREFACTOR_QJ_GRAD = mp_prefactor*qi;

                    if(qi_g_idx >= 0) {
                        RealType *mp_out_q_h_i = d2E_dxdp + conf_idx*DP*N*NDIMS + qi_g_idx*N*NDIMS;
                        for(size_t d=0; d < NDIMS; d++) {
                            atomicAdd(mp_out_q_h_i + h_j_idx*NDIMS + d, PREFACTOR_QI_GRAD * DX[d]);
                        }
                    }

                    if(q1_g_idx >= 0) {
                        RealType *mp_out_q_h_j = d2E_dxdp + conf_idx*DP*N*NDIMS + q1_g_idx*N*NDIMS;
                        for(size_t d=0; d < NDIMS; d++) {
                            atomicAdd(mp_out_q_h_j + h_i_idx*NDIMS + d, -PREFACTOR_QJ_GRAD * DX[d]);
                        }
                    }

                }
            }
        }

        // diagonal elements and mixed partials
        for(int round=0; round < WARP_SIZE; round++) {

            j_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

            if(j_idx < i_idx && i_idx < N && j_idx < N) {

                RealType DX[NDIMS];
                RealType D2X[NDIMS];
                RealType d2ij = 0;
                #pragma unroll
                for(size_t d=0; d < NDIMS; d++) {
                    RealType dx = X0[d] - X1[d];
                    DX[d] = dx;
                    D2X[d] = dx*dx;
                    d2ij += dx*dx;
                }

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

                for(size_t d=0; d < NDIMS; d++) {
                    grad_X[d] -= grad_prefactor*DX[d];
                    shfl_grad_X[d] += grad_prefactor*DX[d];
                }

                // (ytz) todo: optimize for individual dxdps
                if(d2E_dxdp) {

                    RealType mp_prefactor = sij*ONE_4PI_EPS0*inv_d3ij;

                    RealType PREFACTOR_QI_GRAD = mp_prefactor*q1;
                    RealType PREFACTOR_QJ_GRAD = mp_prefactor*q0;

                    // (ytz): We can further optimize the off-diagonal elements if desired.
                    for(size_t d=0; d < NDIMS; d++) {
                        mixed_X[d] -= PREFACTOR_QI_GRAD * DX[d]; 
                        shfl_mixed_X[d] += PREFACTOR_QJ_GRAD * DX[d];
                    }
                }

                // URT hessians
                if(d2E_dx2) {

                    // RENABLE
                    for(size_t d0 = 0; d0 < NDIMS; d0++) {
                        // off-diagonal
                        for(size_t d1 = d0+1; d1 < NDIMS; d1++) {
                            RealType delta_off = 3*hess_prefactor*DX[d0]*DX[d1];
                            hess_X[linearize(d0,d1,NDIMS)] += delta_off;
                            shfl_hess_X[linearize(d0,d1,NDIMS)] += delta_off;
                        }
                        RealType delta_on = hess_prefactor*(-d2ij + 3*D2X[d0]);
                        hess_X[linearize(d0, d0, NDIMS)] += delta_on;
                        shfl_hess_X[linearize(d0, d0, NDIMS)] += delta_on;
                    }
                }

            }

            int srcLane = (threadIdx.x + 1) % WARP_SIZE;

            // we should shuffle no matter what
            for(size_t d=0; d < NDIMS; d++) {
                X1[d] = __shfl_sync(0xffffffff, X1[d], srcLane);
                shfl_grad_X[d] = __shfl_sync(0xffffffff, shfl_grad_X[d], srcLane);
                shfl_mixed_X[d] = __shfl_sync(0xffffffff, shfl_mixed_X[d], srcLane);
            }

            for(size_t d=0; d < NDIMS*(NDIMS-1); d++) {
                shfl_hess_X[d] = __shfl_sync(0xffffffff, shfl_hess_X[d], srcLane);
            }
            q1 = __shfl_sync(0xffffffff, q1, srcLane);
            q1_g_idx = __shfl_sync(0xffffffff, q1_g_idx, srcLane);

            shfl_dE_dp_q = __shfl_sync(0xffffffff, shfl_dE_dp_q, srcLane);
            j_idx += 1;

        }

        int target_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

        if(target_idx < N) {

            if(dE_dx) {
                for(size_t d=0; d < NDIMS; d++) {
                    atomicAdd(dE_dx + conf_idx*N*NDIMS + target_idx*NDIMS + d, shfl_grad_X[d]);
                }             
            }

            if(d2E_dx2) {
                for(size_t d0=0; d0 < NDIMS; d0++) {
                    for(size_t d1=d0; d1 < NDIMS; d1++) {
                        atomicAdd(d2E_dx2 + conf_idx*N*NDIMS*N*NDIMS + HESS_IDX_ND(target_idx, target_idx, N, d1, d0, NDIMS), shfl_hess_X[linearize(d0, d1, NDIMS)]);                        
                    }
                }
            }

            if(dE_dp) {
                if(q1_g_idx >= 0) {
                    atomicAdd(dE_dp + conf_idx*DP + q1_g_idx, shfl_dE_dp_q);
                }
            }

            if(d2E_dxdp) {
                if(q1_g_idx >= 0) {
                    RealType *mp_out_q1 = d2E_dxdp + conf_idx*DP*N*NDIMS + q1_g_idx*N*NDIMS;
                    for(size_t d=0; d < NDIMS; d++) {
                        atomicAdd(mp_out_q1 + target_idx*NDIMS + d, shfl_mixed_X[d]);
                    }
                }
            }

        }

    }

    if(i_idx < N) {

        if(E) {
            atomicAdd(E + conf_idx, energy);   
        }

        if(dE_dx) {
            for(size_t d=0; d < NDIMS; d++) {
                atomicAdd(dE_dx + conf_idx*N*NDIMS + i_idx*NDIMS + d, grad_X[d]);
            }  
        }

        if(d2E_dx2) {
            for(size_t d0=0; d0 < NDIMS; d0++) {
                for(size_t d1=d0; d1 < NDIMS; d1++) {
                    atomicAdd(d2E_dx2 + conf_idx*N*NDIMS*N*NDIMS + HESS_IDX_ND(i_idx, i_idx, N, d1, d0, NDIMS), hess_X[linearize(d0, d1, NDIMS)]);
                }
            }
        }

        if(dE_dp) {
            if(q0_g_idx >= 0) {
                atomicAdd(dE_dp + conf_idx*DP + q0_g_idx, dE_dp_q);
            }
        }

        if(d2E_dxdp) {
            if(q0_g_idx >= 0) {
                RealType *mp_out_q0 = d2E_dxdp + conf_idx*DP*N*NDIMS + q0_g_idx*N*NDIMS;
                for(size_t d=0; d < NDIMS; d++) {
                    atomicAdd(mp_out_q0 + i_idx*NDIMS + d, mixed_X[d]);
                }
            }
        }
    }

}