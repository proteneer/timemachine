#pragma once

#include "kernel_utils.cuh"

template<typename RealType, size_t NDIMS>
void __global__ k_lennard_jones(
    const int num_atoms,    // n
    const RealType *coords, // [n, 3]
    const RealType *params, // [p,]
    const RealType *scale_matrix, // [n, n]
    const int *param_idxs,  // [n, 2] sig eps
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
    
    RealType X0[NDIMS] = {0};
    RealType sig0, eps0;
    int sig0_g_idx, eps0_g_idx;

    if(i_idx >= N) {
        sig0 = 0.0;
        eps0 = 0.0;
        sig0_g_idx = 0;
        eps0_g_idx = 0;
    } else {
        #pragma unroll
        for(size_t d=0; d < NDIMS; d++) {
            X0[d] = coords[conf_idx*N*NDIMS+i_idx*NDIMS+d];
        }
        sig0 = params[param_idxs[i_idx*2+0]];
        eps0 = params[param_idxs[i_idx*2+1]];

        sig0_g_idx = param_gather_idxs[param_idxs[i_idx*2+0]]; // may be -1
        eps0_g_idx = param_gather_idxs[param_idxs[i_idx*2+1]]; // may be -1

    }

    RealType grad_X[NDIMS] = {0};
    RealType mixed_X_sig[NDIMS] = {0};
    RealType mixed_X_eps[NDIMS] = {0};
    RealType hess_X[NDIMS*(NDIMS-1)] = {0};

    RealType dE_dp_sig = 0;
    RealType dE_dp_eps = 0;
    RealType energy = 0;

    int num_y_tiles = blockIdx.x + 1;

    for(int tile_y_idx = 0; tile_y_idx < num_y_tiles; tile_y_idx++) {

        RealType X1[NDIMS] = {0};
        RealType sig1, eps1;
        int sig1_g_idx, eps1_g_idx;


        RealType shfl_grad_X[NDIMS] = {0};
        RealType shfl_mixed_X_sig[NDIMS] = {0};
        RealType shfl_mixed_X_eps[NDIMS] = {0};
        RealType shfl_hess_X[NDIMS*(NDIMS-1)] = {0};

        RealType shfl_dE_dp_sig = 0;
        RealType shfl_dE_dp_eps = 0;

        // load diagonal elements exactly once, shuffle the rest
        int j_idx = tile_y_idx*WARP_SIZE + threadIdx.x;

        if(j_idx >= N) {
            sig1 = 0.0;
            eps1 = 0.0;
            sig1_g_idx = 0;
            eps1_g_idx = 0;
        } else {
            for(size_t d=0; d < NDIMS; d++) {
                X1[d] = coords[conf_idx*N*NDIMS+j_idx*NDIMS+d];
            }
            sig1_g_idx = param_gather_idxs[param_idxs[j_idx*2+0]]; // may be -1
            eps1_g_idx = param_gather_idxs[param_idxs[j_idx*2+1]]; // may be -1
            sig1 = params[param_idxs[j_idx*2+0]];
            eps1 = params[param_idxs[j_idx*2+1]];
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
            RealType sigi = __shfl_sync(0xffffffff, sig0, round);
            RealType epsi = __shfl_sync(0xffffffff, eps0, round);
            int sigi_g_idx = __shfl_sync(0xffffffff, sig0_g_idx, round);
            int epsi_g_idx = __shfl_sync(0xffffffff, eps0_g_idx, round);

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

                RealType sij = scale_matrix[h_i_idx*N + h_j_idx];

                RealType eps = sqrt(epsi * eps1);
                RealType sig = (sigi + sig1)/2;

                // RealType d2ij = d2x + d2y + d2z;
                RealType dij = sqrt(d2ij);
                RealType d4ij = d2ij*d2ij;
                RealType d6ij = d4ij*d2ij;
                RealType d8ij = d4ij*d4ij;
                RealType d16ij = d8ij*d8ij;
                RealType inv_d16ij = 1.0/d16ij;

                RealType sig2 = sig*sig;
                RealType sig3 = sig2*sig;
                RealType sig6 = sig3*sig3;
                RealType prefactor = sij*eps*sig6;

                RealType common = prefactor*96*(2*d6ij - 7*sig6)*inv_d16ij;

                if(d2E_dx2) {
                    // don't need atomic adds because these are unique diagonals
                    for(size_t d0=0; d0 < NDIMS; d0++) {
                        for(size_t d1=0; d1 < NDIMS; d1++) {
                            if(d0 != d1) {
                                d2E_dx2[conf_idx*N*NDIMS*N*NDIMS+HESS_IDX_ND(h_i_idx, h_j_idx, N, d0, d1, NDIMS)] += common*DX[d0]*DX[d1];
                            }
                        }
                        d2E_dx2[conf_idx*N*NDIMS*N*NDIMS+HESS_IDX_ND(h_i_idx, h_j_idx, N, d0, d0, NDIMS)] += prefactor*-24*inv_d16ij*(d8ij- 2*d2ij*sig6 + D2X[d0]*(28*sig6 - 8*d6ij));
                    }

                }

                if(d2E_dxdp) {

                    RealType sig5 = sig3*sig2;
                    RealType rij = d2ij;
                    RealType rij4 = d8ij;

                    // (ytz): 99 % sure this loses precision so we need to refactor
                    RealType sig1rij1 = sig/rij;
                    RealType sig3rij3 = sig1rij1*sig1rij1*sig1rij1;
                    RealType sig6rij3 = sig3*sig3rij3;
                    RealType sig4rij4 = sig3rij3*sig1rij1;
                    RealType sig5rij4 = sig*sig4rij4;
                    RealType sig6rij4 = sig*sig5rij4;

                    RealType EPS_PREFACTOR = sij*12/eps*(sig6rij4)*(2*sig6rij3 - 1);
                    RealType SIG_PREFACTOR = sij*24*eps*(sig5/rij4)*(12*sig6rij3 - 3);

                    if(sigi_g_idx >= 0) {
                        RealType *mp_out_sig_h_i = d2E_dxdp + conf_idx*DP*N*NDIMS + sigi_g_idx*N*NDIMS;
                        for(size_t d=0; d < NDIMS; d++) {
                            atomicAdd(mp_out_sig_h_i + h_j_idx*NDIMS + d, SIG_PREFACTOR * DX[d]);
                        }
                        // atomicAdd(mp_out_sig_h_i + h_j_idx*NDIMS + 0,  SIG_PREFACTOR*dx);
                        // atomicAdd(mp_out_sig_h_i + h_j_idx*NDIMS + 1,  SIG_PREFACTOR*dy);
                        // atomicAdd(mp_out_sig_h_i + h_j_idx*NDIMS + 2,  SIG_PREFACTOR*dz);
                    }

                    if(sig1_g_idx >= 0) {
                        RealType *mp_out_sig_h_j = d2E_dxdp + conf_idx*DP*N*NDIMS + sig1_g_idx*N*NDIMS;
                        for(size_t d=0; d < NDIMS; d++) {
                            atomicAdd(mp_out_sig_h_j + h_i_idx*NDIMS + d, -SIG_PREFACTOR*DX[d]);
                        }
                        // atomicAdd(mp_out_sig_h_j + h_i_idx*NDIMS + 1, -SIG_PREFACTOR*dy);
                        // atomicAdd(mp_out_sig_h_j + h_i_idx*NDIMS + 2, -SIG_PREFACTOR*dz); 
                    }

                    if(epsi_g_idx >= 0) {
                        RealType *mp_out_eps_h_i = d2E_dxdp + conf_idx*DP*N*NDIMS + epsi_g_idx*N*NDIMS;
                        for(size_t d=0; d < NDIMS; d++) {
                            atomicAdd(mp_out_eps_h_i + h_j_idx*NDIMS + d,  EPS_PREFACTOR*eps1*DX[d]);
                        }
                        // atomicAdd(mp_out_eps_h_i + h_j_idx*NDIMS + 0,  EPS_PREFACTOR*eps1*dx);
                        // atomicAdd(mp_out_eps_h_i + h_j_idx*NDIMS + 1,  EPS_PREFACTOR*eps1*dy);
                        // atomicAdd(mp_out_eps_h_i + h_j_idx*NDIMS + 2,  EPS_PREFACTOR*eps1*dz);
                    }

                    if(eps1_g_idx >= 0) {
                        RealType *mp_out_eps_h_j = d2E_dxdp + conf_idx*DP*N*NDIMS + eps1_g_idx*N*NDIMS;
                        for(size_t d=0; d < NDIMS; d++) {
                            atomicAdd(mp_out_eps_h_j + h_i_idx*NDIMS + d, -EPS_PREFACTOR*epsi*DX[d]);
                        }
                        // atomicAdd(mp_out_eps_h_j + h_i_idx*NDIMS + 1, -EPS_PREFACTOR*epsi*dy);
                        // atomicAdd(mp_out_eps_h_j + h_i_idx*NDIMS + 2, -EPS_PREFACTOR*epsi*dz);                        
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

                RealType sij = scale_matrix[i_idx*N + j_idx];
                // RealType d2ij = d2x + d2y + d2z;
                RealType dij = sqrt(d2ij);
                RealType d4ij = d2ij*d2ij;
                RealType d6ij = d4ij*d2ij;
                RealType d8ij = d4ij*d4ij;
                RealType d16ij = d8ij*d8ij;
                RealType inv_d16ij = 1.0/d16ij;

                RealType eps = sqrt(eps0*eps1);
                RealType sig = (sig0 + sig1)/2;

                RealType sig2 = sig*sig;
                RealType sig3 = sig2*sig;
                RealType sig5 = sig3*sig2;
                RealType sig6 = sig3*sig3;
                RealType sig12 = sig6*sig6;

                RealType rij = d2ij;
                RealType rij3 = d6ij;
                RealType rij4 = d8ij;
                RealType rij7 = rij4 * rij3;

                if(E) {
                    energy += sij*4*eps*(sig6/d6ij-1.0)*sig6/d6ij;
                }

                if(dE_dp) {
                    RealType dE_deps = sij*4*(sig6/d6ij-1.0)*sig6/d6ij;
                    dE_dp_eps += dE_deps*eps1/(2*eps);
                    shfl_dE_dp_eps += dE_deps*eps0/(2*eps);
                    RealType dE_dsig = sij*24*eps*(2*sig6/d6ij-1)*(sig5/d6ij);
                    dE_dp_sig += dE_dsig/2;
                    shfl_dE_dp_sig += dE_dsig/2;
                }

                // (ytz): 99% sure this loses precision so we need to refactor
                RealType sig1rij1 = sig/rij;
                RealType sig3rij3 = sig1rij1*sig1rij1*sig1rij1;
                RealType sig6rij3 = sig3*sig3rij3;
                RealType sig4rij4 = sig3rij3*sig1rij1;
                RealType sig5rij4 = sig*sig4rij4;
                RealType sig6rij4 = sig*sig5rij4;

                RealType sig12rij7 = sig12/rij7;


                RealType dEpre = 24*eps*(sig12rij7*2 - sig6rij4);

                // TODO: multiply by sij when atomic adding (But this is trivial relative to cost
                // of hessians.
                for(size_t d=0; d < NDIMS; d++) {
                    grad_X[d] -= sij*dEpre*DX[d];
                    shfl_grad_X[d] += sij*dEpre*DX[d];
                }
                // (ytz) todo: optimize for individual dxdps
                if(d2E_dxdp) {

                    RealType EPS_PREFACTOR = sij*12/eps*(sig6rij4)*(2*sig6rij3 - 1);
                    RealType SIG_PREFACTOR = sij*24*eps*(sig5/rij4)*(12*sig6rij3 - 3);

                    for(size_t d=0; d < NDIMS; d++) {
                        mixed_X_eps[d] -= EPS_PREFACTOR * eps1 * DX[d];
                        mixed_X_sig[d] -= SIG_PREFACTOR * DX[d];

                        shfl_mixed_X_eps[d] += EPS_PREFACTOR * eps0 * DX[d];
                        shfl_mixed_X_sig[d] += SIG_PREFACTOR * DX[d];
                    }

                }

                // hessians
                if(d2E_dx2) {
                    RealType prefactor = sij*eps*sig6;
                    RealType diagonal_prefactor = prefactor*-96*(2*d6ij - 7*sig6)*inv_d16ij;
                    for(size_t d0 = 0; d0 < NDIMS; d0++) {
                        // off-diagonal
                        for(size_t d1 = d0+1; d1 < NDIMS; d1++) {
                            RealType delta_off = diagonal_prefactor*DX[d0]*DX[d1];
                            hess_X[linearize(d0,d1,NDIMS)] += delta_off;
                            shfl_hess_X[linearize(d0,d1,NDIMS)] += delta_off;
                        }
                        // todo: optimize if needed
                        RealType delta_on = prefactor*24*(d8ij - 8*d6ij*D2X[d0] - 2*d2ij*sig6 + 28*D2X[d0]*sig6)*inv_d16ij;
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
                shfl_mixed_X_eps[d] = __shfl_sync(0xffffffff, shfl_mixed_X_eps[d], srcLane);
                shfl_mixed_X_sig[d] = __shfl_sync(0xffffffff, shfl_mixed_X_sig[d], srcLane);
            }

            for(size_t d=0; d < NDIMS*(NDIMS-1); d++) {
                shfl_hess_X[d] = __shfl_sync(0xffffffff, shfl_hess_X[d], srcLane);
            }


            sig1 = __shfl_sync(0xffffffff, sig1, srcLane);
            eps1 = __shfl_sync(0xffffffff, eps1, srcLane);

            // add conditionals depending on if we do certain ops
            sig1_g_idx = __shfl_sync(0xffffffff, sig1_g_idx, srcLane);
            eps1_g_idx = __shfl_sync(0xffffffff, eps1_g_idx, srcLane);

            shfl_dE_dp_sig = __shfl_sync(0xffffffff, shfl_dE_dp_sig, srcLane);
            shfl_dE_dp_eps = __shfl_sync(0xffffffff, shfl_dE_dp_eps, srcLane);

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
                if(sig1_g_idx >= 0) {
                    atomicAdd(dE_dp + conf_idx*DP + sig1_g_idx, shfl_dE_dp_sig);
                }

                if(eps1_g_idx >= 0) {
                    atomicAdd(dE_dp + conf_idx*DP + eps1_g_idx, shfl_dE_dp_eps);
                }
            }

            if(d2E_dxdp) {
                // optimize for only parameters we care about
                if(sig1_g_idx >= 0) {
                    RealType *mp_out_sig1 = d2E_dxdp + conf_idx*DP*N*NDIMS + sig1_g_idx*N*NDIMS;
                    for(size_t d=0; d < NDIMS; d++) {
                        atomicAdd(mp_out_sig1 + target_idx*NDIMS + d, shfl_mixed_X_sig[d]);
                    }
                }

                if(eps1_g_idx >= 0) {
                    RealType *mp_out_eps1 = d2E_dxdp + conf_idx*DP*N*NDIMS + eps1_g_idx*N*NDIMS;
                    for(size_t d=0; d < NDIMS; d++) {
                        atomicAdd(mp_out_eps1 + target_idx*NDIMS + d, shfl_mixed_X_eps[d]);
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
            if(sig0_g_idx >= 0) {
                atomicAdd(dE_dp + conf_idx*DP + sig0_g_idx, dE_dp_sig);
            }

            if(eps0_g_idx >= 0) {
                atomicAdd(dE_dp + conf_idx*DP + eps0_g_idx, dE_dp_eps);
            }
        }

        if(d2E_dxdp) {
            if(sig0_g_idx >= 0) {
                RealType *mp_out_sig0 = d2E_dxdp + conf_idx*DP*N*NDIMS + sig0_g_idx*N*NDIMS;
                for(size_t d=0; d < NDIMS; d++) {
                    atomicAdd(mp_out_sig0 + i_idx*NDIMS + d, mixed_X_sig[d]);
                }
            }

            if(eps0_g_idx >= 0) {
                RealType *mp_out_eps0 = d2E_dxdp + conf_idx*DP*N*NDIMS + eps0_g_idx*N*NDIMS;
                for(size_t d=0; d < NDIMS; d++) {
                    atomicAdd(mp_out_eps0 + i_idx*NDIMS + d, mixed_X_eps[d]);
                }
            }
        }

    }

}