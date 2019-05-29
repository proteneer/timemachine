#pragma once

#include "kernel_utils.cuh"

template<typename RealType>
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
    
    RealType x0, y0, z0, sig0, eps0;
    int sig0_g_idx, eps0_g_idx;

    if(i_idx >= N) {
        x0 = 0.0;
        y0 = 0.0;
        z0 = 0.0;
        sig0 = 0.0;
        eps0 = 0.0;
        sig0_g_idx = 0;
        eps0_g_idx = 0;
    } else {
        x0 = coords[conf_idx*N*3+i_idx*3+0];
        y0 = coords[conf_idx*N*3+i_idx*3+1];
        z0 = coords[conf_idx*N*3+i_idx*3+2];
        sig0 = params[param_idxs[i_idx*2+0]];
        eps0 = params[param_idxs[i_idx*2+1]];

        sig0_g_idx = param_gather_idxs[param_idxs[i_idx*2+0]]; // may be -1
        eps0_g_idx = param_gather_idxs[param_idxs[i_idx*2+1]]; // may be -1

    }

    RealType grad_dx = 0;
    RealType grad_dy = 0;
    RealType grad_dz = 0;

    RealType mixed_dx_sig = 0;
    RealType mixed_dy_sig = 0;
    RealType mixed_dz_sig = 0;

    RealType mixed_dx_eps = 0;
    RealType mixed_dy_eps = 0;
    RealType mixed_dz_eps = 0;

    RealType hess_xx = 0;
    RealType hess_yx = 0;
    RealType hess_yy = 0;
    RealType hess_zx = 0;
    RealType hess_zy = 0;
    RealType hess_zz = 0;

    RealType energy = 0;

    int num_y_tiles = blockIdx.x + 1;

    for(int tile_y_idx = 0; tile_y_idx < num_y_tiles; tile_y_idx++) {

        RealType x1, y1, z1, sig1, eps1;
        int sig1_g_idx, eps1_g_idx;

        RealType shfl_grad_dx = 0;
        RealType shfl_grad_dy = 0;
        RealType shfl_grad_dz = 0;

        RealType shfl_mixed_dx_sig = 0;
        RealType shfl_mixed_dy_sig = 0;
        RealType shfl_mixed_dz_sig = 0;

        RealType shfl_mixed_dx_eps = 0;
        RealType shfl_mixed_dy_eps = 0;
        RealType shfl_mixed_dz_eps = 0;

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
            sig1 = 0.0;
            eps1 = 0.0;
            sig1_g_idx = 0;
            eps1_g_idx = 0;
        } else {
            x1 = coords[conf_idx*N*3+j_idx*3+0];
            y1 = coords[conf_idx*N*3+j_idx*3+1];
            z1 = coords[conf_idx*N*3+j_idx*3+2];
            sig1_g_idx = param_gather_idxs[param_idxs[j_idx*2+0]]; // may be -1
            eps1_g_idx = param_gather_idxs[param_idxs[j_idx*2+1]]; // may be -1
            sig1 = params[param_idxs[j_idx*2+0]];
            eps1 = params[param_idxs[j_idx*2+1]];
        }

        // off diagonal
        // iterate over a block of i's because we improve locality of writes to off diagonal elements
        // RealType *mp_out_sig_h_j = d2E_dxdp + conf_idx*DP*N*3 + sig1_g_idx*N*3;
        // RealType *mp_out_eps_h_j = d2E_dxdp + conf_idx*DP*N*3 + eps1_g_idx*N*3;

        // run only if inference
        // if(dE_dp || d2E_dxdp) {
        for(int round=0; round < WARP_SIZE; round++) {
            RealType xi = __shfl_sync(0xffffffff, x0, round);
            RealType yi = __shfl_sync(0xffffffff, y0, round);
            RealType zi = __shfl_sync(0xffffffff, z0, round);
            RealType sigi = __shfl_sync(0xffffffff, sig0, round);
            RealType epsi = __shfl_sync(0xffffffff, eps0, round);
            int sigi_g_idx = __shfl_sync(0xffffffff, sig0_g_idx, round);
            int epsi_g_idx = __shfl_sync(0xffffffff, eps0_g_idx, round);

            int h_i_idx = blockIdx.x*WARP_SIZE + round;
            int h_j_idx = j_idx;

            if(h_j_idx < h_i_idx && h_i_idx < N && h_j_idx < N) {

                RealType dx = xi - x1;
                RealType dy = yi - y1;
                RealType dz = zi - z1;
                RealType d2x = dx*dx;
                RealType d2y = dy*dy;
                RealType d2z = dz*dz;

                RealType sij = scale_matrix[h_i_idx*N + h_j_idx];

                RealType eps = sqrt(epsi * eps1);
                RealType sig = (sigi + sig1)/2;

                RealType d2ij = d2x + d2y + d2z;
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
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 0, 0)] += prefactor*-24*inv_d16ij*(d8ij- 2*d2ij*sig6 + d2x*(28*sig6 - 8*d6ij));
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 0, 1)] += common*dx*dy;
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 0, 2)] += common*dx*dz;

                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 1, 0)] += common*dx*dy;
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 1, 1)] += prefactor*-24*inv_d16ij*(d8ij- 2*d2ij*sig6 + d2y*(28*sig6 - 8*d6ij));
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 1, 2)] += common*dy*dz;

                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 2, 0)] += common*dx*dz;
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 2, 1)] += common*dy*dz;
                    d2E_dx2[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, N, 2, 2)] += prefactor*-24*inv_d16ij*(d8ij- 2*d2ij*sig6 + d2z*(28*sig6 - 8*d6ij));
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
                        RealType *mp_out_sig_h_i = d2E_dxdp + conf_idx*DP*N*3 + sigi_g_idx*N*3;
                        atomicAdd(mp_out_sig_h_i + h_j_idx*3 + 0,  SIG_PREFACTOR*dx);
                        atomicAdd(mp_out_sig_h_i + h_j_idx*3 + 1,  SIG_PREFACTOR*dy);
                        atomicAdd(mp_out_sig_h_i + h_j_idx*3 + 2,  SIG_PREFACTOR*dz);
                    }

                    if(sig1_g_idx >= 0) {
                        RealType *mp_out_sig_h_j = d2E_dxdp + conf_idx*DP*N*3 + sig1_g_idx*N*3;
                        atomicAdd(mp_out_sig_h_j + h_i_idx*3 + 0, -SIG_PREFACTOR*dx);
                        atomicAdd(mp_out_sig_h_j + h_i_idx*3 + 1, -SIG_PREFACTOR*dy);
                        atomicAdd(mp_out_sig_h_j + h_i_idx*3 + 2, -SIG_PREFACTOR*dz); 
                    }

                    if(epsi_g_idx >= 0) {
                        RealType *mp_out_eps_h_i = d2E_dxdp + conf_idx*DP*N*3 + epsi_g_idx*N*3;
                        atomicAdd(mp_out_eps_h_i + h_j_idx*3 + 0,  EPS_PREFACTOR*eps1*dx);
                        atomicAdd(mp_out_eps_h_i + h_j_idx*3 + 1,  EPS_PREFACTOR*eps1*dy);
                        atomicAdd(mp_out_eps_h_i + h_j_idx*3 + 2,  EPS_PREFACTOR*eps1*dz);
                    }

                    if(eps1_g_idx >= 0) {
                        RealType *mp_out_eps_h_j = d2E_dxdp + conf_idx*DP*N*3 + eps1_g_idx*N*3;
                        atomicAdd(mp_out_eps_h_j + h_i_idx*3 + 0, -EPS_PREFACTOR*epsi*dx);
                        atomicAdd(mp_out_eps_h_j + h_i_idx*3 + 1, -EPS_PREFACTOR*epsi*dy);
                        atomicAdd(mp_out_eps_h_j + h_i_idx*3 + 2, -EPS_PREFACTOR*epsi*dz);                            
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

                RealType sij = scale_matrix[i_idx*N + j_idx];
                RealType d2ij = d2x + d2y + d2z;
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
                    // dE/deps

                    // RealType eps = sqrt(epsi * eps1);
                    RealType dE_deps = sij*4*(sig6/d6ij-1.0)*sig6/d6ij;
                    RealType dE_deps0 = dE_deps*eps1/(2*eps);
                    RealType dE_deps1 = dE_deps*eps0/(2*eps);

                    // RealType sig = (sigi + sig1)/2;
                    RealType dE_dsig = sij*24*eps*(2*sig6/d6ij-1)*(sig5/d6ij);
                    RealType dE_dsig0 = dE_dsig/2;
                    RealType dE_dsig1 = dE_dsig/2;

                    // (ytz): shuffle these

                    if(eps0_g_idx >= 0) {
                        atomicAdd(dE_dp + conf_idx*DP + eps0_g_idx, dE_deps0);
                    }
                    if(eps1_g_idx >= 0) {
                        atomicAdd(dE_dp + conf_idx*DP + eps1_g_idx, dE_deps1);
                    }

                    if(sig0_g_idx >= 0) {
                        atomicAdd(dE_dp + conf_idx*DP + sig0_g_idx, dE_dsig0);
                    }

                    if(sig1_g_idx >= 0) {
                        atomicAdd(dE_dp + conf_idx*DP + sig1_g_idx, dE_dsig1);
                    }

                }

                // (ytz): 99% sure this loses precision so we need to refactor
                RealType sig1rij1 = sig/rij;
                RealType sig3rij3 = sig1rij1*sig1rij1*sig1rij1;
                RealType sig6rij3 = sig3*sig3rij3;
                RealType sig4rij4 = sig3rij3*sig1rij1;
                RealType sig5rij4 = sig*sig4rij4;
                RealType sig6rij4 = sig*sig5rij4;

                RealType sig12rij7 = sig12/rij7;

                RealType dEdx = 24*eps*dx*(sig12rij7*2 - sig6rij4);
                RealType dEdy = 24*eps*dy*(sig12rij7*2 - sig6rij4);
                RealType dEdz = 24*eps*dz*(sig12rij7*2 - sig6rij4);

                // TODO: multiply by sij when atomic adding (But this is trivial relative to cost
                // of hessians.
                grad_dx -= sij*dEdx;
                grad_dy -= sij*dEdy;
                grad_dz -= sij*dEdz;

                shfl_grad_dx += sij*dEdx;
                shfl_grad_dy += sij*dEdy;
                shfl_grad_dz += sij*dEdz;

                // (ytz) todo: optimize for individual dxdps
                if(d2E_dxdp) {

                    RealType EPS_PREFACTOR = sij*12/eps*(sig6rij4)*(2*sig6rij3 - 1);

                    mixed_dx_eps += -EPS_PREFACTOR*eps1*dx;
                    mixed_dy_eps += -EPS_PREFACTOR*eps1*dy;
                    mixed_dz_eps += -EPS_PREFACTOR*eps1*dz;

                    shfl_mixed_dx_eps += EPS_PREFACTOR*eps0*dx;
                    shfl_mixed_dy_eps += EPS_PREFACTOR*eps0*dy;
                    shfl_mixed_dz_eps += EPS_PREFACTOR*eps0*dz;

                    RealType SIG_PREFACTOR = sij*24*eps*(sig5/rij4)*(12*sig6rij3 - 3);

                    mixed_dx_sig += -SIG_PREFACTOR*dx;
                    mixed_dy_sig += -SIG_PREFACTOR*dy;
                    mixed_dz_sig += -SIG_PREFACTOR*dz;

                    shfl_mixed_dx_sig += SIG_PREFACTOR*dx;
                    shfl_mixed_dy_sig += SIG_PREFACTOR*dy;
                    shfl_mixed_dz_sig += SIG_PREFACTOR*dz;

                }

                // hessians
                if(d2E_dx2) {
                    RealType prefactor = sij*eps*sig6;
                    RealType diagonal_prefactor = prefactor*-96*(2*d6ij - 7*sig6)*inv_d16ij;

                    hess_xx += prefactor*24*(d8ij - 8*d6ij*d2x - 2*d2ij*sig6 + 28*d2x*sig6)*inv_d16ij;
                    hess_yx += diagonal_prefactor*dx*dy;
                    hess_yy += prefactor*24*(d8ij - 8*d6ij*d2y - 2*d2ij*sig6 + 28*d2y*sig6)*inv_d16ij;
                    hess_zx += diagonal_prefactor*dx*dz;
                    hess_zy += diagonal_prefactor*dy*dz;
                    hess_zz += prefactor*24*(d8ij - 8*d6ij*d2z - 2*d2ij*sig6 + 28*d2z*sig6)*inv_d16ij;
                    
                    shfl_hess_xx += prefactor*24*(d8ij - 8*d6ij*d2x - 2*d2ij*sig6 + 28*d2x*sig6)*inv_d16ij;
                    shfl_hess_yx += diagonal_prefactor*dx*dy;
                    shfl_hess_yy += prefactor*24*(d8ij - 8*d6ij*d2y - 2*d2ij*sig6 + 28*d2y*sig6)*inv_d16ij;
                    shfl_hess_zx += diagonal_prefactor*dx*dz;
                    shfl_hess_zy += diagonal_prefactor*dy*dz;
                    shfl_hess_zz += prefactor*24*(d8ij - 8*d6ij*d2z - 2*d2ij*sig6 + 28*d2z*sig6)*inv_d16ij;

                }

            }

            int srcLane = (threadIdx.x + 1) % WARP_SIZE;

            // we should shuffle no matter what


            x1 = __shfl_sync(0xffffffff, x1, srcLane);
            y1 = __shfl_sync(0xffffffff, y1, srcLane);
            z1 = __shfl_sync(0xffffffff, z1, srcLane);
            sig1 = __shfl_sync(0xffffffff, sig1, srcLane);
            eps1 = __shfl_sync(0xffffffff, eps1, srcLane);

            // add conditionals

            sig1_g_idx = __shfl_sync(0xffffffff, sig1_g_idx, srcLane);
            eps1_g_idx = __shfl_sync(0xffffffff, eps1_g_idx, srcLane);

            shfl_grad_dx = __shfl_sync(0xffffffff, shfl_grad_dx, srcLane);
            shfl_grad_dy = __shfl_sync(0xffffffff, shfl_grad_dy, srcLane);
            shfl_grad_dz = __shfl_sync(0xffffffff, shfl_grad_dz, srcLane);

            shfl_mixed_dx_sig = __shfl_sync(0xffffffff, shfl_mixed_dx_sig, srcLane);
            shfl_mixed_dy_sig = __shfl_sync(0xffffffff, shfl_mixed_dy_sig, srcLane);
            shfl_mixed_dz_sig = __shfl_sync(0xffffffff, shfl_mixed_dz_sig, srcLane);

            shfl_mixed_dx_eps = __shfl_sync(0xffffffff, shfl_mixed_dx_eps, srcLane);
            shfl_mixed_dy_eps = __shfl_sync(0xffffffff, shfl_mixed_dy_eps, srcLane);
            shfl_mixed_dz_eps = __shfl_sync(0xffffffff, shfl_mixed_dz_eps, srcLane);

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

            if(d2E_dxdp) {
                // optimize for only parameters we care about
                if(sig1_g_idx >= 0) {
                    RealType *mp_out_sig1 = d2E_dxdp + conf_idx*DP*N*3 + sig1_g_idx*N*3;
                    atomicAdd(mp_out_sig1 + target_idx*3 + 0, shfl_mixed_dx_sig);
                    atomicAdd(mp_out_sig1 + target_idx*3 + 1, shfl_mixed_dy_sig);
                    atomicAdd(mp_out_sig1 + target_idx*3 + 2, shfl_mixed_dz_sig);                    
                }

                if(eps1_g_idx >= 0) {
                    RealType *mp_out_eps1 = d2E_dxdp + conf_idx*DP*N*3 + eps1_g_idx*N*3;
                    atomicAdd(mp_out_eps1 + target_idx*3 + 0, shfl_mixed_dx_eps);
                    atomicAdd(mp_out_eps1 + target_idx*3 + 1, shfl_mixed_dy_eps);
                    atomicAdd(mp_out_eps1 + target_idx*3 + 2, shfl_mixed_dz_eps);                    
                }

            }

            if(d2E_dx2) {
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 0, 0), shfl_hess_xx);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 1, 0), shfl_hess_yx);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 1, 1), shfl_hess_yy);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 2, 0), shfl_hess_zx);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 2, 1), shfl_hess_zy);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, N, 2, 2), shfl_hess_zz);                
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


        if(d2E_dxdp) {
            if(sig0_g_idx >= 0) {
                RealType *mp_out_sig0 = d2E_dxdp + conf_idx*DP*N*3 + sig0_g_idx*N*3;                
                atomicAdd(mp_out_sig0 + i_idx*3 + 0, mixed_dx_sig);
                atomicAdd(mp_out_sig0 + i_idx*3 + 1, mixed_dy_sig);
                atomicAdd(mp_out_sig0 + i_idx*3 + 2, mixed_dz_sig);
            }

            if(eps0_g_idx >= 0) {
                RealType *mp_out_eps0 = d2E_dxdp + conf_idx*DP*N*3 + eps0_g_idx*N*3;
                atomicAdd(mp_out_eps0 + i_idx*3 + 0, mixed_dx_eps);
                atomicAdd(mp_out_eps0 + i_idx*3 + 1, mixed_dy_eps);
                atomicAdd(mp_out_eps0 + i_idx*3 + 2, mixed_dz_eps);
            }

        }

        if(d2E_dx2) {
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 0, 0), hess_xx);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 1, 0), hess_yx);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 1, 1), hess_yy);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 2, 0), hess_zx);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 2, 1), hess_zy);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, N, 2, 2), hess_zz);            
        }
    }

}