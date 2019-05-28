#pragma once
#include "kernel_utils.cuh"

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
    
    RealType *mp_out_sig0 = nullptr;
    RealType *mp_out_eps0 = nullptr;

    RealType x0, y0, z0, sig0, eps0;
    int sig0_g_idx, eps0_g_idx;

    if(i_idx >= n_atoms) {
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

        if(j_idx >= n_atoms) {
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

        if(dE_dp || d2E_dxdp) {
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

                // RealType *mp_out_sig_h_i = d2E_dxdp + conf_idx*DP*N*3 + sigi_g_idx;
                // RealType *mp_out_eps_h_i = d2E_dxdp + conf_idx*DP*N*3 + epsi_g_idx;

                if(h_j_idx < h_i_idx && h_i_idx < n_atoms && h_j_idx < n_atoms) {

                    RealType dx = xi - x1;
                    RealType dy = yi - y1;
                    RealType dz = zi - z1;
                    RealType d2x = dx*dx;
                    RealType d2y = dy*dy;
                    RealType d2z = dz*dz;

                    RealType sij = scale_matrix[h_i_idx*n_atoms + h_j_idx];

                    RealType eps = gpuSqrt(epsi * eps1);
                    RealType sig = (sigi + sig1)/2;

                    RealType d2ij = d2x + d2y + d2z;
                    RealType dij = gpuSqrt(d2ij);
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
                        hessian_out[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, n_atoms, 0, 0)] += prefactor*-24*inv_d16ij*(d8ij- 2*d2ij*sig6 + d2x*(28*sig6 - 8*d6ij));
                        hessian_out[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, n_atoms, 0, 1)] += common*dx*dy;
                        hessian_out[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, n_atoms, 0, 2)] += common*dx*dz;

                        hessian_out[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, n_atoms, 1, 0)] += common*dx*dy;
                        hessian_out[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, n_atoms, 1, 1)] += prefactor*-24*inv_d16ij*(d8ij- 2*d2ij*sig6 + d2y*(28*sig6 - 8*d6ij));
                        hessian_out[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, n_atoms, 1, 2)] += common*dy*dz;

                        hessian_out[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, n_atoms, 2, 0)] += common*dx*dz;
                        hessian_out[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, n_atoms, 2, 1)] += common*dy*dz;
                        hessian_out[conf_idx*N*3*N*3+HESS_IDX(h_i_idx, h_j_idx, n_atoms, 2, 2)] += prefactor*-24*inv_d16ij*(d8ij- 2*d2ij*sig6 + d2z*(28*sig6 - 8*d6ij));
                    }

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

                    if(d2E_dxdp) {
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

        }

        // diagonal elements and mixed partials
        for(int round=0; round < WARP_SIZE; round++) {

            j_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

            if(j_idx < i_idx && i_idx < n_atoms && j_idx < n_atoms) {

                RealType dx = x0 - x1;
                RealType dy = y0 - y1;
                RealType dz = z0 - z1;
                RealType d2x = dx*dx;
                RealType d2y = dy*dy;
                RealType d2z = dz*dz;

                RealType sij = scale_matrix[i_idx*n_atoms + j_idx];
                RealType d2ij = d2x + d2y + d2z;
                RealType dij = gpuSqrt(d2ij);
                RealType d4ij = d2ij*d2ij;
                RealType d6ij = d4ij*d2ij;
                RealType d8ij = d4ij*d4ij;
                RealType d16ij = d8ij*d8ij;
                RealType inv_d16ij = 1.0/d16ij;

                RealType eps = gpuSqrt(eps0*eps1);
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

                energy += sij*4*eps*(sig6/d6ij-1.0)*sig6/d6ij;

                // (ytz): 99 % sure this loses precision so we need to refactor
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

                if(!inference) {

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

            x1 = __shfl_sync(0xffffffff, x1, srcLane);
            y1 = __shfl_sync(0xffffffff, y1, srcLane);
            z1 = __shfl_sync(0xffffffff, z1, srcLane);
            sig1 = __shfl_sync(0xffffffff, sig1, srcLane);
            eps1 = __shfl_sync(0xffffffff, eps1, srcLane);
            sig1_g_idx = __shfl_sync(0xffffffff, sig1_g_idx, srcLane);
            eps1_g_idx = __shfl_sync(0xffffffff, eps1_g_idx, srcLane);

            shfl_grad_dx = __shfl_sync(0xffffffff, shfl_grad_dx, srcLane);
            shfl_grad_dy = __shfl_sync(0xffffffff, shfl_grad_dy, srcLane);
            shfl_grad_dz = __shfl_sync(0xffffffff, shfl_grad_dz, srcLane);

            if(!inference) {
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
            }

            j_idx += 1;

        }

        int target_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

        if(target_idx < n_atoms) {

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
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, n_atoms, 0, 0), shfl_hess_xx);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, n_atoms, 1, 0), shfl_hess_yx);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, n_atoms, 1, 1), shfl_hess_yy);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, n_atoms, 2, 0), shfl_hess_zx);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, n_atoms, 2, 1), shfl_hess_zy);
                atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(target_idx, target_idx, n_atoms, 2, 2), shfl_hess_zz);                
            }

        }

    }

    if(i_idx < n_atoms) {

        if(E) {
            atomicAdd(energy_out, energy);   
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
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, n_atoms, 0, 0), hess_xx);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, n_atoms, 1, 0), hess_yx);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, n_atoms, 1, 1), hess_yy);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, n_atoms, 2, 0), hess_zx);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, n_atoms, 2, 1), hess_zy);
            atomicAdd(d2E_dx2 + conf_idx*N*3*N*3 + HESS_IDX(i_idx, i_idx, n_atoms, 2, 2), hess_zz);            
        }
    }



    // Surreal<RealType> x0(0, 0);
    // Surreal<RealType> y0(0, 0);
    // Surreal<RealType> z0(0, 0);
    // Surreal<RealType> sig0(0, 0);
    // Surreal<RealType> eps0(0, 0);

    // Surreal<RealType> grad_dx(0, 0);
    // Surreal<RealType> grad_dy(0, 0);
    // Surreal<RealType> grad_dz(0, 0);

    // Surreal<RealType> energy(0, 0);

    // const auto dp_idx = blockIdx.y;
    // const auto num_dp = gridDim.y;
    // const RealType step_size = 1e-7;
    // const bool compute_dp = (dp_idxs != nullptr);

    // if(i_idx < num_atoms) {
    //     RealType rx0 = coords[conf_idx*num_atoms*3 + i_idx*3 + 0];
    //     RealType ry0 = coords[conf_idx*num_atoms*3 + i_idx*3 + 1];
    //     RealType rz0 = coords[conf_idx*num_atoms*3 + i_idx*3 + 2];
    //     RealType rsig0 = params[param_idxs[i_idx*2+0]];
    //     RealType reps0 = params[param_idxs[i_idx*2+1]];

    //     RealType ix0 = 0.0;
    //     RealType iy0 = 0.0;
    //     RealType iz0 = 0.0;
    //     RealType isig0 = 0.0;
    //     RealType ieps0 = 0.0;

    //     if(compute_dp) {
    //         if(dx_dp != nullptr) {
    //             ix0 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3+0]*step_size;
    //             iy0 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3+1]*step_size;
    //             iz0 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3+2]*step_size;  
    //         }

    //         if(param_idxs[i_idx*2+0] == dp_idxs[dp_idx]) {
    //             isig0 = step_size;
    //         }

    //         if(param_idxs[i_idx*2+1] == dp_idxs[dp_idx]) {
    //             ieps0 = step_size;
    //         }
    //     }

    //     x0 = Surreal<RealType>(rx0, ix0);
    //     y0 = Surreal<RealType>(ry0, iy0);
    //     z0 = Surreal<RealType>(rz0, iz0);
    //     sig0 = Surreal<RealType>(rsig0, isig0);
    //     eps0 = Surreal<RealType>(reps0, ieps0);
    // }

    // int num_y_tiles = blockIdx.x + 1;

    // // (ytz): compute the upper right triangular contribution only.
    // for(int tile_y_idx = 0; tile_y_idx < num_y_tiles; tile_y_idx++) {

    //     Surreal<RealType> x1(0, 0);
    //     Surreal<RealType> y1(0, 0);
    //     Surreal<RealType> z1(0, 0);
    //     Surreal<RealType> sig1(0, 0);
    //     Surreal<RealType> eps1(0, 0);

    //     Surreal<RealType> shfl_grad_dx(0, 0);
    //     Surreal<RealType> shfl_grad_dy(0, 0);
    //     Surreal<RealType> shfl_grad_dz(0, 0);

    //     int j_idx = tile_y_idx*WARP_SIZE + threadIdx.x;

    //     if(j_idx < num_atoms) {
    //         RealType rx1 = coords[conf_idx*num_atoms*3 + j_idx*3 + 0];
    //         RealType ry1 = coords[conf_idx*num_atoms*3 + j_idx*3 + 1];
    //         RealType rz1 = coords[conf_idx*num_atoms*3 + j_idx*3 + 2];
    //         RealType rsig1 = params[param_idxs[j_idx*2+0]];
    //         RealType reps1 = params[param_idxs[j_idx*2+1]];

    //         RealType ix1 = 0.0;
    //         RealType iy1 = 0.0;
    //         RealType iz1 = 0.0;
    //         RealType isig1 = 0.0;
    //         RealType ieps1 = 0.0;

    //         if(compute_dp) {
    //             if(dx_dp != nullptr) {
    //                 ix1 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+j_idx*3+0]*step_size;
    //                 iy1 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+j_idx*3+1]*step_size;
    //                 iz1 = dx_dp[conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+j_idx*3+2]*step_size;  
    //             }

    //             if(param_idxs[j_idx*2+0] == dp_idxs[dp_idx]) {
    //                 isig1 = step_size;
    //             }

    //             if(param_idxs[j_idx*2+1] == dp_idxs[dp_idx]) {
    //                 ieps1 = step_size;
    //             }
    //         }

    //         x1 = Surreal<RealType>(rx1, ix1);
    //         y1 = Surreal<RealType>(ry1, iy1);
    //         z1 = Surreal<RealType>(rz1, iz1);
    //         sig1 = Surreal<RealType>(rsig1, isig1);
    //         eps1 = Surreal<RealType>(reps1, ieps1);
    //     }

    //     for(int round=0; round < WARP_SIZE; round++) {

    //         j_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

    //         if(j_idx < i_idx && i_idx < num_atoms && j_idx < num_atoms) {

    //             Surreal<RealType> dx = x0 - x1;
    //             Surreal<RealType> dy = y0 - y1;
    //             Surreal<RealType> dz = z0 - z1;
    //             Surreal<RealType> d2x = dx*dx;
    //             Surreal<RealType> d2y = dy*dy;
    //             Surreal<RealType> d2z = dz*dz;

    //             Surreal<RealType> sij = scale_matrix[i_idx*num_atoms + j_idx];
    //             Surreal<RealType> d2ij = d2x + d2y + d2z;
    //             Surreal<RealType> dij = sqrt(d2ij);
    //             Surreal<RealType> d4ij = d2ij*d2ij;
    //             Surreal<RealType> d6ij = d4ij*d2ij;
    //             Surreal<RealType> d8ij = d4ij*d4ij;
    //             Surreal<RealType> d16ij = d8ij*d8ij;
    //             Surreal<RealType> inv_d16ij = 1.0/d16ij;

    //             Surreal<RealType> eps = sqrt(eps0*eps1);
    //             Surreal<RealType> sig = (sig0 + sig1)/2;

    //             Surreal<RealType> sig2 = sig*sig;
    //             Surreal<RealType> sig3 = sig2*sig;
    //             Surreal<RealType> sig5 = sig3*sig2;
    //             Surreal<RealType> sig6 = sig3*sig3;
    //             Surreal<RealType> sig12 = sig6*sig6;

    //             Surreal<RealType> rij = d2ij;
    //             Surreal<RealType> rij3 = d6ij;
    //             Surreal<RealType> rij4 = d8ij;
    //             Surreal<RealType> rij7 = rij4 * rij3;

    //             // energy += sij*4*eps*(sig6/d6ij-1.0)*sig6/d6ij;
    //             energy += 0;

    //             // loss of precision?
    //             Surreal<RealType> sig1rij1 = sig/rij;
    //             Surreal<RealType> sig3rij3 = sig1rij1*sig1rij1*sig1rij1;
    //             Surreal<RealType> sig6rij3 = sig3*sig3rij3;
    //             Surreal<RealType> sig4rij4 = sig3rij3*sig1rij1;
    //             Surreal<RealType> sig5rij4 = sig*sig4rij4;
    //             Surreal<RealType> sig6rij4 = sig*sig5rij4;

    //             Surreal<RealType> sig12rij7 = sig12/rij7;

    //             Surreal<RealType> dEdx = 24*eps*dx*(sig12rij7*2 - sig6rij4);
    //             Surreal<RealType> dEdy = 24*eps*dy*(sig12rij7*2 - sig6rij4);
    //             Surreal<RealType> dEdz = 24*eps*dz*(sig12rij7*2 - sig6rij4);

    //             // TODO: multiply by sij when atomic adding (But this is trivial relative to cost
    //             // of hessians.
    //             grad_dx -= sij*dEdx;
    //             grad_dy -= sij*dEdy;
    //             grad_dz -= sij*dEdz;

    //             shfl_grad_dx += sij*dEdx;
    //             shfl_grad_dy += sij*dEdy;
    //             shfl_grad_dz += sij*dEdz;
    //         }

    //         int srcLane = (threadIdx.x + 1) % WARP_SIZE;

    //         x1 = __shfl_sync(0xffffffff, x1, srcLane);
    //         y1 = __shfl_sync(0xffffffff, y1, srcLane);
    //         z1 = __shfl_sync(0xffffffff, z1, srcLane);
    //         sig1 = __shfl_sync(0xffffffff, sig1, srcLane);
    //         eps1 = __shfl_sync(0xffffffff, eps1, srcLane);

    //         shfl_grad_dx = __shfl_sync(0xffffffff, shfl_grad_dx, srcLane);
    //         shfl_grad_dy = __shfl_sync(0xffffffff, shfl_grad_dy, srcLane);
    //         shfl_grad_dz = __shfl_sync(0xffffffff, shfl_grad_dz, srcLane);

    //         j_idx += 1;

    //     }

    //     int target_idx = tile_y_idx*WARP_SIZE + j_idx % WARP_SIZE;

    //     if(target_idx < num_atoms) {

    //         if(blockIdx.y == 0 && dE_dx != nullptr) {
    //             atomicAdd(dE_dx + conf_idx*num_atoms*3 + target_idx*3 + 0, shfl_grad_dx.real);
    //             atomicAdd(dE_dx + conf_idx*num_atoms*3 + target_idx*3 + 1, shfl_grad_dy.real);
    //             atomicAdd(dE_dx + conf_idx*num_atoms*3 + target_idx*3 + 2, shfl_grad_dz.real);
    //         }

    //         if(compute_dp && d2E_dxdp != nullptr) {
    //             atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+target_idx*3 + 0, shfl_grad_dx.imag/step_size);
    //             atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+target_idx*3 + 1, shfl_grad_dy.imag/step_size);
    //             atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+target_idx*3 + 2, shfl_grad_dz.imag/step_size);
    //         }
    //     }
    // }


    // if(blockIdx.y == 0) {
    //     atomicAdd(E + conf_idx, energy.real);
    // }

    // if(compute_dp && dE_dp != nullptr) {
    //     atomicAdd(dE_dp+conf_idx*num_dp+dp_idx, energy.imag/step_size);
    // }

    // if(i_idx < num_atoms) {
    //     if(blockIdx.y == 0 && dE_dx != nullptr) {
    //         atomicAdd(dE_dx + conf_idx*num_atoms*3 + i_idx*3 + 0, grad_dx.real);
    //         atomicAdd(dE_dx + conf_idx*num_atoms*3 + i_idx*3 + 1, grad_dy.real);
    //         atomicAdd(dE_dx + conf_idx*num_atoms*3 + i_idx*3 + 2, grad_dz.real);
    //     }

    //     if(compute_dp && d2E_dxdp != nullptr) {
    //         atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3 + 0, grad_dx.imag/step_size);
    //         atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3 + 1, grad_dy.imag/step_size);
    //         atomicAdd(d2E_dxdp+conf_idx*num_dp*num_atoms*3+dp_idx*num_atoms*3+i_idx*3 + 2, grad_dz.imag/step_size);
    //     }
    // }

}