#include <stdexcept>
#include <iostream> 
#include "../fixed_point.hpp"
#include "surreal.cuh"
#include "kernel_utils.cuh"

#define WARPSIZE 32

// since we need to do a full O(N^2) computing and we don't need to broadcast the forces,
// this should just be extremely efficient already
template<typename RealType, int D>
__global__ void compute_born_radii_gpu(
    const int N,
    const double* coords,
    const double* params,
    const int* atomic_radii_idxs,
    const int* scale_factor_idxs,
    const double dielectric_offset,
    const double alpha_obc,
    const double beta_obc,
    const double gamma_obc,
    const double cutoff,
    RealType* born_radii,
    RealType* obc_chain,
    RealType* obc_chain_ri) {

    int atom_i_idx = blockIdx.x*32 + threadIdx.x;

    if(atom_i_idx >= N) {
        return;
    }
    RealType ci[D];
    for(int d=0; d < D; d++) {
        ci[d] = coords[atom_i_idx*D+d];
    }
    int radii_param_idx_i = atom_i_idx < N ? atomic_radii_idxs[atom_i_idx] : 0;

    RealType radiusI = atom_i_idx < N ? params[radii_param_idx_i] : 0;
    RealType offsetRadiusI   = radiusI - dielectric_offset;
    RealType radiusIInverse  = 1.0/offsetRadiusI;

    RealType sum = 0;
 
    for(int atom_j_idx = 0; atom_j_idx < N; atom_j_idx++) {

        int radii_param_idx_j = atom_j_idx < N ? atomic_radii_idxs[atom_j_idx] : 0;
        int scale_param_idx_j = atom_j_idx < N ? scale_factor_idxs[atom_j_idx] : 0;

        RealType radiusJ = atom_j_idx < N ? params[radii_param_idx_j] : 0;
        RealType scaleFactorJ = atom_j_idx < N ? params[scale_param_idx_j] : 0;

        RealType offsetRadiusJ   = radiusJ - dielectric_offset; 
        RealType scaledRadiusJ   = offsetRadiusJ*scaleFactorJ;

        RealType r = 0;
        for(int d=0; d < D; d++) {
            RealType dx = ci[d] - coords[atom_j_idx*D+d];
            r += dx*dx;
        }
        r = sqrt(r);

        RealType rScaledRadiusJ  = r + scaledRadiusJ;
        RealType rSubScaledRadiusJ =  r - scaledRadiusJ;

        // printf("%d %d %d %d\n",atom_j_idx != atom_i_idx, r < cutoff, atom_j_idx < N, atom_i_idx < N);

        if(atom_j_idx != atom_i_idx && r < cutoff) {

            if (offsetRadiusI < rScaledRadiusJ) {
                RealType rInverse = 1.0/r;

                RealType l_ij = 0;
                if(offsetRadiusI > abs(rSubScaledRadiusJ)) {
                  l_ij = offsetRadiusI;
                } else {
                  l_ij = abs(rSubScaledRadiusJ);
                }

                l_ij     = 1.0/l_ij;

                RealType u_ij     = 1.0/rScaledRadiusJ;

                RealType l_ij2    = l_ij*l_ij;
                RealType u_ij2    = u_ij*u_ij;

                RealType ratio    = log((u_ij/l_ij));
                RealType term     = l_ij - u_ij + 0.25*r*(u_ij2 - l_ij2)  + (0.5*rInverse*ratio) + (0.25*scaledRadiusJ*scaledRadiusJ*rInverse)*(l_ij2 - u_ij2);

                // this case (atom i completely inside atom j) is not considered in the original paper
                // Jay Ponder and the authors of Tinker recognized this and
                // worked out the details
                if (offsetRadiusI < (scaledRadiusJ - r)) {
                    term += 2.0*(radiusIInverse - l_ij);
                }
                sum += term;
            }
        }
    }



    sum                *= 0.5*offsetRadiusI;

    RealType sum2       = sum*sum;
    RealType sum3       = sum*sum2;
    RealType tanhSum    = tanh(alpha_obc*sum - beta_obc*sum2 + gamma_obc*sum3);

    if(atom_i_idx < N) {

        born_radii[atom_i_idx]      = 1.0/(1.0/offsetRadiusI - tanhSum/radiusI); 

        // dRi/dPsi
        obc_chain[atom_i_idx]       = (alpha_obc - 2.0*beta_obc*sum + 3.0*gamma_obc*sum2); // !@#$ why did you move it here!
        obc_chain[atom_i_idx]       = (1.0 - tanhSum*tanhSum)*obc_chain[atom_i_idx]/radiusI; // this takes care of the radiusI prefactor
        obc_chain[atom_i_idx]      *= born_radii[atom_i_idx]*born_radii[atom_i_idx];

        // dRi/dri
        obc_chain_ri[atom_i_idx]    = 1.0/(offsetRadiusI*offsetRadiusI) - tanhSum/(radiusI*radiusI);
        obc_chain_ri[atom_i_idx]   *= born_radii[atom_i_idx]*born_radii[atom_i_idx];

    }

}



template<typename RealType, int D>
__global__ void compute_born_radii_gpu_jvp(
    const int N,
    const Surreal<RealType>* coords,
    const double* params,
    const int* atomic_radii_idxs,
    const int* scale_factor_idxs,
    const double dielectric_offset,
    const double alpha_obc,
    const double beta_obc,
    const double gamma_obc,
    const double cutoff,
    Surreal<RealType>* born_radii,
    Surreal<RealType>* obc_chain,
    Surreal<RealType>* obc_chain_ri) {

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;

    if(atom_i_idx >= N) {
        return;
    }
    Surreal<RealType> ci[D] = {Surreal<RealType>(0, 0)};
    for(int d=0; d < D; d++) {
        ci[d] = coords[atom_i_idx*D+d];
    }
    int radii_param_idx_i = atom_i_idx < N ? atomic_radii_idxs[atom_i_idx] : 0;
    // int scale_param_idx_i = atom_i_idx < N ? scale_factor_idxs[atom_i_idx] : 0;

    RealType radiusI = atom_i_idx < N ? params[radii_param_idx_i] : 0;
    // RealType scaleFactorI = atom_i_idx < N ? params[scale_param_idx_i] : 0;
    RealType offsetRadiusI   = radiusI - dielectric_offset;
    RealType radiusIInverse  = 1.0/offsetRadiusI;

    Surreal<RealType> sum(0, 0);
 
    for(int atom_j_idx = 0; atom_j_idx < N; atom_j_idx++) {

        int radii_param_idx_j = atom_j_idx < N ? atomic_radii_idxs[atom_j_idx] : 0;
        int scale_param_idx_j = atom_j_idx < N ? scale_factor_idxs[atom_j_idx] : 0;

        RealType radiusJ = atom_j_idx < N ? params[radii_param_idx_j] : 0;
        RealType scaleFactorJ = atom_j_idx < N ? params[scale_param_idx_j] : 0;
        RealType offsetRadiusJ   = radiusJ - dielectric_offset; 
        RealType scaledRadiusJ   = offsetRadiusJ*scaleFactorJ;

        Surreal<RealType> r(0, 0);
        for(int d=0; d < D; d++) {
            Surreal<RealType> dx = ci[d] - coords[atom_j_idx*D+d];
            r += dx*dx;
        }
        r = sqrt(r);

        Surreal<RealType> rScaledRadiusJ  = r + scaledRadiusJ;
        Surreal<RealType> rSubScaledRadiusJ =  r - scaledRadiusJ;

        // printf("%d %d %d %d\n",atom_j_idx != atom_i_idx, r < cutoff, atom_j_idx < N, atom_i_idx < N);

        if(atom_j_idx != atom_i_idx && r.real < cutoff) {

            if (offsetRadiusI < rScaledRadiusJ.real) {
                Surreal<RealType> rInverse = 1.0/r;

                Surreal<RealType> l_ij(0,0);
                if(offsetRadiusI > abs(rSubScaledRadiusJ).real) {
                  l_ij.real = offsetRadiusI;
                  l_ij.imag = 0;
                } else {
                  l_ij = abs(rSubScaledRadiusJ);
                }

                l_ij     = 1.0/l_ij;

                Surreal<RealType> u_ij     = 1.0/rScaledRadiusJ;

                Surreal<RealType> l_ij2    = l_ij*l_ij;
                Surreal<RealType> u_ij2    = u_ij*u_ij;

                Surreal<RealType> ratio    = log((u_ij/l_ij));
                Surreal<RealType> term     = l_ij - u_ij + 0.25*r*(u_ij2 - l_ij2)  + (0.5*rInverse*ratio) + (0.25*scaledRadiusJ*scaledRadiusJ*rInverse)*(l_ij2 - u_ij2);

                // this case (atom i completely inside atom j) is not considered in the original paper
                // Jay Ponder and the authors of Tinker recognized this and
                // worked out the details
                if (offsetRadiusI < (scaledRadiusJ - r).real) {
                    term += 2.0*(radiusIInverse - l_ij);
                }
                sum += term;
            }
        }
    }



    sum                *= 0.5*offsetRadiusI;

    Surreal<RealType> sum2       = sum*sum;
    Surreal<RealType> sum3       = sum*sum2;
    Surreal<RealType> tanhSum    = tanh(alpha_obc*sum - beta_obc*sum2 + gamma_obc*sum3);

    if(atom_i_idx < N) {

        born_radii[atom_i_idx]      = 1.0/(1.0/offsetRadiusI - tanhSum/radiusI); 

        // dRi/dPsi
        obc_chain[atom_i_idx]       = (alpha_obc - 2.0*beta_obc*sum + 3.0*gamma_obc*sum2); // !@#$ why did you move it here!
        obc_chain[atom_i_idx]       = (1.0 - tanhSum*tanhSum)*obc_chain[atom_i_idx]/radiusI; // this takes care of the radiusI prefactor
        obc_chain[atom_i_idx]      *= born_radii[atom_i_idx]*born_radii[atom_i_idx];

        // dRi/dri
        obc_chain_ri[atom_i_idx]    = 1.0/(offsetRadiusI*offsetRadiusI) - tanhSum/(radiusI*radiusI);
        obc_chain_ri[atom_i_idx]   *= born_radii[atom_i_idx]*born_radii[atom_i_idx];

    }

}


template <typename RealType, int D>
void __global__ k_compute_born_first_loop_gpu(
    const int N,
    const double* coords,
    const double* params,
    const int* charge_param_idxs,
    const double* born_radii,
    const double prefactor,
    const double cutoff,
    double *bornForces,
    double *out_forces,
    double *dU_dp) {

    if(blockIdx.y > blockIdx.x) {
        return;
    }

    // RealType block_d2ij = 0; 
    // for(int d=0; d < D; d++) {
    //     RealType block_row_ctr = block_bounds_ctr[blockIdx.x*D+d];
    //     RealType block_col_ctr = block_bounds_ctr[blockIdx.y*D+d];
    //     RealType block_row_ext = block_bounds_ext[blockIdx.x*D+d];
    //     RealType block_col_ext = block_bounds_ext[blockIdx.y*D+d];
    //     RealType dx = max(0.0, fabs(block_row_ctr-block_col_ctr) - (block_row_ext+block_col_ext));
    //     block_d2ij += dx*dx;
    // }

    // if(block_d2ij > cutoff*cutoff) {
    //     return;
    // }

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    RealType ci[D];
    RealType gi[D] = {0};
    for(int d=0; d < D; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*D+d] : 0;
    }
    int charge_param_idx_i = atom_i_idx < N ? charge_param_idxs[atom_i_idx] : 0;
    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType born_radii_i = atom_i_idx < N ? born_radii[atom_i_idx] : 0;

    RealType dE_dqi_accum = 0;
    RealType born_force_i_accum = 0;

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    RealType cj[D];
    RealType gj[D] = {0};
    for(int d=0; d < D; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*D+d] : 0;
    }
    int charge_param_idx_j = atom_j_idx < N ? charge_param_idxs[atom_j_idx] : 0;
    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    RealType born_radii_j = atom_j_idx < N ? born_radii[atom_j_idx] : 0;
    RealType dE_dqj_accum = 0;
    RealType born_force_j_accum = 0;

    // In inference mode, we don't care about gradients with respect to parameters.
    for(int round = 0; round < 32; round++) {

        RealType dxs[D];
        RealType r2 = 0;
        for(int d=0; d < D; d++) {
            dxs[d] = ci[d] - cj[d];
            r2 += dxs[d]*dxs[d];
        }
        RealType r = sqrt(r2);

        if(atom_j_idx <= atom_i_idx && r < cutoff && atom_j_idx < N && atom_i_idx < N) {

            RealType alpha2_ij          = born_radii_i*born_radii_j;
            RealType D_ij               = r2/(4.0*alpha2_ij);
            RealType expTerm            = exp(-D_ij);
            RealType denominator2       = r2 + alpha2_ij*expTerm; 
            RealType denominator        = sqrt(denominator2);
            RealType Gpol               = (prefactor*qi*qj)/denominator; 
            RealType dGpol_dr           = -Gpol*(1.0 - 0.25*expTerm)/denominator2;  
            RealType dGpol_dalpha2_ij   = -0.5*Gpol*expTerm*(1.0 + D_ij)/denominator2;

            RealType energy = Gpol;

            RealType dE_dqi = prefactor*qj/denominator;
            RealType dE_dqj = prefactor*qi/denominator;

            if (atom_i_idx != atom_j_idx) {

                // TBD: determine what we should do with cutoff
                // energy -= qi*partialCharges[atom_j_idx]/cutoff;
                // bornForces[atom_j_idx]        += dGpol_dalpha2_ij*born_radii[atom_i_idx];
                born_force_j_accum += dGpol_dalpha2_ij*born_radii_i;

                for(int d=0; d < D; d++) {
                    gi[d] += dxs[d]*dGpol_dr;
                    gj[d] -= dxs[d]*dGpol_dr;
                }

            } else {
                dE_dqi *= 0.5;
                dE_dqj *= 0.5;
                energy *= 0.5;
            }

            dE_dqi_accum += dE_dqi;
            dE_dqj_accum += dE_dqj;
            born_force_i_accum += dGpol_dalpha2_ij*born_radii_j;
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE;
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        born_radii_j = __shfl_sync(0xffffffff, born_radii_j, srcLane);
        dE_dqj_accum = __shfl_sync(0xffffffff, dE_dqj_accum, srcLane);
        born_force_j_accum = __shfl_sync(0xffffffff, born_force_j_accum, srcLane);
        for(size_t d=0; d < D; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
    }

    for(int d=0; d < D; d++) {
        if(atom_i_idx < N) {
            atomicAdd(out_forces + atom_i_idx*D + d, gi[d]);
        }
        if(atom_j_idx < N) {
            atomicAdd(out_forces + atom_j_idx*D + d, gj[d]);
        }
    }

    if(atom_i_idx < N) {
        atomicAdd(bornForces + atom_i_idx, born_force_i_accum);
        atomicAdd(dU_dp + charge_param_idx_i, dE_dqi_accum);
    }

    if(atom_j_idx < N) {
        atomicAdd(bornForces + atom_j_idx, born_force_j_accum);
        atomicAdd(dU_dp + charge_param_idx_j, dE_dqj_accum);
    }



}
