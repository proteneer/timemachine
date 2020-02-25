#include <stdexcept>
#include <iostream> 
#include "../fixed_point.hpp"
#include "surreal.cuh"
#include "kernel_utils.cuh"

#define WARPSIZE 32

#include <cooperative_groups.h>

// since we need to do a full O(N^2) computing and we don't need to broadcast the forces,
// this should just be extremely efficient already


// nope, still need to parallelize out
template<typename RealType, int D>
__global__ void k_compute_born_radii_gpu(
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
    double* born_radii,
    double* obc_chain) {
    // double* obc_chain_ri) {

    int atom_i_idx = blockIdx.x*32 + threadIdx.x;

    if(threadIdx.x == 0) {
        printf("%d %d\n", blockIdx.x, blockIdx.y);
    }
    // if(atom_i_idx >= N) {
    //     return;
    // }

    RealType ci[D];
    for(int d=0; d < D; d++) {
        ci[d] = coords[atom_i_idx*D+d];
    }
    int radii_param_idx_i = atom_i_idx < N ? atomic_radii_idxs[atom_i_idx] : 0;

    RealType radiusI = atom_i_idx < N ? params[radii_param_idx_i] : 0;
    RealType offsetRadiusI   = radiusI - dielectric_offset;
    RealType radiusIInverse  = 1/offsetRadiusI;

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;

    // *always* accumulate in 64 bit.
    double sum = 0;
 
    RealType cj[D];
    for(int d=0; d < D; d++) {
        cj[d] = coords[atom_j_idx*D+d];
    }

    int radii_param_idx_j = atom_j_idx < N ? atomic_radii_idxs[atom_j_idx] : 0;
    int scale_param_idx_j = atom_j_idx < N ? scale_factor_idxs[atom_j_idx] : 0;

    RealType radiusJ = atom_j_idx < N ? params[radii_param_idx_j] : 0;
    RealType scaleFactorJ = atom_j_idx < N ? params[scale_param_idx_j] : 0;

    RealType offsetRadiusJ   = radiusJ - dielectric_offset; 
    RealType scaledRadiusJ   = offsetRadiusJ*scaleFactorJ;

    // for(int atom_j_idx = 0; atom_j_idx < N; atom_j_idx++) {
    for(int round = 0; round < 32; round++) {

        RealType dxs[D];
        for(int d=0; d < D; d++) {
            dxs[d] = ci[d] - cj[d];
        }

        RealType r = fast_vec_norm<RealType, D>(dxs);
        RealType rInverse = fast_vec_rnorm<RealType, D>(dxs);
        RealType rScaledRadiusJ  = r + scaledRadiusJ;
        RealType rSubScaledRadiusJ =  r - scaledRadiusJ;

        if(atom_j_idx != atom_i_idx && r < cutoff) {

            if (offsetRadiusI < rScaledRadiusJ) {

                RealType l_ij = 0;
                if(offsetRadiusI > abs(rSubScaledRadiusJ)) {
                  l_ij = offsetRadiusI;
                } else {
                  l_ij = abs(rSubScaledRadiusJ);
                }

                l_ij = 1/l_ij;

                // RealType inv_uij = rScaledRadiusJ;
                RealType u_ij     = 1/rScaledRadiusJ;

                RealType l_ij2    = l_ij*l_ij;
                RealType u_ij2    = u_ij*u_ij;

                RealType ratio    = log(u_ij/l_ij);
                // RealType term     = l_ij - u_ij + 0.25*r*(u_ij2 - l_ij2)  + (0.5*rInverse*ratio) + (0.25*scaledRadiusJ*scaledRadiusJ*rInverse)*(l_ij2 - u_ij2);
                RealType term     = l_ij - u_ij + r*(u_ij2 - l_ij2)/4 + scaledRadiusJ*scaledRadiusJ*rInverse*(l_ij2 - u_ij2)/4 + rInverse*ratio/2;

                // this case (atom i completely inside atom j) is not considered in the original paper
                // Jay Ponder and the authors of Tinker recognized this and
                // worked out the details
                if (offsetRadiusI < (scaledRadiusJ - r)) {
                    term += 2*(radiusIInverse - l_ij);
                }
                sum += term;
            }
        }


        const int srcLane = (threadIdx.x + 1) % WARPSIZE;
        scaledRadiusJ = __shfl_sync(0xffffffff, scaledRadiusJ, srcLane);
        for(int d=0; d < D; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
        }

    }

    sum *= offsetRadiusI/2;

    double sum2       = sum*sum;
    double sum3       = sum*sum2;
    double inner      = alpha_obc*sum - beta_obc*sum2 + gamma_obc*sum3;
    double tanhSum    = tanh(inner);
    // double coshSum    = cosh(inner);
    // double sechSum    = 1/coshSum;

    if(atom_i_idx < N) {

        // born_radii[atom_i_idx]      = 1.0/(1.0/offsetRadiusI - tanhSum/radiusI);
        auto br = offsetRadiusI*radiusI/(radiusI - offsetRadiusI*tanhSum);
        atomicAdd(born_radii + atom_i_idx, br);
        atomicAdd(obc_chain + atom_i_idx, (1 - tanhSum*tanhSum)*(alpha_obc - 2*beta_obc*sum + 3*gamma_obc*sum2)/radiusI);
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
    unsigned long long *bornForces,
    unsigned long long *out_forces) {

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

    // RealType dE_dqi_accum = 0;
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
    // RealType dE_dqj_accum = 0;
    RealType born_force_j_accum = 0;

    // In inference mode, we don't care about gradients with respect to parameters.
    for(int round = 0; round < 32; round++) {

        RealType dxs[D];
        RealType r2 = 0;
        for(int d=0; d < D; d++) {
            RealType dx = ci[d] - cj[d];
            dxs[d] = dx;
            r2 += dx*dx;
        }
        RealType r = fast_vec_norm<RealType, D>(dxs);
        RealType rInverse = fast_vec_rnorm<RealType, D>(dxs);

        if(atom_j_idx <= atom_i_idx && r < cutoff && atom_j_idx < N && atom_i_idx < N) {

            RealType alpha2_ij          = born_radii_i*born_radii_j;
            RealType D_ij               = r2/(4.0*alpha2_ij);
            RealType expTerm            = exp(-D_ij);
            RealType denominator2       = r2 + alpha2_ij*expTerm; 
            RealType denominator        = sqrt(denominator2);
            RealType Gpol               = (prefactor*qi*qj)/denominator; 
            RealType dGpol_dr           = -Gpol*(1.0 - 0.25*expTerm)/denominator2;  
            RealType dGpol_dalpha2_ij   = -0.5*Gpol*expTerm*(1.0 + D_ij)/denominator2;

            // RealType energy = Gpol;
            // RealType dE_dqi = prefactor*qj/denominator;
            // RealType dE_dqj = prefactor*qi/denominator;

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
                // dE_dqi *= 0.5;
                // dE_dqj *= 0.5;
                // energy *= 0.5;
            }

            born_force_i_accum += dGpol_dalpha2_ij*born_radii_j;
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE;
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        born_radii_j = __shfl_sync(0xffffffff, born_radii_j, srcLane);
        born_force_j_accum = __shfl_sync(0xffffffff, born_force_j_accum, srcLane);
        for(size_t d=0; d < D; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
    }

    for(int d=0; d < D; d++) {
        if(atom_i_idx < N) {
            atomicAdd(out_forces + atom_i_idx*D + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));
        }
        if(atom_j_idx < N) {
            atomicAdd(out_forces + atom_j_idx*D + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));
        }
    }

    if(atom_i_idx < N) {
        atomicAdd(bornForces + atom_i_idx, static_cast<unsigned long long>((long long) (born_force_i_accum*FIXED_EXPONENT_BORN_FORCES)));
    }

    if(atom_j_idx < N) {
        atomicAdd(bornForces + atom_j_idx, static_cast<unsigned long long>((long long) (born_force_j_accum*FIXED_EXPONENT_BORN_FORCES)));
    }

}


// this is entirely done in double precision
template <typename RealType, int D>
__global__ void k_reduce_born_forces(
    const int N,
    const double* params,
    const int* atomic_radii_idxs,
    const double* born_radii,
    double* obc_chain,
    // const double* obc_chain_ri,
    const double surface_tension, // surface area factor
    const double probe_radius,
    unsigned long long* bornForces // dU/Ri
    // double* out_dU_dp
) {

    int atomI =  blockIdx.x*32 + threadIdx.x;
    if(atomI >= N) {
        return;
    }

    // double radii_derivs = 0;
    double born_force_i = static_cast<double>(static_cast<long long>(bornForces[atomI]))/FIXED_EXPONENT_BORN_FORCES;
    double br = born_radii[atomI];
    if (br > 0.0) {
        double atomic_radii = params[atomic_radii_idxs[atomI]];
        double r            = atomic_radii + probe_radius;
        double ratio6       = pow(atomic_radii/born_radii[atomI], 6.0);
        double saTerm       = surface_tension*r*r*ratio6;
        born_force_i  -= 6.0*saTerm/born_radii[atomI];
    }
    // radii_derivs += born_force_i * obc_chain_ri[atomI];
    // out_dU_dp[atomic_radii_idxs[atomI]] += radii_derivs;

    obc_chain[atomI] *= br*br;
    born_force_i *= obc_chain[atomI];
    bornForces[atomI] = static_cast<unsigned long long>((long long) ( born_force_i*FIXED_EXPONENT_BORN_FORCES));

}



template <typename RealType, int D>
__global__ void k_compute_born_energy_and_forces(
    const int N,
    const double* coords,
    const double* params,
    const int* atomic_radii_idxs,
    const int* scale_factor_idxs,
    const double* born_radii,
    const double* obc_chain,
    // const double* obc_chain_ri,
    const double dielectric_offset,
    const double cutoff,
    const unsigned long long* bornForces,
    unsigned long long* out_forces) {

    // we always do the full interaction matrix due to non-symmetry

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    RealType ci[D];
    RealType dPsi_dx_i[D] = {0};
    for(int d=0; d < D; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*D+d] : 0;
    }

    int atomic_radii_idx_i = atom_i_idx < N ? atomic_radii_idxs[atom_i_idx] : 0;
    RealType radiusI = atom_i_idx < N ? params[atomic_radii_idx_i] : 0;
    RealType born_force_i = atom_i_idx < N ? static_cast<RealType>(static_cast<long long>(bornForces[atom_i_idx]))/FIXED_EXPONENT_BORN_FORCES : 0;
    // RealType born_radii_i = atom_i_idx < N ? born_radii[atom_i_idx] : 0;
    // RealType dPsi_dri = 0;

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    RealType cj[D];
    RealType dPsi_dx_j[D] = {0};
    for(int d=0; d < D; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*D+d] : 0;
    }
    int atomic_radii_idx_j = atom_j_idx < N ? atomic_radii_idxs[atom_j_idx] : 0;
    RealType radiusJ = atom_j_idx < N ? params[atomic_radii_idx_j] : 0;

    int scale_factor_idx_j = atom_j_idx < N ? scale_factor_idxs[atom_j_idx] : 0;
    RealType scaleFactorJ = atom_j_idx < N ? params[scale_factor_idx_j] : 0;
    RealType born_radii_j = atom_j_idx < N ? born_radii[atom_j_idx] : 0;
    // RealType dPsi_drj = 0;
    // RealType dPsi_dsj = 0;

    const RealType dielectricOffset = dielectric_offset;
    // const RealType cutoffDistance = cutoff;

    // RealType radiusI        = params[atomic_radii_idxs[atomI]];
    RealType offsetRadiusI  = radiusI - dielectricOffset;
    RealType offsetRadiusI2 = offsetRadiusI*offsetRadiusI;
    RealType offsetRadiusI3 = offsetRadiusI2*offsetRadiusI;

    int atomI = atom_i_idx;
    int atomJ = atom_j_idx;

    for(int round = 0; round < 32; round++) {

        if (atomJ != atomI) {

            RealType dxs[D];
            for(int d=0; d < D; d++) {
                dxs[d] = ci[d] - cj[d];
            }

            RealType r = fast_vec_norm<RealType, D>(dxs);
            RealType rInverse = fast_vec_rnorm<RealType, D>(dxs);
            // radius w/ dielectric offset applied
            // RealType radiusJ            = params[atomic_radii_idxs[atomJ]];
            RealType offsetRadiusJ      = radiusJ - dielectricOffset;
            // RealType offsetRadiusJ2     = offsetRadiusJ*offsetRadiusJ;

            // RealType scaleFactorJ2      = scaleFactorJ*scaleFactorJ;
            // RealType scaleFactorJ3      = scaleFactorJ2*scaleFactorJ;
            RealType scaledRadiusJ      = offsetRadiusJ*scaleFactorJ;
            RealType rScaledRadiusJ     = r + scaledRadiusJ;
            // RealType rScaledRadiusJ2    = rScaledRadiusJ*rScaledRadiusJ;
            // RealType rScaledRadiusJ3    = rScaledRadiusJ2*rScaledRadiusJ;

            if (offsetRadiusI < rScaledRadiusJ) {

                // RealType l_ij          = offsetRadiusI > abs(rSubScaledRadiusJ) ? offsetRadiusI : abs(rSubScaledRadiusJ);
                //        l_ij          = 1.0/l_ij;
                // RealType u_ij          = 1.0/rScaledRadiusJ;
                // RealType l_ij2         = l_ij*l_ij;
                // RealType u_ij2         = u_ij*u_ij; 
                // RealType rInverse      = 1.0/r;
                // RealType r2Inverse     = rInverse*rInverse;
                // RealType t3            = 0.125*(1.0 + scaledRadiusJ2*r2Inverse)*(l_ij2 - u_ij2) + 0.25*log(u_ij/l_ij)*r2Inverse;

                RealType rSubScaledRadiusJ = r - scaledRadiusJ;
                RealType rSubScaledRadiusJ2 = rSubScaledRadiusJ*rSubScaledRadiusJ;
                // RealType rSubScaledRadiusJ3 = rSubScaledRadiusJ2*rSubScaledRadiusJ;

                RealType l_ij = offsetRadiusI > abs(rSubScaledRadiusJ) ? offsetRadiusI : abs(rSubScaledRadiusJ);
                l_ij = 1.0/l_ij;
                RealType l_ij2 = l_ij*l_ij;

                RealType u_ij = 1.0/rScaledRadiusJ;
                RealType u_ij2 = u_ij*u_ij;

                // original expression
                // RealType term = l_ij - u_ij + 0.25*(u_ij2 - l_ij2)*t1 + (0.5*rInverse*ratio);
                RealType dl_dr = offsetRadiusI > abs(rSubScaledRadiusJ) ? 0 : -(l_ij*l_ij)*(rSubScaledRadiusJ/abs(rSubScaledRadiusJ));
                RealType du_dr = -(u_ij*u_ij)*(rScaledRadiusJ/abs(rScaledRadiusJ));

                // ugh so much loss of significance here
                RealType t1 = r - scaledRadiusJ*scaledRadiusJ*rInverse;
                RealType dt1_dr = (1 + scaledRadiusJ*scaledRadiusJ*rInverse*rInverse);
                // RealType dt1_dsj = -(2/r)*(offsetRadiusJ*offsetRadiusJ*scaleFactorJ);

                RealType ratio = log((u_ij/l_ij));

                //          l_ij - u_ij   + 0.25*(u_ij2 - l_ij2)*t1                          + (0.5*rInverse*ratio);
                // RealType du_dr_div_u = -u_ij*u_ij*abs(rScaledRadiusJ);
                // RealType dl_dr_div_l = offsetRadiusI > abs(rSubScaledRadiusJ) ? 0 : -l_ij*l_ij*abs(rSubScaledRadiusJ);

                // RealType du_dr_mul_u = -u_ij*u_ij/abs(rScaledRadiusJ);
                // RealType dl_dr_mul_l = offsetRadiusI > abs(rSubScaledRadiusJ) ? 0 : -l_ij*l_ij/abs(rSubScaledRadiusJ);

                RealType de1 = static_cast<RealType>(dl_dr) - static_cast<RealType>(du_dr);
                // we may need three separate accumulators for precision
                RealType de2 = 0.5*(u_ij*du_dr - l_ij*dl_dr)*t1;
                RealType de3 = 0.25*(u_ij*u_ij - l_ij*l_ij)*dt1_dr;
                RealType de4 = -0.5*rInverse*(rInverse*ratio - (du_dr/u_ij - dl_dr/l_ij));
                RealType de = de1 + de2 + de3 + de4;

                if(offsetRadiusI > abs(rSubScaledRadiusJ)) {
                    if(offsetRadiusI < (scaledRadiusJ - r)) {
                        de += 0;
                    }
                } else {
                    if(offsetRadiusI < (scaledRadiusJ - r)) {
                        de += 2.0*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                    }
                }

                de *= 0.5*born_force_i*offsetRadiusI;
                for(int d=0; d < D; d++) {
                    dPsi_dx_i[d] += (dxs[d]/r)*de;
                    dPsi_dx_j[d] -= (dxs[d]/r)*de;
                }
            }
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE;
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        atomJ = __shfl_sync(0xffffffff, atomJ, srcLane);
        born_radii_j = __shfl_sync(0xffffffff, born_radii_j, srcLane);
        radiusJ = __shfl_sync(0xffffffff, radiusJ, srcLane);
        scaleFactorJ = __shfl_sync(0xffffffff, scaleFactorJ, srcLane);
        // dPsi_drj = __shfl_sync(0xffffffff, dPsi_drj, srcLane);
        // dPsi_dsj = __shfl_sync(0xffffffff, dPsi_dsj, srcLane);

        atomic_radii_idx_i = __shfl_sync(0xffffffff, atomic_radii_idx_i, srcLane);
        scale_factor_idx_j = __shfl_sync(0xffffffff, scale_factor_idx_j, srcLane);

        for(int d=0; d < D; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            dPsi_dx_j[d] = __shfl_sync(0xffffffff, dPsi_dx_j[d], srcLane);
        }

    }

    for(int d=0; d < D; d++) {
        if(atomI < N) {
            // printf("%d %d dPsi_dx_i[d] %f\n", atomI, d, dPsi_dx_i[d]);
            atomicAdd(out_forces + atomI*D+d,  static_cast<unsigned long long>((long long) (dPsi_dx_i[d]*FIXED_EXPONENT)));
        }
        if(atomJ < N) {
            atomicAdd(out_forces + atomJ*D+d,  static_cast<unsigned long long>((long long) (dPsi_dx_j[d]*FIXED_EXPONENT)));
        }
    }


    // if(atomI < N) {
    //     atomicAdd(out_dU_dp + atomic_radii_idx_i, dPsi_dri);
    // }

    // if(atomJ < N) {
    //     atomicAdd(out_dU_dp + atomic_radii_idx_j, dPsi_drj);
    //     atomicAdd(out_dU_dp + scale_factor_idx_j, dPsi_dsj);
    // }


}


