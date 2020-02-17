#include <stdexcept>
#include <iostream> 
#include "../fixed_point.hpp"
#include "surreal.cuh"
#include "kernel_utils.cuh"

#define WARPSIZE 32

// since we need to do a full O(N^2) computing and we don't need to broadcast the forces,
// this should just be extremely efficient already
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
__global__ void k_compute_born_radii_gpu_jvp(
    const int N,
    const double* coords,
    const double* coords_tangents,
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
        ci[d].real = coords[atom_i_idx*D+d];
        ci[d].imag = coords_tangents[atom_i_idx*D+d];
    }
    int radii_param_idx_i = atom_i_idx < N ? atomic_radii_idxs[atom_i_idx] : 0;

    RealType radiusI = atom_i_idx < N ? params[radii_param_idx_i] : 0;
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
            Surreal<RealType> cjd(coords[atom_j_idx*D+d], coords_tangents[atom_j_idx*D+d]);
            Surreal<RealType> dx = ci[d] - cjd;
            r += dx*dx;
        }
        r = sqrt(r);

        Surreal<RealType> rScaledRadiusJ  = r + scaledRadiusJ;
        Surreal<RealType> rSubScaledRadiusJ =  r - scaledRadiusJ;

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
    unsigned long long *bornForces,
    unsigned long long *out_forces,
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

            // RealType energy = Gpol;

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
                // energy *= 0.5;
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
            atomicAdd(out_forces + atom_i_idx*D + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));
        }
        if(atom_j_idx < N) {
            atomicAdd(out_forces + atom_j_idx*D + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));
        }
    }

    if(atom_i_idx < N) {
        atomicAdd(bornForces + atom_i_idx, static_cast<unsigned long long>((long long) (born_force_i_accum*FIXED_EXPONENT_BORN_FORCES)));
        atomicAdd(dU_dp + charge_param_idx_i, dE_dqi_accum);
    }

    if(atom_j_idx < N) {
        atomicAdd(bornForces + atom_j_idx, static_cast<unsigned long long>((long long) (born_force_j_accum*FIXED_EXPONENT_BORN_FORCES)));
        atomicAdd(dU_dp + charge_param_idx_j, dE_dqj_accum);
    }



}



template <typename RealType, int D>
void __global__ k_compute_born_first_loop_gpu_jvp(
    const int N,
    const double* coords,
    const double* coords_tangents,
    const double* params,
    const int* charge_param_idxs,
    const Surreal<double>* born_radii,
    const double prefactor,
    const double cutoff,
    Surreal<double> *bornForces,
    double *out_HvP,
    double *out_MvP) {

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
    Surreal<RealType> ci[D];
    Surreal<RealType> gi[D];
    for(int d=0; d < D; d++) {
        gi[d] = Surreal<RealType>(0,0);
        ci[d].real = atom_i_idx < N ? coords[atom_i_idx*D+d] : 0;
        ci[d].imag = atom_i_idx < N ? coords_tangents[atom_i_idx*D+d] : 0;
    }
    int charge_param_idx_i = atom_i_idx < N ? charge_param_idxs[atom_i_idx] : 0;
    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    Surreal<RealType> born_radii_i = atom_i_idx < N ? born_radii[atom_i_idx] : Surreal<RealType>(0,0);

    Surreal<RealType> dE_dqi_accum(0, 0);
    Surreal<RealType> born_force_i_accum(0, 0);

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    Surreal<RealType> cj[D];
    Surreal<RealType> gj[D];
    for(int d=0; d < D; d++) {
        gj[d] = Surreal<RealType>(0,0);
        cj[d].real = atom_j_idx < N ? coords[atom_j_idx*D+d] : 0;
        cj[d].imag = atom_j_idx < N ? coords_tangents[atom_j_idx*D+d] : 0;
    }
    int charge_param_idx_j = atom_j_idx < N ? charge_param_idxs[atom_j_idx] : 0;
    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    Surreal<RealType> born_radii_j = atom_j_idx < N ? born_radii[atom_j_idx] : Surreal<RealType>(0,0);
    Surreal<RealType> dE_dqj_accum(0, 0);
    Surreal<RealType> born_force_j_accum(0, 0);

    // In inference mode, we don't care about gradients with respect to parameters.
    for(int round = 0; round < 32; round++) {

        Surreal<RealType> dxs[D];
        Surreal<RealType> r2(0, 0);
        for(int d=0; d < D; d++) {
            dxs[d] = ci[d] - cj[d];
            r2 += dxs[d]*dxs[d];
        }
        Surreal<RealType> r = sqrt(r2);

        if(atom_j_idx <= atom_i_idx && r.real < cutoff && atom_j_idx < N && atom_i_idx < N) {

            Surreal<RealType> alpha2_ij          = born_radii_i*born_radii_j;
            Surreal<RealType> D_ij               = r2/(4.0*alpha2_ij);
            Surreal<RealType> expTerm            = exp(-D_ij);
            Surreal<RealType> denominator2       = r2 + alpha2_ij*expTerm; 
            Surreal<RealType> denominator        = sqrt(denominator2);
            Surreal<RealType> Gpol               = (prefactor*qi*qj)/denominator; 
            Surreal<RealType> dGpol_dr           = -Gpol*(1.0 - 0.25*expTerm)/denominator2;  
            Surreal<RealType> dGpol_dalpha2_ij   = -0.5*Gpol*expTerm*(1.0 + D_ij)/denominator2;

            // Surreal<RealType> energy = Gpol;

            Surreal<RealType> dE_dqi = prefactor*qj/denominator;
            Surreal<RealType> dE_dqj = prefactor*qi/denominator;

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
                // energy *= 0.5;
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
            atomicAdd(out_HvP + atom_i_idx*D + d, gi[d].imag);
        }
        if(atom_j_idx < N) {
            atomicAdd(out_HvP + atom_j_idx*D + d, gj[d].imag);
        }
    }

    if(atom_i_idx < N) {
        atomicAddOffset(bornForces, atom_i_idx, born_force_i_accum); // add both real and imaginary part
        atomicAdd(out_MvP + charge_param_idx_i, dE_dqi_accum.imag); 
    }

    if(atom_j_idx < N) {
        atomicAddOffset(bornForces, atom_j_idx, born_force_j_accum); // add both real and imaginary
        atomicAdd(out_MvP + charge_param_idx_j, dE_dqj_accum.imag);
    }



}


template <typename RealType, int D>
__global__ void k_reduce_born_forces(
    const int N,
    const double* params,
    const int* atomic_radii_idxs,
    const double* born_radii,
    const double* obc_chain,
    const double* obc_chain_ri,
    const double surface_tension, // surface area factor
    const double probe_radius,
    unsigned long long* bornForces, // dU/Ri
    double* out_dU_dp
) {

    int atomI =  blockIdx.x*32 + threadIdx.x;
    if(atomI >= N) {
        return;
    }

    RealType radii_derivs = 0;
    RealType born_force_i = static_cast<RealType>(static_cast<long long>(bornForces[atomI]))/FIXED_EXPONENT_BORN_FORCES;
    if (born_radii[atomI] > 0.0) {
        RealType atomic_radii = params[atomic_radii_idxs[atomI]];
        RealType r            = atomic_radii + probe_radius;
        RealType ratio6       = pow(atomic_radii/born_radii[atomI], 6.0);
        RealType saTerm       = surface_tension*r*r*ratio6;
        born_force_i  -= 6.0*saTerm/born_radii[atomI]; 
        RealType br2 = born_radii[atomI]*born_radii[atomI];
        RealType br4 = br2*br2;
        RealType br6 = br4*br2;
        radii_derivs += 2*pow(atomic_radii, 5)*surface_tension*(probe_radius + atomic_radii)*(3*probe_radius + 4*atomic_radii)/br6;
    }
    radii_derivs += born_force_i * obc_chain_ri[atomI];
    out_dU_dp[atomic_radii_idxs[atomI]] += radii_derivs;

    born_force_i *= obc_chain[atomI];
    bornForces[atomI] = static_cast<unsigned long long>((long long) ( born_force_i*FIXED_EXPONENT_BORN_FORCES));

}


template <typename RealType, int D>
__global__ void k_reduce_born_forces_jvp(
    const int N,
    const double* params,
    const int* atomic_radii_idxs,
    const Surreal<double>* born_radii,
    const Surreal<double>* obc_chain,
    const Surreal<double>* obc_chain_ri,
    const double surface_tension, // surface area factor
    const double probe_radius,
    Surreal<double>* bornForces, // dU/Ri
    double* out_dU_dp
) {

    int atomI =  blockIdx.x*32 + threadIdx.x;
    if(atomI >= N) {
        return;
    }

    Surreal<RealType> radii_derivs(0, 0);
    if (born_radii[atomI] > 0.0) {
        RealType atomic_radii = params[atomic_radii_idxs[atomI]];
        RealType r            = atomic_radii + probe_radius;

        Surreal<RealType> ar = atomic_radii/born_radii[atomI];
        Surreal<RealType> ar2 = ar*ar;
        Surreal<RealType> ar4 = ar2*ar2;
        Surreal<RealType> ratio6 = ar4*ar2;

        Surreal<RealType> saTerm       = surface_tension*r*r*ratio6;
        bornForces[atomI]  -= 6.0*saTerm/born_radii[atomI]; 
        Surreal<RealType> br2 = born_radii[atomI]*born_radii[atomI];
        Surreal<RealType> br4 = br2*br2;
        Surreal<RealType> br6 = br4*br2;
        radii_derivs += 2*pow(atomic_radii, 5)*surface_tension*(probe_radius + atomic_radii)*(3*probe_radius + 4*atomic_radii)/br6;
    }
    radii_derivs += bornForces[atomI] * obc_chain_ri[atomI];
    out_dU_dp[atomic_radii_idxs[atomI]] += radii_derivs.imag;
    bornForces[atomI] *= obc_chain[atomI];

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
    const double* obc_chain_ri,
    const double dielectric_offset,
    const double cutoff,
    const unsigned long long* bornForces,
    unsigned long long* out_forces,
    double* out_dU_dp) {

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
    RealType born_radii_i = atom_i_idx < N ? born_radii[atom_i_idx] : 0;
    RealType dPsi_dri = 0;

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
    RealType dPsi_drj = 0;
    RealType dPsi_dsj = 0;

    const double dielectricOffset = dielectric_offset;
    const RealType cutoffDistance = cutoff;

    // RealType radiusI        = params[atomic_radii_idxs[atomI]];
    RealType offsetRadiusI  = radiusI - dielectricOffset;
    RealType offsetRadiusI2 = offsetRadiusI*offsetRadiusI;
    RealType offsetRadiusI3 = offsetRadiusI2*offsetRadiusI;

    int atomI = atom_i_idx;
    int atomJ = atom_j_idx;

    for(int round = 0; round < 32; round++) {

        if (atomJ != atomI) {

            RealType dxs[D];
            RealType r2 = 0;
            for(int d=0; d < D; d++) {
                dxs[d] = ci[d] - cj[d];
                r2 += dxs[d]*dxs[d];
            }
            RealType r = sqrt(r2);
            // radius w/ dielectric offset applied

            // RealType radiusJ            = params[atomic_radii_idxs[atomJ]];
            RealType offsetRadiusJ      = radiusJ - dielectricOffset;
            RealType offsetRadiusJ2     = offsetRadiusJ*offsetRadiusJ;

            // RealType scaleFactorJ       = params[scale_factor_idxs[atomJ]];
            RealType scaleFactorJ2      = scaleFactorJ*scaleFactorJ;
            RealType scaleFactorJ3      = scaleFactorJ2*scaleFactorJ;
            RealType scaledRadiusJ      = offsetRadiusJ*scaleFactorJ;
            // RealType scaledRadiusJ2     = scaledRadiusJ*scaledRadiusJ;
            RealType rScaledRadiusJ     = r + scaledRadiusJ;
            RealType rScaledRadiusJ2    = rScaledRadiusJ*rScaledRadiusJ;
            RealType rScaledRadiusJ3    = rScaledRadiusJ2*rScaledRadiusJ;

            if (offsetRadiusI < rScaledRadiusJ) {

                // double l_ij          = offsetRadiusI > abs(rSubScaledRadiusJ) ? offsetRadiusI : abs(rSubScaledRadiusJ);
                //        l_ij          = 1.0/l_ij;
                // double u_ij          = 1.0/rScaledRadiusJ;
                // double l_ij2         = l_ij*l_ij;
                // double u_ij2         = u_ij*u_ij; 
                // double rInverse      = 1.0/r;
                // double r2Inverse     = rInverse*rInverse;
                // double t3            = 0.125*(1.0 + scaledRadiusJ2*r2Inverse)*(l_ij2 - u_ij2) + 0.25*log(u_ij/l_ij)*r2Inverse;

                // printf("%d t3 RHS: %.8f\n", atomI, t3);
                // double de            = bornForces[atomI]*t3*rInverse;

                // for(int d=0; d < D; d++) {
                //     dPsi_dx[atomI*D+d] -= dxs[d]*de;
                //     dPsi_dx[atomJ*D+d] += dxs[d]*de;
                // }

                // start manual derivative
                RealType de = 0; // derivative of Psi wrt the distance
                RealType dpsi_dri = 0;
                RealType dpsi_drj = 0;
                RealType dpsi_dsj = 0;

                RealType rSubScaledRadiusJ = r - scaledRadiusJ;
                RealType rSubScaledRadiusJ2 = rSubScaledRadiusJ*rSubScaledRadiusJ;
                RealType rSubScaledRadiusJ3 = rSubScaledRadiusJ2*rSubScaledRadiusJ;

                // factor out as much as we can to outside of the conditional for reduce convergence
                if(offsetRadiusI > abs(rSubScaledRadiusJ)) {
                  RealType term = 0.5*(-offsetRadiusI)*(-0.25*r*(1/rScaledRadiusJ2 - 1/offsetRadiusI2) + 1.0/rScaledRadiusJ + 1.0/(-offsetRadiusI) + 0.25*scaleFactorJ2*offsetRadiusJ2*(1/rScaledRadiusJ2 - 1/offsetRadiusI2)/r - 0.5*log(offsetRadiusI/rScaledRadiusJ)/r);
                  de = -0.5*r/rScaledRadiusJ3 + (5.0/4.0)/rScaledRadiusJ2 - 0.25/offsetRadiusI2 + 0.5*scaleFactorJ2*offsetRadiusJ2/(r*rScaledRadiusJ3) - 0.5/(r*rScaledRadiusJ) - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r2 - 0.5*log(offsetRadiusI/rScaledRadiusJ)/r2;
                  dpsi_dri = 0.25*r*(1/rScaledRadiusJ2 - 1/offsetRadiusI2) + offsetRadiusI*(0.5*r/offsetRadiusI3 - 1/offsetRadiusI2 - 0.5*scaleFactorJ2*offsetRadiusJ2/(r*offsetRadiusI3) + 0.5/(r*offsetRadiusI)) - 1/rScaledRadiusJ + 1.0/offsetRadiusI + 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r + 0.5*log(offsetRadiusI/rScaledRadiusJ)/r;
                  dpsi_drj = offsetRadiusI*(-0.5*r*scaleFactorJ/rScaledRadiusJ3 + scaleFactorJ/rScaledRadiusJ2 + 0.5*scaleFactorJ3*offsetRadiusJ2/(r*rScaledRadiusJ3) + 0.25*scaleFactorJ2*(-2*dielectricOffset + 2*radiusJ)*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r - 0.5*scaleFactorJ/(r*rScaledRadiusJ));
                  dpsi_dsj = offsetRadiusI*(0.25*r*(2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3 + offsetRadiusJ/rScaledRadiusJ2 - 0.25*scaleFactorJ2*offsetRadiusJ2*(2*dielectricOffset - 2*radiusJ)/(r*rScaledRadiusJ3) + 0.5*scaleFactorJ*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r + 0.5*(-offsetRadiusJ)/(r*rScaledRadiusJ));
                  if(offsetRadiusI < (scaledRadiusJ - r)) {
                   de += 0;
                   dpsi_dri += 0;
                   dpsi_drj += 0;
                   dpsi_dsj += 0;
                  }

                } else {
                  RealType term = -0.5*(-offsetRadiusI)*(-0.25*r*(1/rSubScaledRadiusJ2 - 1/rScaledRadiusJ2) + 1.0/fabs(rSubScaledRadiusJ) - 1/rScaledRadiusJ - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rSubScaledRadiusJ2 + 1/rScaledRadiusJ2)/r + 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r);
                  de = 0.25*r*(-2/rScaledRadiusJ3 + 2/rSubScaledRadiusJ3) + (5.0/4.0)/rScaledRadiusJ2 - sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 - 0.25/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*offsetRadiusJ2*(2/rScaledRadiusJ3 - 2/rSubScaledRadiusJ3)/r + 0.5*rScaledRadiusJ*(sign(rSubScaledRadiusJ)/rScaledRadiusJ - fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)) - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r2 - 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r2;
                  dpsi_dri = 0.25*r*(1/rScaledRadiusJ2 - 1/rSubScaledRadiusJ2) + 1.0/fabs(rSubScaledRadiusJ) - 1/rScaledRadiusJ + 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r;
                  dpsi_drj = offsetRadiusI*(0.25*r*(-2*scaleFactorJ/rScaledRadiusJ3 - 2*scaleFactorJ/rSubScaledRadiusJ3) + scaleFactorJ/rScaledRadiusJ2 + scaleFactorJ*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*(-2*dielectricOffset + 2*radiusJ)*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.25*scaleFactorJ2*offsetRadiusJ2*(2*scaleFactorJ/rScaledRadiusJ3 + 2*scaleFactorJ/rSubScaledRadiusJ3)/r + 0.5*rScaledRadiusJ*(-scaleFactorJ*sign(rSubScaledRadiusJ)/rScaledRadiusJ - scaleFactorJ*fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)));
                  dpsi_dsj = offsetRadiusI*(0.25*r*(-(-2*dielectricOffset + 2*radiusJ)/rSubScaledRadiusJ3 + (2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3) + offsetRadiusJ/rScaledRadiusJ2 + offsetRadiusJ*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*offsetRadiusJ2*((-2*dielectricOffset + 2*radiusJ)/rSubScaledRadiusJ3 - (2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3)/r + 0.5*scaleFactorJ*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.5*rScaledRadiusJ*((-offsetRadiusJ)*sign(rSubScaledRadiusJ)/rScaledRadiusJ + (-offsetRadiusJ)*fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)));
                  if (offsetRadiusI < (scaledRadiusJ - r)) {
                   de += 2.0*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                   dpsi_dri += -2.0/fabs(rSubScaledRadiusJ);
                   dpsi_drj += -2.0*scaleFactorJ*offsetRadiusI*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                   dpsi_dsj += 2.0*offsetRadiusI*(-offsetRadiusJ)*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                  }
                }

                // is bornForces

                de *= 0.5*born_force_i*offsetRadiusI;
                dpsi_dri *= 0.5*born_force_i;
                dpsi_drj *= 0.5*born_force_i;
                dpsi_dsj *= 0.5*born_force_i;

                // parameter derivatives wrt atomic radii and scale factors
                dPsi_dri += dpsi_dri;
                dPsi_drj += dpsi_drj;
                dPsi_dsj += dpsi_dsj;

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
        dPsi_drj = __shfl_sync(0xffffffff, dPsi_drj, srcLane);
        dPsi_dsj = __shfl_sync(0xffffffff, dPsi_dsj, srcLane);

        atomic_radii_idx_i = __shfl_sync(0xffffffff, atomic_radii_idx_i, srcLane);
        scale_factor_idx_j = __shfl_sync(0xffffffff, scale_factor_idx_j, srcLane);

        for(int d=0; d < D; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            dPsi_dx_j[d] = __shfl_sync(0xffffffff, dPsi_dx_j[d], srcLane);
        }

    }

    for(int d=0; d < D; d++) {
        if(atomI < N) {
            atomicAdd(out_forces + atomI*D+d,  static_cast<unsigned long long>((long long) (dPsi_dx_i[d]*FIXED_EXPONENT)));
        }
        if(atomJ < N) {
            atomicAdd(out_forces + atomJ*D+d,  static_cast<unsigned long long>((long long) (dPsi_dx_j[d]*FIXED_EXPONENT)));
        }
    }


    if(atomI < N) {
        atomicAdd(out_dU_dp + atomic_radii_idx_i, dPsi_dri);
    }

    if(atomJ < N) {
        atomicAdd(out_dU_dp + atomic_radii_idx_j, dPsi_drj);
        atomicAdd(out_dU_dp + scale_factor_idx_j, dPsi_dsj);
    }


}



template <typename RealType, int D>
__global__ void k_compute_born_energy_and_forces_jvp(
    const int N,
    const double* coords,
    const double* coords_tangents,
    const double* params,
    const int* atomic_radii_idxs,
    const int* scale_factor_idxs,
    const Surreal<double>* born_radii,
    const Surreal<double>* obc_chain,
    const Surreal<double>* obc_chain_ri,
    const double dielectric_offset,
    const double cutoff,
    const Surreal<double>* bornForces,
    double* out_HvP,
    double* out_MvP) {

    // we always do the full interaction matrix due to non-symmetry

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    Surreal<RealType> ci[D];
    Surreal<RealType> dPsi_dx_i[D];
    for(int d=0; d < D; d++) {
        dPsi_dx_i[d] = Surreal<RealType>(0,0);
        ci[d].real = atom_i_idx < N ? coords[atom_i_idx*D+d] : 0;
        ci[d].imag = atom_i_idx < N ? coords_tangents[atom_i_idx*D+d] : 0;
    }

    int atomic_radii_idx_i = atom_i_idx < N ? atomic_radii_idxs[atom_i_idx] : 0;
    RealType radiusI = atom_i_idx < N ? params[atomic_radii_idx_i] : 0;
    Surreal<RealType> born_force_i = atom_i_idx < N ? bornForces[atom_i_idx] : Surreal<RealType>(0,0);
    Surreal<RealType> born_radii_i = atom_i_idx < N ? born_radii[atom_i_idx] : Surreal<RealType>(0,0);
    Surreal<RealType> dPsi_dri = Surreal<RealType>(0,0);

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    Surreal<RealType> cj[D];
    Surreal<RealType> dPsi_dx_j[D];
    for(int d=0; d < D; d++) {
        dPsi_dx_j[d]  = Surreal<RealType>(0,0);
        cj[d].real = atom_j_idx < N ? coords[atom_j_idx*D+d] : 0;
        cj[d].imag = atom_j_idx < N ? coords_tangents[atom_j_idx*D+d] : 0;
    }
    int atomic_radii_idx_j = atom_j_idx < N ? atomic_radii_idxs[atom_j_idx] : 0;
    RealType radiusJ = atom_j_idx < N ? params[atomic_radii_idx_j] : 0;

    int scale_factor_idx_j = atom_j_idx < N ? scale_factor_idxs[atom_j_idx] : 0;
    RealType scaleFactorJ = atom_j_idx < N ? params[scale_factor_idx_j] : 0;
    Surreal<RealType> born_radii_j = atom_j_idx < N ? born_radii[atom_j_idx] : Surreal<RealType>(0,0);
    Surreal<RealType> dPsi_drj = Surreal<RealType>(0,0);
    Surreal<RealType> dPsi_dsj = Surreal<RealType>(0,0);

    const double dielectricOffset = dielectric_offset;
    const RealType cutoffDistance = cutoff;

    // RealType radiusI        = params[atomic_radii_idxs[atomI]];
    RealType offsetRadiusI  = radiusI - dielectricOffset;
    RealType offsetRadiusI2 = offsetRadiusI*offsetRadiusI;
    RealType offsetRadiusI3 = offsetRadiusI2*offsetRadiusI;

    int atomI = atom_i_idx;
    int atomJ = atom_j_idx;

    // for (int atomJ = 0; atomJ < numberOfAtoms; atomJ++) {
    for(int round = 0; round < 32; round++) {

        if (atomJ != atomI) {

            Surreal<RealType> dxs[D];
            Surreal<RealType> r2(0, 0);
            for(int d=0; d < D; d++) {
                dxs[d] = ci[d] - cj[d];
                r2 += dxs[d]*dxs[d];
            }
            Surreal<RealType> r = sqrt(r2);
            // radius w/ dielectric offset applied

            // RealType radiusJ            = params[atomic_radii_idxs[atomJ]];
            RealType offsetRadiusJ      = radiusJ - dielectricOffset;
            RealType offsetRadiusJ2     = offsetRadiusJ*offsetRadiusJ;

            // RealType scaleFactorJ       = params[scale_factor_idxs[atomJ]];
            RealType scaleFactorJ2      = scaleFactorJ*scaleFactorJ;
            RealType scaleFactorJ3      = scaleFactorJ2*scaleFactorJ;
            RealType scaledRadiusJ      = offsetRadiusJ*scaleFactorJ;
            RealType scaledRadiusJ2     = scaledRadiusJ*scaledRadiusJ;
            Surreal<RealType> rScaledRadiusJ     = r + scaledRadiusJ;
            Surreal<RealType> rScaledRadiusJ2    = rScaledRadiusJ*rScaledRadiusJ;
            Surreal<RealType> rScaledRadiusJ3    = rScaledRadiusJ2*rScaledRadiusJ;

            if (offsetRadiusI < rScaledRadiusJ.real) {

                // double l_ij          = offsetRadiusI > abs(rSubScaledRadiusJ) ? offsetRadiusI : abs(rSubScaledRadiusJ);
                //        l_ij          = 1.0/l_ij;
                // double u_ij          = 1.0/rScaledRadiusJ;
                // double l_ij2         = l_ij*l_ij;
                // double u_ij2         = u_ij*u_ij; 
                // double rInverse      = 1.0/r;
                // double r2Inverse     = rInverse*rInverse;
                // double t3            = 0.125*(1.0 + scaledRadiusJ2*r2Inverse)*(l_ij2 - u_ij2) + 0.25*log(u_ij/l_ij)*r2Inverse;

                // printf("%d t3 RHS: %.8f\n", atomI, t3);
                // double de            = bornForces[atomI]*t3*rInverse;

                // for(int d=0; d < D; d++) {
                //     dPsi_dx[atomI*D+d] -= dxs[d]*de;
                //     dPsi_dx[atomJ*D+d] += dxs[d]*de;
                // }

                // start manual derivative
                Surreal<RealType> de = Surreal<RealType>(0, 0); // derivative of Psi wrt the distance
                Surreal<RealType> dpsi_dri = Surreal<RealType>(0, 0);
                Surreal<RealType> dpsi_drj = Surreal<RealType>(0, 0);
                Surreal<RealType> dpsi_dsj = Surreal<RealType>(0, 0);

                Surreal<RealType> rSubScaledRadiusJ = r - scaledRadiusJ;
                Surreal<RealType> rSubScaledRadiusJ2 = rSubScaledRadiusJ*rSubScaledRadiusJ;
                Surreal<RealType> rSubScaledRadiusJ3 = rSubScaledRadiusJ2*rSubScaledRadiusJ;

                // factor out as much as we can to outside of the conditional for reduce convergence
                if(offsetRadiusI > abs(rSubScaledRadiusJ).real) {
                    Surreal<RealType> term = 0.5*(-offsetRadiusI)*(-0.25*r*(1/rScaledRadiusJ2 - 1/offsetRadiusI2) + 1.0/rScaledRadiusJ + 1.0/(-offsetRadiusI) + 0.25*scaleFactorJ2*offsetRadiusJ2*(1/rScaledRadiusJ2 - 1/offsetRadiusI2)/r - 0.5*log(offsetRadiusI/rScaledRadiusJ)/r);
                    de = -0.5*r/rScaledRadiusJ3 + (5.0/4.0)/rScaledRadiusJ2 - 0.25/offsetRadiusI2 + 0.5*scaleFactorJ2*offsetRadiusJ2/(r*rScaledRadiusJ3) - 0.5/(r*rScaledRadiusJ) - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r2 - 0.5*log(offsetRadiusI/rScaledRadiusJ)/r2;
                    dpsi_dri = 0.25*r*(1/rScaledRadiusJ2 - 1/offsetRadiusI2) + offsetRadiusI*(0.5*r/offsetRadiusI3 - 1/offsetRadiusI2 - 0.5*scaleFactorJ2*offsetRadiusJ2/(r*offsetRadiusI3) + 0.5/(r*offsetRadiusI)) - 1/rScaledRadiusJ + 1.0/offsetRadiusI + 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r + 0.5*log(offsetRadiusI/rScaledRadiusJ)/r;
                    dpsi_drj = offsetRadiusI*(-0.5*r*scaleFactorJ/rScaledRadiusJ3 + scaleFactorJ/rScaledRadiusJ2 + 0.5*scaleFactorJ3*offsetRadiusJ2/(r*rScaledRadiusJ3) + 0.25*scaleFactorJ2*(-2*dielectricOffset + 2*radiusJ)*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r - 0.5*scaleFactorJ/(r*rScaledRadiusJ));
                    dpsi_dsj = offsetRadiusI*(0.25*r*(2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3 + offsetRadiusJ/rScaledRadiusJ2 - 0.25*scaleFactorJ2*offsetRadiusJ2*(2*dielectricOffset - 2*radiusJ)/(r*rScaledRadiusJ3) + 0.5*scaleFactorJ*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r + 0.5*(-offsetRadiusJ)/(r*rScaledRadiusJ));
                    
                    if(offsetRadiusI < (scaledRadiusJ - r).real) {
                        de += 0;
                        dpsi_dri += 0;
                        dpsi_drj += 0;
                    dpsi_dsj += 0;
                    }

                } else {
                    Surreal<RealType> term = -0.5*(-offsetRadiusI)*(-0.25*r*(1/rSubScaledRadiusJ2 - 1/rScaledRadiusJ2) + 1.0/fabs(rSubScaledRadiusJ) - 1/rScaledRadiusJ - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rSubScaledRadiusJ2 + 1/rScaledRadiusJ2)/r + 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r);
                    de = 0.25*r*(-2/rScaledRadiusJ3 + 2/rSubScaledRadiusJ3) + (5.0/4.0)/rScaledRadiusJ2 - sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 - 0.25/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*offsetRadiusJ2*(2/rScaledRadiusJ3 - 2/rSubScaledRadiusJ3)/r + 0.5*rScaledRadiusJ*(sign(rSubScaledRadiusJ)/rScaledRadiusJ - fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)) - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r2 - 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r2;
                    dpsi_dri = 0.25*r*(1/rScaledRadiusJ2 - 1/rSubScaledRadiusJ2) + 1.0/fabs(rSubScaledRadiusJ) - 1/rScaledRadiusJ + 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r;
                    dpsi_drj = offsetRadiusI*(0.25*r*(-2*scaleFactorJ/rScaledRadiusJ3 - 2*scaleFactorJ/rSubScaledRadiusJ3) + scaleFactorJ/rScaledRadiusJ2 + scaleFactorJ*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*(-2*dielectricOffset + 2*radiusJ)*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.25*scaleFactorJ2*offsetRadiusJ2*(2*scaleFactorJ/rScaledRadiusJ3 + 2*scaleFactorJ/rSubScaledRadiusJ3)/r + 0.5*rScaledRadiusJ*(-scaleFactorJ*sign(rSubScaledRadiusJ)/rScaledRadiusJ - scaleFactorJ*fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)));
                    dpsi_dsj = offsetRadiusI*(0.25*r*(-(-2*dielectricOffset + 2*radiusJ)/rSubScaledRadiusJ3 + (2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3) + offsetRadiusJ/rScaledRadiusJ2 + offsetRadiusJ*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*offsetRadiusJ2*((-2*dielectricOffset + 2*radiusJ)/rSubScaledRadiusJ3 - (2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3)/r + 0.5*scaleFactorJ*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.5*rScaledRadiusJ*((-offsetRadiusJ)*sign(rSubScaledRadiusJ)/rScaledRadiusJ + (-offsetRadiusJ)*fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)));
                    
                    if (offsetRadiusI < (scaledRadiusJ - r).real) {
                        de += 2.0*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                        dpsi_dri += -2.0/fabs(rSubScaledRadiusJ);
                        dpsi_drj += -2.0*scaleFactorJ*offsetRadiusI*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                        dpsi_dsj += 2.0*offsetRadiusI*(-offsetRadiusJ)*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                    }
                }

                // is bornForces

                de *= 0.5*born_force_i*offsetRadiusI;
                dpsi_dri *= 0.5*born_force_i;
                dpsi_drj *= 0.5*born_force_i;
                dpsi_dsj *= 0.5*born_force_i;

                // parameter derivatives wrt atomic radii and scale factors
                dPsi_dri += dpsi_dri;
                dPsi_drj += dpsi_drj;
                dPsi_dsj += dpsi_dsj;

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
        dPsi_drj = __shfl_sync(0xffffffff, dPsi_drj, srcLane);
        dPsi_dsj = __shfl_sync(0xffffffff, dPsi_dsj, srcLane);

        atomic_radii_idx_i = __shfl_sync(0xffffffff, atomic_radii_idx_i, srcLane);
        scale_factor_idx_j = __shfl_sync(0xffffffff, scale_factor_idx_j, srcLane);

        for(int d=0; d < D; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            dPsi_dx_j[d] = __shfl_sync(0xffffffff, dPsi_dx_j[d], srcLane);
        }

    }

    for(int d=0; d < D; d++) {
        if(atomI < N) {
            atomicAdd(out_HvP + atomI*D+d, dPsi_dx_i[d].imag);
        }
        if(atomJ < N) {
            atomicAdd(out_HvP + atomJ*D+d, dPsi_dx_j[d].imag);
        }
    }


    if(atomI < N) {
        atomicAdd(out_MvP + atomic_radii_idx_i, dPsi_dri.imag);
    }

    if(atomJ < N) {
        atomicAdd(out_MvP + atomic_radii_idx_j, dPsi_drj.imag);
        atomicAdd(out_MvP + scale_factor_idx_j, dPsi_dsj.imag);
    }


}
