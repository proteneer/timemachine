#include <stdexcept>
#include <iostream> 
#include "../fixed_point.hpp"
#include "surreal.cuh"
#include "kernel_utils.cuh"

#include "k_periodic_utils.cuh"

#define WARPSIZE 32

#include <cooperative_groups.h>

// since we need to do a full O(N^2) computing and we don't need to broadcast the forces,
// this should just be extremely efficient already

#define RADII_EXP 4


// nope, still need to parallelize out
template<typename RealType>
__global__ void k_compute_born_radii(
    const int N,
    const double* coords,
    const double* params,
    const double lambda,
    const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const int* atomic_radii_idxs,
    const int* scale_factor_idxs,
    const double dielectric_offset,
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    unsigned long long* born_psi) {

    RealType block_d2ij = 0; 
    for(int d=0; d < 3; d++) {
        RealType block_row_ctr = block_bounds_ctr[blockIdx.x*3+d];
        RealType block_col_ctr = block_bounds_ctr[blockIdx.y*3+d];
        RealType block_row_ext = block_bounds_ext[blockIdx.x*3+d];
        RealType block_col_ext = block_bounds_ext[blockIdx.y*3+d];
        RealType dx = max(0.0, fabs(block_row_ctr-block_col_ctr) - (block_row_ext+block_col_ext));
        block_d2ij += dx*dx;
    }

    if(block_d2ij > cutoff*cutoff) {
        return;
    }

    int atom_i_idx = blockIdx.x*32 + threadIdx.x;
    RealType lambda_i = lambda;
    if(atom_i_idx < N) {
        lambda_i = cutoff*lambda_plane_idxs[atom_i_idx] + lambda_offset_idxs[atom_i_idx]*lambda_i;
    }

    RealType ci[3];
    for(int d=0; d < 3; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
    }
    int radii_param_idx_i = atom_i_idx < N ? atomic_radii_idxs[atom_i_idx] : 0;

    RealType radiusI = atom_i_idx < N ? params[radii_param_idx_i] : 0;
    RealType offsetRadiusI   = radiusI - dielectric_offset;
    RealType radiusIInverse  = 1/offsetRadiusI;

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    RealType lambda_j = lambda;
    if(atom_j_idx < N) {
        lambda_j = cutoff*lambda_plane_idxs[atom_j_idx] + lambda_offset_idxs[atom_j_idx]*lambda_j;
    }

    // *always* accumulate in 64 bit.
    double sum = 0;
 
    RealType cj[3];
    for(int d=0; d < 3; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
    }

    int radii_param_idx_j = atom_j_idx < N ? atomic_radii_idxs[atom_j_idx] : 0;
    int scale_param_idx_j = atom_j_idx < N ? scale_factor_idxs[atom_j_idx] : 0;

    RealType radiusJ = atom_j_idx < N ? params[radii_param_idx_j] : 0;
    RealType scaleFactorJ = atom_j_idx < N ? params[scale_param_idx_j] : 0;

    RealType offsetRadiusJ = radiusJ - dielectric_offset; 
    RealType scaledRadiusJ = offsetRadiusJ*scaleFactorJ;

    for(int round = 0; round < 32; round++) {

        RealType dxs[4];
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
        }
        RealType delta_lambda = lambda_i - lambda_j;
        dxs[3] = apply_delta(delta_lambda, BOXSIZE); 

        RealType r = fast_vec_norm<RealType, 4>(dxs);
        RealType rInverse = 1/r;
        RealType rScaledRadiusJ  = r + scaledRadiusJ;
        RealType rSubScaledRadiusJ =  r - scaledRadiusJ;

        if(atom_j_idx != atom_i_idx && r < cutoff && atom_j_idx < N && atom_i_idx < N) {

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


                RealType inner = (PI*pow(r,RADII_EXP))/(BOXSIZE);
                RealType sw = cos(inner);
                sw = sw*sw;

                sw = 1;
                // at some point we need to compute the derivative of term with respect to the distance
                term = sw*term;

                sum += term;
            }
        }


        const int srcLane = (threadIdx.x + 1) % WARPSIZE;
        scaledRadiusJ = __shfl_sync(0xffffffff, scaledRadiusJ, srcLane);
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        for(int d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
        }
        lambda_j = __shfl_sync(0xffffffff, lambda_j, srcLane);

    }
    
    if(atom_i_idx < N) {
        atomicAdd(born_psi + atom_i_idx, static_cast<unsigned long long>((long long) (sum*FIXED_BORN_PSI)));
    }

}



template <typename RealType>
void __global__ k_compute_born_first_loop_gpu(
    const int N,
    const double* coords,
    const double* params,
    const double lambda,
    const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const int* charge_param_idxs,
    const double* born_radii,
    const double prefactor,
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    unsigned long long *bornForces,
    unsigned long long *out_forces,
    double *du_dl,
    double *out_energy) {

    if(blockIdx.y > blockIdx.x) {
        return;
    }

    RealType block_d2ij = 0; 
    for(int d=0; d < 3; d++) {
        RealType block_row_ctr = block_bounds_ctr[blockIdx.x*3+d];
        RealType block_col_ctr = block_bounds_ctr[blockIdx.y*3+d];
        RealType block_row_ext = block_bounds_ext[blockIdx.x*3+d];
        RealType block_col_ext = block_bounds_ext[blockIdx.y*3+d];
        RealType dx = max(0.0, fabs(block_row_ctr-block_col_ctr) - (block_row_ext+block_col_ext));
        block_d2ij += dx*dx;
    }

    if(block_d2ij > cutoff*cutoff) {
        return;
    }

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    RealType du_dl_i = 0;
    RealType dlambda_i = 0;
    RealType lambda_i = lambda;
    if(atom_i_idx < N) {
        lambda_i = cutoff*lambda_plane_idxs[atom_i_idx] + lambda_offset_idxs[atom_i_idx]*lambda_i;
        dlambda_i = lambda_offset_idxs[atom_i_idx];
    }

    RealType ci[3];
    RealType gi[3] = {0};
    for(int d=0; d < 3; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
    }
    int charge_param_idx_i = atom_i_idx < N ? charge_param_idxs[atom_i_idx] : 0;
    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType born_radii_i = atom_i_idx < N ? born_radii[atom_i_idx] : 0;

    // RealType dE_dqi_accum = 0;
    RealType born_force_i_accum = 0;

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    RealType du_dl_j = 0;
    RealType dlambda_j = 0;
    RealType lambda_j = lambda;
    if(atom_j_idx < N) {
        lambda_j = cutoff*lambda_plane_idxs[atom_j_idx] + lambda_offset_idxs[atom_j_idx]*lambda_j;
        dlambda_j = lambda_offset_idxs[atom_j_idx];
    }

    RealType cj[3];
    RealType gj[3] = {0};
    for(int d=0; d < 3; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
    }
    int charge_param_idx_j = atom_j_idx < N ? charge_param_idxs[atom_j_idx] : 0;
    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    RealType born_radii_j = atom_j_idx < N ? born_radii[atom_j_idx] : 0;
    // RealType dE_dqj_accum = 0;
    RealType born_force_j_accum = 0;

    RealType obc_energy = 0;

    // In inference mode, we don't care about gradients with respect to parameters.
    for(int round = 0; round < 32; round++) {

        RealType dxs[4];
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
        }
        RealType delta_lambda = lambda_i - lambda_j;
        dxs[3] = apply_delta(delta_lambda, BOXSIZE); 
        RealType r = fast_vec_norm<RealType, 4>(dxs);
        RealType r2 = r*r;
        RealType rInverse = 1/r;
        // RealType rInverse = fast_vec_rnorm<RealType, D>(dxs);

        if(atom_j_idx <= atom_i_idx && r < cutoff && atom_j_idx < N && atom_i_idx < N) {

            RealType alpha2_ij          = born_radii_i*born_radii_j;
            RealType D_ij               = r2/(4*alpha2_ij);
            RealType expTerm            = exp(-D_ij);
            RealType denominator2       = r2 + alpha2_ij*expTerm; 
            RealType denominator        = sqrt(denominator2);
            RealType Gpol               = (prefactor*qi*qj)/denominator; 
            RealType dGpol_dr           = -Gpol*(1 - expTerm/4)/denominator2;  
            RealType dGpol_dalpha2_ij   = -(Gpol/2)*expTerm*(1 + D_ij)/denominator2;
        
            RealType energy = Gpol;

            RealType inner = (PI*pow(r,8))/(BOXSIZE);
            RealType sw = cos(inner);
            sw = sw*sw;
            RealType dsw_dr = -(RADII_EXP)*pow(r, RADII_EXP-1)*(PI/cutoff)*sin(inner)*cos(inner);

            sw = 1;
            dsw_dr = 0;

            RealType dsw_dr_dot_E = dsw_dr*energy;


            if (atom_i_idx != atom_j_idx) {
                
                energy = sw*energy;

                born_force_j_accum += sw*dGpol_dalpha2_ij*born_radii_i;

                for(int d=0; d < 3; d++) {

                    // gi[d] += dxs[d]*dGpol_dr;
                    // gj[d] -= dxs[d]*dGpol_dr;
                    gi[d] += dxs[d]*sw*dGpol_dr + dsw_dr_dot_E*dxs[d]/r;
                    gj[d] -= dxs[d]*sw*dGpol_dr + dsw_dr_dot_E*dxs[d]/r;
                }

                RealType dw_i = dlambda_i;
                RealType dw_j = dlambda_j; // shuffled


                // du_dl_i += dxs[3]*dGpol_dr*dw_i;
                // du_dl_j -= dxs[3]*dGpol_dr*dw_j;

                du_dl_i += (dxs[3]*sw*dGpol_dr + dsw_dr_dot_E*dxs[3]/r)*dw_i;
                du_dl_j -= (dxs[3]*sw*dGpol_dr + dsw_dr_dot_E*dxs[3]/r)*dw_j;

            } else {
                energy *= 0.5;
            }

            obc_energy += energy;
            born_force_i_accum += sw*dGpol_dalpha2_ij*born_radii_j;
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE;
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        born_radii_j = __shfl_sync(0xffffffff, born_radii_j, srcLane);
        born_force_j_accum = __shfl_sync(0xffffffff, born_force_j_accum, srcLane);
        for(size_t d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
        lambda_j = __shfl_sync(0xffffffff, lambda_j, srcLane);
        dlambda_j = __shfl_sync(0xffffffff, dlambda_j, srcLane);
        du_dl_j = __shfl_sync(0xffffffff, du_dl_j, srcLane);
    }

    for(int d=0; d < 3; d++) {
        if(atom_i_idx < N) {
            atomicAdd(out_forces + atom_i_idx*3 + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));
        }
        if(atom_j_idx < N) {
            atomicAdd(out_forces + atom_j_idx*3 + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));
        }
    }

    if(atom_i_idx < N) {
        atomicAdd(bornForces + atom_i_idx, static_cast<unsigned long long>((long long) (born_force_i_accum*FIXED_EXPONENT_BORN_FORCES)));
    }

    if(atom_j_idx < N) {
        atomicAdd(bornForces + atom_j_idx, static_cast<unsigned long long>((long long) (born_force_j_accum*FIXED_EXPONENT_BORN_FORCES)));
    }

    atomicAdd(du_dl, du_dl_i + du_dl_j);
    atomicAdd(out_energy, obc_energy);

}


__global__ void k_reduce_born_radii(
    const int N,
    const double *params,
    const int* atomic_radii_idxs,
    const double dielectric_offset,
    const double alpha_obc,
    const double beta_obc,
    const double gamma_obc,
    const unsigned long long *born_psi,
    double *born_radii,
    double *obc_chain
) {

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    if(atom_i_idx >= N) {
        return;
    }

    int radii_param_idx_i = atom_i_idx < N ? atomic_radii_idxs[atom_i_idx] : 0;
    double radiusI = atom_i_idx < N ? params[radii_param_idx_i] : 0;
    double offsetRadiusI   = radiusI - dielectric_offset;

    double sum = static_cast<double>(static_cast<long long>(born_psi[atom_i_idx]))/FIXED_BORN_PSI;
    sum *= offsetRadiusI/2;

    double sum2       = sum*sum;
    double sum3       = sum*sum2;
    double inner      = alpha_obc*sum - beta_obc*sum2 + gamma_obc*sum3;
    double tanhSum    = tanh(inner);

    if(atom_i_idx < N) {
        double br = offsetRadiusI*radiusI/(radiusI - offsetRadiusI*tanhSum);
        born_radii[atom_i_idx] = br;
        obc_chain[atom_i_idx] = br*br*(1 - tanhSum*tanhSum)*(alpha_obc - 2*beta_obc*sum + 3*gamma_obc*sum2)/radiusI;
    }
}


// this is entirely done in double precision
__global__ void k_reduce_born_forces(
    const int N,
    const double* params,
    const int* atomic_radii_idxs,
    const double* born_radii,
    const double* obc_chain,
    const double surface_tension, // surface area factor
    const double probe_radius,
    unsigned long long* bornForces, // dU/Ri
    double *energy
) {

    // surface area term

    int atomI =  blockIdx.x*32 + threadIdx.x;
    if(atomI >= N) {
        return;
    }

    // double radii_derivs = 0;
    double born_force_i = static_cast<double>(static_cast<long long>(bornForces[atomI]))/FIXED_EXPONENT_BORN_FORCES;
    double br = born_radii[atomI];


    // ACE term
    if (br > 0.0) {
        double atomic_radii = params[atomic_radii_idxs[atomI]];
        double r            = atomic_radii + probe_radius;
        double ratio6       = pow(atomic_radii/born_radii[atomI], 6.0);
        double saTerm       = surface_tension*r*r*ratio6;
        atomicAdd(energy, saTerm);
        born_force_i  -= 6.0*saTerm/born_radii[atomI];
    }

    born_force_i *= obc_chain[atomI];
    bornForces[atomI] = static_cast<unsigned long long>((long long) (born_force_i*FIXED_EXPONENT_BORN_FORCES));

}



template <typename RealType>
__global__ void k_compute_born_energy_and_forces(
    const int N,
    const double* coords,
    const double* params,
    const double lambda,
    const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const int* atomic_radii_idxs,
    const int* scale_factor_idxs,
    const double* born_radii,
    const double* obc_chain,
    const double dielectric_offset,
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    const unsigned long long* bornForces,
    unsigned long long* out_forces,
    double *du_dl) {

    RealType block_d2ij = 0; 
    for(int d=0; d < 3; d++) {
        RealType block_row_ctr = block_bounds_ctr[blockIdx.x*3+d];
        RealType block_col_ctr = block_bounds_ctr[blockIdx.y*3+d];
        RealType block_row_ext = block_bounds_ext[blockIdx.x*3+d];
        RealType block_col_ext = block_bounds_ext[blockIdx.y*3+d];
        RealType dx = max(0.0, fabs(block_row_ctr-block_col_ctr) - (block_row_ext+block_col_ext));
        block_d2ij += dx*dx;
    }

    if(block_d2ij > cutoff*cutoff) {
        return;
    }

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    RealType du_dl_i = 0;
    RealType lambda_i = lambda;
    RealType dlambda_i = 0;
    if(atom_i_idx < N) {
        lambda_i = cutoff*lambda_plane_idxs[atom_i_idx] + lambda_offset_idxs[atom_i_idx]*lambda_i;
        dlambda_i = lambda_offset_idxs[atom_i_idx];
    }

    RealType ci[3];
    RealType dPsi_dx_i[3] = {0};
    for(int d=0; d < 3; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
    }

    int atomic_radii_idx_i = atom_i_idx < N ? atomic_radii_idxs[atom_i_idx] : 0;
    RealType radiusI = atom_i_idx < N ? params[atomic_radii_idx_i] : 0;
    RealType born_force_i = atom_i_idx < N ? static_cast<RealType>(static_cast<long long>(bornForces[atom_i_idx]))/FIXED_EXPONENT_BORN_FORCES : 0;
    // RealType born_radii_i = atom_i_idx < N ? born_radii[atom_i_idx] : 0;
    // RealType dPsi_dri = 0;

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    RealType du_dl_j = 0;
    RealType lambda_j = lambda;
    RealType dlambda_j = 0;
    if(atom_j_idx < N) {
        lambda_j = cutoff*lambda_plane_idxs[atom_j_idx] + lambda_offset_idxs[atom_j_idx]*lambda_j;
        dlambda_j = lambda_offset_idxs[atom_j_idx];
    }

    RealType cj[3];
    RealType dPsi_dx_j[3] = {0};
    for(int d=0; d < 3; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
    }
    int atomic_radii_idx_j = atom_j_idx < N ? atomic_radii_idxs[atom_j_idx] : 0;
    RealType radiusJ = atom_j_idx < N ? params[atomic_radii_idx_j] : 0;

    int scale_factor_idx_j = atom_j_idx < N ? scale_factor_idxs[atom_j_idx] : 0;
    RealType scaleFactorJ = atom_j_idx < N ? params[scale_factor_idx_j] : 0;
    RealType born_radii_j = atom_j_idx < N ? born_radii[atom_j_idx] : 0;

    const RealType dielectricOffset = dielectric_offset;

    RealType offsetRadiusI  = radiusI - dielectricOffset;

    // int atomI = atom_i_idx;
    // int atomJ = atom_j_idx;

    for(int round = 0; round < 32; round++) {

        RealType dxs[4];
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
        }
        RealType delta_lambda = lambda_i - lambda_j;
        dxs[3] = apply_delta(delta_lambda, BOXSIZE); 
        RealType r = fast_vec_norm<RealType, 4>(dxs);

        if (atom_j_idx != atom_i_idx && r < cutoff && atom_j_idx < N && atom_i_idx < N) {
            RealType rInverse = 1/r;
            // RealType rInverse = fast_vec_rnorm<RealType, D>(dxs);
            // radius w/ dielectric offset applied
            RealType offsetRadiusJ      = radiusJ - dielectricOffset;
            RealType scaledRadiusJ      = offsetRadiusJ*scaleFactorJ;
            RealType scaledRadiusJ2     = scaledRadiusJ*scaledRadiusJ;
            RealType rScaledRadiusJ     = r + scaledRadiusJ;

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

                RealType arss = abs(rSubScaledRadiusJ);

                RealType l_ij = offsetRadiusI > arss ? offsetRadiusI : arss;
                l_ij = 1/l_ij;
                RealType l_ij2 = l_ij*l_ij;

                RealType u_ij = 1/rScaledRadiusJ;
                RealType u_ij2 = u_ij*u_ij;

                // original expression
                // RealType term = l_ij - u_ij + 0.25*(u_ij2 - l_ij2)*t1 + (0.5*rInverse*ratio);

                RealType dl_dr = offsetRadiusI > arss ? 0 : -l_ij2*sign(rSubScaledRadiusJ);
                RealType du_dr = -u_ij2*sign(rScaledRadiusJ);

                RealType t1 = r - scaledRadiusJ2*rInverse;
                RealType dt1_dr = (1 + scaledRadiusJ2*rInverse*rInverse);
                RealType ratio = log(u_ij/l_ij);

                RealType de1 = dl_dr - du_dr;
                // we may need three separate accumulators for precision
                RealType de2 = (u_ij*du_dr - l_ij*dl_dr)*t1;
                RealType de3 = (u_ij2 - l_ij2)*dt1_dr/2;
                RealType de4 = rInverse*(rInverse*ratio - (du_dr/u_ij - dl_dr/l_ij));
                RealType de = de1 + (de2 + de3 - de4)/2; // this is the derivative with respect to r

                if(offsetRadiusI > arss) {
                    // if(offsetRadiusI < (scaledRadiusJ - r)) {
                        // de += 0;
                    // }
                } else {
                    if(offsetRadiusI >= rSubScaledRadiusJ) {
                        de += 2*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                    }
                }

                // needed for switch rescale
                RealType term = l_ij - u_ij + r*(u_ij2 - l_ij2)/4 + scaledRadiusJ*scaledRadiusJ*rInverse*(l_ij2 - u_ij2)/4 + rInverse*ratio/2;
                if (offsetRadiusI < (scaledRadiusJ - r)) {
                    RealType radiusIInverse  = 1/offsetRadiusI;
                    term += 2*(radiusIInverse - l_ij);
                }

                RealType inner = (PI*pow(r, RADII_EXP))/(BOXSIZE);
                RealType sw = cos(inner);
                sw = sw*sw;


                RealType dsw_dr = -(RADII_EXP)*pow(r, RADII_EXP-1)*(PI/cutoff)*sin(inner)*cos(inner);

                sw = 1;
                dsw_dr = 0;

                de = dsw_dr*term + de*sw;

                de *= born_force_i*offsetRadiusI/2;

                for(int d=0; d < 3; d++) {
                    RealType deriv = dxs[d]*de*rInverse;
                    dPsi_dx_i[d] += deriv;
                    dPsi_dx_j[d] -= deriv;
                }

                RealType dw_i = dlambda_i;
                RealType dw_j = dlambda_j;

                du_dl_i += dxs[3]*de*dw_i*rInverse;
                du_dl_j -= dxs[3]*de*dw_j*rInverse;

            }
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE;
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        // atomJ = __shfl_sync(0xffffffff, atomJ, srcLane);
        born_radii_j = __shfl_sync(0xffffffff, born_radii_j, srcLane);
        radiusJ = __shfl_sync(0xffffffff, radiusJ, srcLane);
        scaleFactorJ = __shfl_sync(0xffffffff, scaleFactorJ, srcLane);

        atomic_radii_idx_i = __shfl_sync(0xffffffff, atomic_radii_idx_i, srcLane);
        scale_factor_idx_j = __shfl_sync(0xffffffff, scale_factor_idx_j, srcLane);

        for(int d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            dPsi_dx_j[d] = __shfl_sync(0xffffffff, dPsi_dx_j[d], srcLane);
        }

        lambda_j = __shfl_sync(0xffffffff, lambda_j, srcLane);
        dlambda_j = __shfl_sync(0xffffffff, dlambda_j, srcLane);
        du_dl_j = __shfl_sync(0xffffffff, du_dl_j, srcLane);

    }

    for(int d=0; d < 3; d++) {
        if(atom_i_idx < N) {
            atomicAdd(out_forces + atom_i_idx*3+d,  static_cast<unsigned long long>((long long) (dPsi_dx_i[d]*FIXED_EXPONENT)));
        }
        if(atom_j_idx < N) {
            atomicAdd(out_forces + atom_j_idx*3+d,  static_cast<unsigned long long>((long long) (dPsi_dx_j[d]*FIXED_EXPONENT)));
        }
    }

    atomicAdd(du_dl, du_dl_i + du_dl_j);

}
