#include <stdexcept>
#include <iostream> 
#include "../fixed_point.hpp"
#include "surreal.cuh"
#include "kernel_utils.cuh"

#define WARPSIZE 32

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
    Surreal<double>* born_radii,
    Surreal<double>* obc_chain,
    Surreal<double>* obc_chain_ri) {

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;

    if(atom_i_idx >= N) {
        return;
    }

    // {
    // Surreal<RealType> terms_single[2976];
    Surreal<RealType> ci[D];
    for(int d=0; d < D; d++) {
        ci[d].real = coords[atom_i_idx*D+d];
        ci[d].imag = coords_tangents[atom_i_idx*D+d];
    }
    int radii_param_idx_i = atom_i_idx < N ? atomic_radii_idxs[atom_i_idx] : 0;

    RealType radiusI = atom_i_idx < N ? params[radii_param_idx_i] : 0;
    RealType offsetRadiusI   = radiusI - dielectric_offset;
    RealType radiusIInverse  = 1.0/offsetRadiusI;

    // do accumulation in RealType precision
    Surreal<double> sum_single(0, 0);

    const auto dielectricOffset = dielectric_offset;
 
    for(int atom_j_idx = 0; atom_j_idx < N; atom_j_idx++) {

        int radii_param_idx_j = atom_j_idx < N ? atomic_radii_idxs[atom_j_idx] : 0;
        int scale_param_idx_j = atom_j_idx < N ? scale_factor_idxs[atom_j_idx] : 0;

        RealType radiusJ = atom_j_idx < N ? params[radii_param_idx_j] : 0;
        RealType scaleFactorJ = atom_j_idx < N ? params[scale_param_idx_j] : 0;
        RealType scaleFactorJ2 = scaleFactorJ*scaleFactorJ;
        RealType offsetRadiusJ   = radiusJ - dielectric_offset; 
        RealType scaledRadiusJ   = offsetRadiusJ*scaleFactorJ;

        Surreal<RealType> r2(0, 0);
        for(int d=0; d < D; d++) {
            Surreal<RealType> cjd(coords[atom_j_idx*D+d], coords_tangents[atom_j_idx*D+d]);
            Surreal<RealType> dx = ci[d] - cjd;
            r2 += dx*dx;
        }
        Surreal<RealType> r = sqrt(r2);
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

                Surreal<RealType> ratio    = log(u_ij/l_ij);

                Surreal<RealType> term0    = l_ij - u_ij;
                Surreal<RealType> term1    = 0.25*r*(u_ij2 - l_ij2);
                Surreal<RealType> term2    = (0.5*rInverse*ratio); // add using double precision
                Surreal<RealType> term3    = (0.25*scaledRadiusJ*scaledRadiusJ*rInverse)*(l_ij2 - u_ij2);

                // Surreal<RealType> term     = term0 + term1 + term2 + term3;
                Surreal<RealType> term     = term0 + term1 + term3;

                // this case (atom i completely inside atom j) is not considered in the original paper
                // Jay Ponder and the authors of Tinker recognized this and
                // worked out the details
                if (offsetRadiusI < (scaledRadiusJ - r).real) {
                    term += 2.0*(1.0/offsetRadiusI - l_ij);
                }

                // terms_single[atom_j_idx] = u_ij/l_ij;

                sum_single.real += term.real;
                sum_single.imag += term.imag;

                sum_single.real += term2.real;
                sum_single.imag += term2.imag;

            }
        }
    }

    Surreal<double> sum = sum_single;
    sum                *= 0.5*offsetRadiusI;

    Surreal<double> sum2       = sum*sum;
    Surreal<double> sum3       = sum*sum2;

    Surreal<double> inner   = alpha_obc*sum - beta_obc*sum2 + gamma_obc*sum3;
    Surreal<double> tanhSum = tanh(inner);

    if(atom_i_idx < N) {

        // born_radii[atom_i_idx]      = 1.0/(1.0/offsetRadiusI - tanhSum/radiusI); 

        born_radii[atom_i_idx]      = (offsetRadiusI*radiusI)/(radiusI - offsetRadiusI*tanhSum);


        // printf("debug imag %f %f\n", sum.imag, sum.real);
        // dRi/dPsi
        obc_chain[atom_i_idx]       = (alpha_obc - 2.0*beta_obc*sum + 3.0*gamma_obc*sum2);
                                        // use secHsum
        // obc_chain[atom_i_idx]       = sechSum*sechSum*obc_chain[atom_i_idx]/radiusI;
        obc_chain[atom_i_idx]       = (1.0 - tanhSum*tanhSum)*obc_chain[atom_i_idx]/radiusI;
        obc_chain[atom_i_idx]      *= born_radii[atom_i_idx]*born_radii[atom_i_idx];

        // dRi/dri, this is per particle
        obc_chain_ri[atom_i_idx]    = 1.0/(offsetRadiusI*offsetRadiusI) - tanhSum/(radiusI*radiusI);
        obc_chain_ri[atom_i_idx]   *= born_radii[atom_i_idx]*born_radii[atom_i_idx];

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
    Surreal<RealType> born_radii_i;
    born_radii_i.real = atom_i_idx < N ? born_radii[atom_i_idx].real : 0;
    born_radii_i.imag = atom_i_idx < N ? born_radii[atom_i_idx].imag : 0;

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
    Surreal<RealType> born_radii_j;
    born_radii_j.real = atom_j_idx < N ? born_radii[atom_j_idx].real : 0;
    born_radii_j.imag = atom_j_idx < N ? born_radii[atom_j_idx].imag : 0;
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
            Surreal<RealType> dGpol_dr           = -Gpol*(static_cast<RealType>(1.0) - 0.25*expTerm)/denominator2;  
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
        atomicAddOffsetSplit(bornForces, atom_i_idx, born_force_i_accum); // add both real and imaginary part
        atomicAdd(out_MvP + charge_param_idx_i, dE_dqi_accum.imag); 
    }

    if(atom_j_idx < N) {
        atomicAddOffsetSplit(bornForces, atom_j_idx, born_force_j_accum); // add both real and imaginary
        atomicAdd(out_MvP + charge_param_idx_j, dE_dqj_accum.imag);
    }



}



// this is entirely done in double precision
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
    double* out_MvP
) {

    int atomI =  blockIdx.x*32 + threadIdx.x;
    if(atomI >= N) {
        return;
    }

    Surreal<double> radii_derivs(0, 0);
    if (born_radii[atomI] > 0.0) {
        double atomic_radii = params[atomic_radii_idxs[atomI]];
        double r            = atomic_radii + probe_radius;

        Surreal<double> ar = atomic_radii/born_radii[atomI];
        Surreal<double> ar2 = ar*ar;
        Surreal<double> ar4 = ar2*ar2;
        Surreal<double> ratio6 = ar4*ar2;

        Surreal<double> saTerm       = surface_tension*r*r*ratio6;
        bornForces[atomI]  -= 6.0*saTerm/born_radii[atomI]; 
        Surreal<double> br2 = born_radii[atomI]*born_radii[atomI];
        Surreal<double> br4 = br2*br2;
        Surreal<double> br6 = br4*br2;
        radii_derivs += 2*pow(atomic_radii, 5)*surface_tension*(probe_radius + atomic_radii)*(3*probe_radius + 4*atomic_radii)/br6;
    }
    radii_derivs += bornForces[atomI] * obc_chain_ri[atomI];
    // printf("%d adding to %d\n", atomI, atomic_radii_idxs[atomI]);
    atomicAdd(out_MvP + atomic_radii_idxs[atomI], radii_derivs.imag);
    bornForces[atomI] *= obc_chain[atomI];

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
    Surreal<RealType> born_force_i;
    born_force_i.real = atom_i_idx < N ? bornForces[atom_i_idx].real : 0;
    born_force_i.imag = atom_i_idx < N ? bornForces[atom_i_idx].imag : 0;
    Surreal<RealType> born_radii_i;
    born_radii_i.real = atom_i_idx < N ? born_radii[atom_i_idx].real : 0;
    born_radii_i.imag = atom_i_idx < N ? born_radii[atom_i_idx].imag : 0;
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
    Surreal<RealType> born_radii_j;
    born_radii_j.real = atom_j_idx < N ? born_radii[atom_j_idx].real : 0;
    born_radii_j.imag = atom_j_idx < N ? born_radii[atom_j_idx].imag : 0;

    Surreal<RealType> dPsi_drj = Surreal<RealType>(0,0);
    Surreal<RealType> dPsi_dsj = Surreal<RealType>(0,0);

    const RealType dielectricOffset = dielectric_offset;
    // const RealType cutoffDistance = cutoff;

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
            Surreal<RealType> rInverse = 1/r;
            // radius w/ dielectric offset applied

            // RealType radiusJ            = params[atomic_radii_idxs[atomJ]];
            RealType offsetRadiusJ      = radiusJ - dielectricOffset;
            RealType offsetRadiusJ2     = offsetRadiusJ*offsetRadiusJ;

            // RealType scaleFactorJ       = params[scale_factor_idxs[atomJ]];
            RealType scaleFactorJ2      = scaleFactorJ*scaleFactorJ;
            RealType scaleFactorJ3      = scaleFactorJ2*scaleFactorJ;
            RealType scaledRadiusJ      = offsetRadiusJ*scaleFactorJ;
            // RealType scaledRadiusJ2     = scaledRadiusJ*scaledRadiusJ;
            Surreal<RealType> rScaledRadiusJ     = r + scaledRadiusJ;
            Surreal<RealType> rScaledRadiusJ2    = rScaledRadiusJ*rScaledRadiusJ;
            Surreal<RealType> rScaledRadiusJ3    = rScaledRadiusJ2*rScaledRadiusJ;

            if (offsetRadiusI < rScaledRadiusJ.real) {

                // RealType l_ij          = offsetRadiusI > abs(rSubScaledRadiusJ) ? offsetRadiusI : abs(rSubScaledRadiusJ);
                //        l_ij          = 1.0/l_ij;
                // RealType u_ij          = 1.0/rScaledRadiusJ;
                // RealType l_ij2         = l_ij*l_ij;
                // RealType u_ij2         = u_ij*u_ij; 
                // RealType rInverse      = 1.0/r;
                // RealType r2Inverse     = rInverse*rInverse;
                // RealType t3            = 0.125*(1.0 + scaledRadiusJ2*r2Inverse)*(l_ij2 - u_ij2) + 0.25*log(u_ij/l_ij)*r2Inverse;

                // printf("%d t3 RHS: %.8f\n", atomI, t3);
                // RealType de            = bornForces[atomI]*t3*rInverse;

                // for(int d=0; d < D; d++) {
                //     dPsi_dx[atomI*D+d] -= dxs[d]*de;
                //     dPsi_dx[atomJ*D+d] += dxs[d]*de;
                // }

                // start manual derivative
                // Surreal<RealType> de = Surreal<RealType>(0, 0); // derivative of Psi wrt the distance
                Surreal<RealType> dpsi_dri(0, 0);
                Surreal<RealType> dpsi_drj(0, 0);


                Surreal<RealType> rSubScaledRadiusJ = r - scaledRadiusJ;
                Surreal<RealType> rSubScaledRadiusJ2 = rSubScaledRadiusJ*rSubScaledRadiusJ;
                Surreal<RealType> rSubScaledRadiusJ3 = rSubScaledRadiusJ2*rSubScaledRadiusJ;

                // Surreal<RealType> rInverse = 1/r; // oof high error

                Surreal<RealType> l_ij = offsetRadiusI > abs(rSubScaledRadiusJ).real ? Surreal<RealType>(offsetRadiusI, 0) : abs(rSubScaledRadiusJ);
                l_ij = 1.0/l_ij;
                Surreal<RealType> l_ij2 = l_ij*l_ij;

                Surreal<RealType> u_ij = 1.0/rScaledRadiusJ;
                Surreal<RealType> u_ij2 = u_ij*u_ij;


                Surreal<RealType> dl_dr = offsetRadiusI > abs(rSubScaledRadiusJ).real ? Surreal<RealType>(0, 0) : -(l_ij*l_ij)*(rSubScaledRadiusJ/abs(rSubScaledRadiusJ));
                Surreal<RealType> du_dr = -u_ij*u_ij*rScaledRadiusJ/abs(rScaledRadiusJ);


                Surreal<RealType> t1 = r - scaledRadiusJ*scaledRadiusJ*rInverse;
                Surreal<RealType> dt1_dr = (1 + scaledRadiusJ*scaledRadiusJ*rInverse*rInverse);
                // Surreal<RealType> dt1_dsj = -(2/r)*(offsetRadiusJ*offsetRadiusJ*scaleFactorJ);

                Surreal<RealType> ratio = log(u_ij/l_ij);

                //          l_ij - u_ij   + 0.25*(u_ij2 - l_ij2)*t1                          + (0.5*rInverse*ratio);
                // Surreal<RealType> du_dr_div_u = -u_ij*u_ij*abs(rScaledRadiusJ);
                // Surreal<RealType> dl_dr_div_l = offsetRadiusI > abs(rSubScaledRadiusJ) ? 0 : -l_ij*l_ij*abs(rSubScaledRadiusJ);

                // Surreal<RealType> du_dr_mul_u = -u_ij*u_ij/abs(rScaledRadiusJ);
                // Surreal<RealType> dl_dr_mul_l = offsetRadiusI > abs(rSubScaledRadiusJ) ? 0 : -l_ij*l_ij/abs(rSubScaledRadiusJ);


                Surreal<RealType> de1 = dl_dr - du_dr;
                Surreal<RealType> de2 = 0.5*(u_ij*du_dr - l_ij*dl_dr)*t1;
                Surreal<RealType> de3 = 0.25*(u_ij*u_ij - l_ij*l_ij)*dt1_dr;
                Surreal<RealType> de4 = -0.5*rInverse*(rInverse*ratio - (du_dr/u_ij - dl_dr/l_ij));
                Surreal<RealType> de = de1 + de2 + de3 + de4;

                Surreal<RealType> dl_dsj = offsetRadiusI > abs(rSubScaledRadiusJ).real ? Surreal<RealType>(0, 0) : l_ij*l_ij*offsetRadiusJ*(rSubScaledRadiusJ/abs(rSubScaledRadiusJ));
                Surreal<RealType> du_dsj = -u_ij*u_ij*offsetRadiusJ;

                Surreal<RealType> dsj1 = dl_dsj - du_dsj;
                Surreal<RealType> dsj2 = 0.5*r*(u_ij*du_dsj - l_ij*dl_dsj);
                Surreal<RealType> dsj3 = 0.5*rInverse*(du_dsj/u_ij - dl_dsj/l_ij);
                Surreal<RealType> dsj4 = 0.5*rInverse*(scaledRadiusJ*scaledRadiusJ)*(l_ij*dl_dsj - u_ij*du_dsj);
                Surreal<RealType> dsj5 = 0.5*rInverse*(scaledRadiusJ*offsetRadiusJ)*(l_ij2 - u_ij2);

                Surreal<RealType> dpsi_dsj = (dsj1 + dsj2 + dsj3 + dsj4 + dsj5)*offsetRadiusI;

                // Surreal<double> dpsi_dsj(0, 0);

                // factor out as much as we can to outside of the conditional for reduce convergence
                if(offsetRadiusI > abs(rSubScaledRadiusJ).real) {
                    // Surreal<double> term = 0.5*(-offsetRadiusI)*(-0.25*r*(1/rScaledRadiusJ2 - 1/offsetRadiusI2) + 1.0/rScaledRadiusJ + 1.0/(-offsetRadiusI) + 0.25*scaleFactorJ2*offsetRadiusJ2*(1/rScaledRadiusJ2 - 1/offsetRadiusI2)/r - 0.5*log(offsetRadiusI/rScaledRadiusJ)/r);
                    // de = -0.5*r/rScaledRadiusJ3 + (5.0/4.0)/rScaledRadiusJ2 - 0.25/offsetRadiusI2 + 0.5*scaleFactorJ2*offsetRadiusJ2/(r*rScaledRadiusJ3) - 0.5/(r*rScaledRadiusJ) - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r2 - 0.5*log(offsetRadiusI/rScaledRadiusJ)/r2;
                    dpsi_dri = 0.25*r*(1/rScaledRadiusJ2 - 1/offsetRadiusI2) + offsetRadiusI*(0.5*r/offsetRadiusI3 - 1/offsetRadiusI2 - 0.5*scaleFactorJ2*offsetRadiusJ2/(r*offsetRadiusI3) + 0.5/(r*offsetRadiusI)) - 1/rScaledRadiusJ + 1.0/offsetRadiusI + 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r + 0.5*log(offsetRadiusI/rScaledRadiusJ)/r;
                    dpsi_drj = offsetRadiusI*(-0.5*r*scaleFactorJ/rScaledRadiusJ3 + scaleFactorJ/rScaledRadiusJ2 + 0.5*scaleFactorJ3*offsetRadiusJ2/(r*rScaledRadiusJ3) + 0.25*scaleFactorJ2*(-2*dielectricOffset + 2*radiusJ)*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r - 0.5*scaleFactorJ/(r*rScaledRadiusJ));
                    // dpsi_dsj = offsetRadiusI*(0.25*r*(2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3 + offsetRadiusJ/rScaledRadiusJ2 - 0.25*scaleFactorJ2*offsetRadiusJ2*(2*dielectricOffset - 2*radiusJ)/(r*rScaledRadiusJ3) + 0.5*scaleFactorJ*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/offsetRadiusI2)/r + 0.5*(-offsetRadiusJ)/(r*rScaledRadiusJ));
                    
                    if(offsetRadiusI < (scaledRadiusJ - r).real) {
                        de += 0;
                        dpsi_dri += 0;
                        dpsi_drj += 0;
                        dpsi_dsj += 0;
                    }

                } else {
                    // Surreal<double> term = -0.5*(-offsetRadiusI)*(-0.25*r*(1/rSubScaledRadiusJ2 - 1/rScaledRadiusJ2) + 1.0/fabs(rSubScaledRadiusJ) - 1/rScaledRadiusJ - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rSubScaledRadiusJ2 + 1/rScaledRadiusJ2)/r + 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r);
                    // de = 0.25*r*(-2/rScaledRadiusJ3 + 2/rSubScaledRadiusJ3) + (5.0/4.0)/rScaledRadiusJ2 - sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 - 0.25/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*offsetRadiusJ2*(2/rScaledRadiusJ3 - 2/rSubScaledRadiusJ3)/r + 0.5*rScaledRadiusJ*(sign(rSubScaledRadiusJ)/rScaledRadiusJ - fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)) - 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r2 - 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r2;
                    dpsi_dri = 0.25*r*(1/rScaledRadiusJ2 - 1/rSubScaledRadiusJ2) + 1.0/fabs(rSubScaledRadiusJ) - 1/rScaledRadiusJ + 0.25*scaleFactorJ2*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.5*log(fabs(rSubScaledRadiusJ)/rScaledRadiusJ)/r;
                    dpsi_drj = offsetRadiusI*(0.25*r*(-2*scaleFactorJ/rScaledRadiusJ3 - 2*scaleFactorJ/rSubScaledRadiusJ3) + scaleFactorJ/rScaledRadiusJ2 + scaleFactorJ*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*(-2*dielectricOffset + 2*radiusJ)*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.25*scaleFactorJ2*offsetRadiusJ2*(2*scaleFactorJ/rScaledRadiusJ3 + 2*scaleFactorJ/rSubScaledRadiusJ3)/r + 0.5*rScaledRadiusJ*(-scaleFactorJ*sign(rSubScaledRadiusJ)/rScaledRadiusJ - scaleFactorJ*fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)));
                    // dpsi_dsj = offsetRadiusI*(0.25*r*(-(-2*dielectricOffset + 2*radiusJ)/rSubScaledRadiusJ3 + (2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3) + offsetRadiusJ/rScaledRadiusJ2 + offsetRadiusJ*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2 + 0.25*scaleFactorJ2*offsetRadiusJ2*((-2*dielectricOffset + 2*radiusJ)/rSubScaledRadiusJ3 - (2*dielectricOffset - 2*radiusJ)/rScaledRadiusJ3)/r + 0.5*scaleFactorJ*offsetRadiusJ2*(-1/rScaledRadiusJ2 + 1/rSubScaledRadiusJ2)/r + 0.5*rScaledRadiusJ*((-offsetRadiusJ)*sign(rSubScaledRadiusJ)/rScaledRadiusJ + (-offsetRadiusJ)*fabs(rSubScaledRadiusJ)/rScaledRadiusJ2)/(r*fabs(rSubScaledRadiusJ)));
                    
                    if (offsetRadiusI < (scaledRadiusJ - r).real) {
                        de += 2.0*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                        dpsi_dri += -2.0/fabs(rSubScaledRadiusJ);
                        dpsi_drj += -2.0*scaleFactorJ*offsetRadiusI*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                        // dpsi_dsj += 2.0*offsetRadiusI*(-offsetRadiusJ)*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                        dpsi_dsj += -2*dl_dsj*offsetRadiusI;
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


