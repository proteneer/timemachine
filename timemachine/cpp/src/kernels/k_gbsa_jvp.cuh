// MODIFIED

#include <stdexcept>
#include <iostream> 
#include "../fixed_point.hpp"
#include "surreal.cuh"
#include "kernel_utils.cuh"

#define WARPSIZE 32

template<typename RealType>
__global__ void k_compute_born_radii_gpu_jvp(
    const int N,
    const double* coords,
    const double* coords_tangents,
    // const double* params,
    const double lambda_primal,
    const double lambda_tangent,
    const int* lambda_plane_idxs,
    const int* lambda_offset_idxs,
    const double *gb_params,
    // const int* atomic_radii_idxs,
    // const int* scale_factor_idxs,
    const double dielectric_offset,
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    Surreal<double>* born_radii) {

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

    int lambda_plane_i = 0;
    int lambda_offset_i = 0;

    if(atom_i_idx < N) {
        lambda_plane_i = lambda_plane_idxs[atom_i_idx];
        lambda_offset_i = lambda_offset_idxs[atom_i_idx];
    }

    Surreal<RealType> ci[3];
    for(int d=0; d < 3; d++) {
        ci[d].real = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
        ci[d].imag = atom_i_idx < N ? coords_tangents[atom_i_idx*3+d] : 0;
    }
    int radii_param_idx_i = atom_i_idx < N ? atom_i_idx*2 + 0 : 0;

    RealType radiusI = atom_i_idx < N ? gb_params[radii_param_idx_i] : 0;
    RealType offsetRadiusI   = radiusI - dielectric_offset;

    // test accumulation in RealType precision 
    Surreal<double> sum(0, 0);

    const auto dielectricOffset = dielectric_offset;

    int atom_j_idx =  blockIdx.y*32 + threadIdx.x;
    int lambda_plane_j = 0;
    int lambda_offset_j = 0;
    if(atom_j_idx < N) {
        lambda_plane_j = lambda_plane_idxs[atom_j_idx];
        lambda_offset_j = lambda_offset_idxs[atom_j_idx];
    }

    int radii_param_idx_j = atom_j_idx < N ? atom_j_idx*2+0 : 0;
    int scale_param_idx_j = atom_j_idx < N ? atom_j_idx*2+1 : 0;

    Surreal<RealType> cj[3];
    for(int d=0; d < 3; d++) {
        cj[d].real = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
        cj[d].imag = atom_j_idx < N ? coords_tangents[atom_j_idx*3+d] : 0;
    }

    RealType radiusJ = atom_j_idx < N ? gb_params[radii_param_idx_j] : 0;
    RealType scaleFactorJ = atom_j_idx < N ? gb_params[scale_param_idx_j] : 0;
    RealType scaleFactorJ2 = scaleFactorJ*scaleFactorJ;
    RealType offsetRadiusJ   = radiusJ - dielectric_offset; 
    RealType scaledRadiusJ   = offsetRadiusJ*scaleFactorJ;

    Surreal<RealType> lambda(lambda_primal, lambda_tangent);

    for(int round = 0; round < 32; round++) {

        Surreal<RealType> r2(0,0);
        for(int d=0; d < 3; d++) {
            Surreal<RealType> dx = ci[d] - cj[d];
            r2 += dx*dx;
        }
        // Surreal<RealType> delta_lambda = lambda_i - lambda_j;
        Surreal<RealType> delta_lambda = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*lambda;
        r2 += delta_lambda * delta_lambda;

        if(atom_j_idx != atom_i_idx && r2.real < cutoff*cutoff && atom_i_idx < N && atom_j_idx < N) {

            Surreal<RealType> r = sqrt(r2);
            Surreal<RealType> rScaledRadiusJ  = r + scaledRadiusJ;
            Surreal<RealType> rSubScaledRadiusJ =  r - scaledRadiusJ;

            if (offsetRadiusI < rScaledRadiusJ.real) {
                Surreal<RealType> rInverse = 1/r;

                Surreal<RealType> l_ij(0,0);
                if(offsetRadiusI > abs(rSubScaledRadiusJ).real) {
                  l_ij.real = offsetRadiusI;
                  l_ij.imag = 0;
                } else {
                  l_ij = abs(rSubScaledRadiusJ);
                }

                l_ij     = 1/l_ij;

                Surreal<RealType> u_ij     = 1/rScaledRadiusJ;

                Surreal<RealType> l_ij2    = l_ij*l_ij;
                Surreal<RealType> u_ij2    = u_ij*u_ij;

                Surreal<RealType> ratio    = log(u_ij/l_ij);

                Surreal<RealType> term0    = l_ij - u_ij;
                Surreal<RealType> term1    = r*(u_ij2 - l_ij2)/4;
                Surreal<RealType> term2    = rInverse*ratio/2; // add using double precision
                Surreal<RealType> term3    = scaledRadiusJ*scaledRadiusJ*rInverse*(l_ij2 - u_ij2)/4;

                Surreal<RealType> term     = term0 + term1 + term3 + term2;

                // this case (atom i completely inside atom j) is not considered in the original paper
                // Jay Ponder and the authors of Tinker recognized this and
                // worked out the details
                if (offsetRadiusI < (scaledRadiusJ - r).real) {
                    term += 2*(1/offsetRadiusI - l_ij);
                }

                sum.real += term.real;
                sum.imag += term.imag;

            }
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE;
        scaledRadiusJ = __shfl_sync(0xffffffff, scaledRadiusJ, srcLane);
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        for(int d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
        }
        // lambda_j = __shfl_sync(0xffffffff, lambda_j, srcLane);
        lambda_plane_j = __shfl_sync(0xffffffff, lambda_plane_j, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane);
    }

    if(atom_i_idx < N) {
        atomicAddOffset(born_radii, atom_i_idx, sum);
    }

}


template <typename RealType>
void __global__ k_compute_born_first_loop_gpu_jvp(
    const int N,
    const double* coords,
    const double* coords_tangents,
    // const double* params,
    const double lambda_primal,
    const double lambda_tangent,
    const int* lambda_plane_idxs,
    const int* lambda_offset_idxs,
    // const int* charge_param_idxs,
    const double *charge_params,
    const Surreal<double>* born_radii,
    const double prefactor,
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    Surreal<double> *bornForces,
    double *out_HvP_primals,
    double *out_HvP_tangents,
    double *out_grad_charge_primals,
    double *out_grad_charge_tangents) {

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
    int lambda_plane_i = 0;
    int lambda_offset_i = 0;

    if(atom_i_idx < N) {
        lambda_plane_i = lambda_plane_idxs[atom_i_idx];
        lambda_offset_i = lambda_offset_idxs[atom_i_idx];
    }

    Surreal<RealType> ci[3];
    Surreal<RealType> gi[3];
    for(int d=0; d < 3; d++) {
        gi[d].real = 0;
        gi[d].imag = 0;
        ci[d].real = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
        ci[d].imag = atom_i_idx < N ? coords_tangents[atom_i_idx*3+d] : 0;
    }
    int charge_param_idx_i = atom_i_idx < N ? atom_i_idx : 0;
    RealType qi = atom_i_idx < N ? charge_params[charge_param_idx_i] : 0;
    Surreal<RealType> born_radii_i;
    born_radii_i.real = atom_i_idx < N ? born_radii[atom_i_idx].real : 0;
    born_radii_i.imag = atom_i_idx < N ? born_radii[atom_i_idx].imag : 0;

    Surreal<RealType> dE_dqi_accum(0, 0);
    Surreal<RealType> born_force_i_accum(0, 0);

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    int lambda_plane_j = 0;
    int lambda_offset_j = 0;

    if(atom_j_idx < N) {
        lambda_plane_j = lambda_plane_idxs[atom_j_idx];
        lambda_offset_j = lambda_offset_idxs[atom_j_idx];
    }

    Surreal<RealType> cj[3];
    Surreal<RealType> gj[3];
    for(int d=0; d < 3; d++) {
        gj[d].real = 0;
        gj[d].imag = 0;
        cj[d].real = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
        cj[d].imag = atom_j_idx < N ? coords_tangents[atom_j_idx*3+d] : 0;
    }
    int charge_param_idx_j = atom_j_idx < N ? atom_j_idx : 0;
    RealType qj = atom_j_idx < N ? charge_params[charge_param_idx_j] : 0;
    Surreal<RealType> born_radii_j;
    born_radii_j.real = atom_j_idx < N ? born_radii[atom_j_idx].real : 0;
    born_radii_j.imag = atom_j_idx < N ? born_radii[atom_j_idx].imag : 0;
    Surreal<RealType> dE_dqj_accum(0, 0);
    Surreal<RealType> born_force_j_accum(0, 0);

    Surreal<RealType> lambda(lambda_primal, lambda_tangent);

    // In inference mode, we don't care about gradients with respect to parameters.
    for(int round = 0; round < 32; round++) {

        Surreal<RealType> dxs[3];
        Surreal<RealType> r2(0, 0);
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
            r2 += dxs[d]*dxs[d];
        }

        Surreal<RealType> delta_lambda = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*lambda;
        r2 += delta_lambda * delta_lambda;

        if(atom_j_idx <= atom_i_idx && r2.real < cutoff*cutoff && atom_j_idx < N && atom_i_idx < N) {

            Surreal<RealType> r = sqrt(r2);

            Surreal<RealType> alpha2_ij          = born_radii_i*born_radii_j;
            Surreal<RealType> D_ij               = r2/(4*alpha2_ij);
            Surreal<RealType> expTerm            = exp(-D_ij);
            Surreal<RealType> denominator2       = r2 + alpha2_ij*expTerm; 
            Surreal<RealType> denominator        = sqrt(denominator2);
            Surreal<RealType> Gpol               = (prefactor*qi*qj)/denominator; 
            Surreal<RealType> dGpol_dr           = -Gpol*(1 - expTerm/4)/denominator2;  
            Surreal<RealType> dGpol_dalpha2_ij   = -Gpol*expTerm*(1 + D_ij)/(2*denominator2);

            Surreal<RealType> dE_dqi = prefactor*qj/denominator;
            Surreal<RealType> dE_dqj = prefactor*qi/denominator;

            if (atom_i_idx != atom_j_idx) {

                // TBD: determine what we should do with cutoff
                // energy -= qi*partialCharges[atom_j_idx]/cutoff;
                // bornForces[atom_j_idx]        += dGpol_dalpha2_ij*born_radii[atom_i_idx];
                born_force_j_accum += dGpol_dalpha2_ij*born_radii_i;

                for(int d=0; d < 3; d++) {
                    gi[d] += dxs[d]*dGpol_dr;
                    gj[d] -= dxs[d]*dGpol_dr;
                }

            } else {
                dE_dqi *= 0.5;
                dE_dqj *= 0.5;
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
        for(size_t d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
        // lambda_j = __shfl_sync(0xffffffff, lambda_j, srcLane);
        lambda_plane_j = __shfl_sync(0xffffffff, lambda_plane_j, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane);
    }

    for(int d=0; d < 3; d++) {
        if(atom_i_idx < N) {
            atomicAdd(out_HvP_primals + atom_i_idx*3 + d, gi[d].real);
            atomicAdd(out_HvP_tangents + atom_i_idx*3 + d, gi[d].imag);
        }
        if(atom_j_idx < N) {
            atomicAdd(out_HvP_primals + atom_j_idx*3 + d, gj[d].real);
            atomicAdd(out_HvP_tangents + atom_j_idx*3 + d, gj[d].imag);
        }
    }

    if(atom_i_idx < N) {
        atomicAddOffsetSplit(bornForces, atom_i_idx, born_force_i_accum); // add both real and imaginary part
        atomicAdd(out_grad_charge_primals + charge_param_idx_i, dE_dqi_accum.real); 
        atomicAdd(out_grad_charge_tangents + charge_param_idx_i, dE_dqi_accum.imag); 
    }

    if(atom_j_idx < N) {
        atomicAddOffsetSplit(bornForces, atom_j_idx, born_force_j_accum); // add both real and imaginary
        atomicAdd(out_grad_charge_primals + charge_param_idx_j, dE_dqj_accum.real);
        atomicAdd(out_grad_charge_tangents + charge_param_idx_j, dE_dqj_accum.imag);
    }

}

__global__ void k_reduce_born_radii_jvp(
    const int N,
    const double *gb_params,
    // const int* atomic_radii_idxs,
    const double dielectric_offset,
    const double alpha_obc,
    const double beta_obc,
    const double gamma_obc,
    Surreal<double> *born_radii,
    Surreal<double> *obc_chain,
    Surreal<double> *obc_chain_ri
) {

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    if(atom_i_idx >= N) {
        return;
    }

    int radii_param_idx_i = atom_i_idx < N ? atom_i_idx*2+0 : 0;
    double radiusI = atom_i_idx < N ? gb_params[radii_param_idx_i] : 0;
    double offsetRadiusI = radiusI - dielectric_offset;

    Surreal<double> sum = born_radii[atom_i_idx];

    sum *= offsetRadiusI/2;

    Surreal<double> sum2       = sum*sum;
    Surreal<double> sum3       = sum*sum2;
    Surreal<double> inner      = alpha_obc*sum - beta_obc*sum2 + gamma_obc*sum3;
    Surreal<double> tanhSum    = tanh(inner);

    if(atom_i_idx < N) {
        Surreal<double> br = offsetRadiusI*radiusI/(radiusI - offsetRadiusI*tanhSum);

        born_radii[atom_i_idx] = br;
        obc_chain[atom_i_idx] = br*br*(1 - tanhSum*tanhSum)*(alpha_obc - 2*beta_obc*sum + 3*gamma_obc*sum2)/radiusI;
        obc_chain_ri[atom_i_idx] = br*br*(1/(offsetRadiusI*offsetRadiusI) - tanhSum/(radiusI*radiusI));
    }
}

// this is entirely done in double precision
__global__ void k_reduce_born_forces_jvp(
    const int N,
    const double* gb_params,
    // const int* atomic_radii_idxs,
    const Surreal<double>* born_radii,
    const Surreal<double>* obc_chain,
    const Surreal<double>* obc_chain_ri,
    const double surface_tension, // surface area factor
    const double probe_radius,
    Surreal<double>* bornForces, // dU/Ri
    double *out_grad_gb_primals,
    double *out_grad_gb_tangents
    // double* out_MvP_primals,
    // double* out_MvP_tangents
) {

    int atomI =  blockIdx.x*32 + threadIdx.x;
    if(atomI >= N) {
        return;
    }
    
    int atomic_radii_idx_i = atomI*2 + 0;
    Surreal<double> radii_derivs(0, 0);
    if (born_radii[atomI] > 0.0) {
        double atomic_radii = gb_params[atomic_radii_idx_i];
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
    atomicAdd(out_grad_gb_primals + atomic_radii_idx_i, radii_derivs.real);
    atomicAdd(out_grad_gb_tangents + atomic_radii_idx_i, radii_derivs.imag);
    bornForces[atomI] *= obc_chain[atomI];

}


template <typename RealType, int D>
__global__ void k_compute_born_energy_and_forces_jvp(
    const int N,
    const double* coords,
    const double* coords_tangents,
    // const double* params,
    const double lambda_primal,
    const double lambda_tangent,
    const int* lambda_plane_idxs,
    const int* lambda_offset_idxs,
    const double *gb_params,
    // const int* atomic_radii_idxs,
    // const int* scale_factor_idxs,
    const Surreal<double>* born_radii,
    const Surreal<double>* obc_chain,
    const Surreal<double>* obc_chain_ri,
    const double dielectric_offset,
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    const Surreal<double>* bornForces,
    double* out_HvP_primals,
    double* out_HvP_tangents,
    double* out_MvP_primals,
    double* out_MvP_tangents) {

    // we always do the full interaction matrix due to non-symmetry
    RealType block_d2ij = 0; 
    for(int d=0; d < D; d++) {
        RealType block_row_ctr = block_bounds_ctr[blockIdx.x*D+d];
        RealType block_col_ctr = block_bounds_ctr[blockIdx.y*D+d];
        RealType block_row_ext = block_bounds_ext[blockIdx.x*D+d];
        RealType block_col_ext = block_bounds_ext[blockIdx.y*D+d];
        RealType dx = max(0.0, fabs(block_row_ctr-block_col_ctr) - (block_row_ext+block_col_ext));
        block_d2ij += dx*dx;
    }

    if(block_d2ij > cutoff*cutoff) {
        return;
    }

    // (ytz): we can probably do this computation twice by flipping i and j, but we have more
    // parallelism as is
    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    int lambda_plane_i = 0;
    int lambda_offset_i = 0;

    if(atom_i_idx < N) {
        lambda_plane_i = lambda_plane_idxs[atom_i_idx];
        lambda_offset_i = lambda_offset_idxs[atom_i_idx];
    }

    Surreal<RealType> ci[D];
    Surreal<RealType> dPsi_dx_i[D];
    for(int d=0; d < D; d++) {
        dPsi_dx_i[d].real = 0;
        dPsi_dx_i[d].imag = 0;
        ci[d].real = atom_i_idx < N ? coords[atom_i_idx*D+d] : 0;
        ci[d].imag = atom_i_idx < N ? coords_tangents[atom_i_idx*D+d] : 0;
    }

    int atomic_radii_idx_i = atom_i_idx < N ? atom_i_idx*2+0 : 0;
    RealType radiusI = atom_i_idx < N ? gb_params[atomic_radii_idx_i] : 0;
    Surreal<RealType> born_force_i;
    born_force_i.real = atom_i_idx < N ? bornForces[atom_i_idx].real : 0;
    born_force_i.imag = atom_i_idx < N ? bornForces[atom_i_idx].imag : 0;
    Surreal<RealType> born_radii_i;
    born_radii_i.real = atom_i_idx < N ? born_radii[atom_i_idx].real : 0;
    born_radii_i.imag = atom_i_idx < N ? born_radii[atom_i_idx].imag : 0;

    Surreal<RealType> dPsi_dri(0, 0);

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    int lambda_plane_j = 0;
    int lambda_offset_j = 0;

    if(atom_j_idx < N) {
        lambda_plane_j = lambda_plane_idxs[atom_j_idx];
        lambda_offset_j = lambda_offset_idxs[atom_j_idx];
    }

    Surreal<RealType> cj[D];
    Surreal<RealType> dPsi_dx_j[D];
    for(int d=0; d < D; d++) {
        dPsi_dx_j[d].real = 0;
        dPsi_dx_j[d].imag = 0;
        cj[d].real = atom_j_idx < N ? coords[atom_j_idx*D+d] : 0;
        cj[d].imag = atom_j_idx < N ? coords_tangents[atom_j_idx*D+d] : 0;
    }
    int atomic_radii_idx_j = atom_j_idx < N ? atom_j_idx*2+0 : 0;
    RealType radiusJ = atom_j_idx < N ? gb_params[atomic_radii_idx_j] : 0;

    int scale_factor_idx_j = atom_j_idx < N ? atom_j_idx*2+1 : 0;
    RealType scaleFactorJ = atom_j_idx < N ? gb_params[scale_factor_idx_j] : 0;
    Surreal<RealType> born_radii_j;
    born_radii_j.real = atom_j_idx < N ? born_radii[atom_j_idx].real : 0;
    born_radii_j.imag = atom_j_idx < N ? born_radii[atom_j_idx].imag : 0;

    Surreal<RealType> dPsi_drj(0, 0);
    Surreal<RealType> dPsi_dsj(0, 0);

    const RealType dielectricOffset = dielectric_offset;
    // const RealType cutoffDistance = cutoff;

    // RealType radiusI        = params[atomic_radii_idxs[atomI]];
    RealType offsetRadiusI  = radiusI - dielectricOffset;
    RealType radiusIInverse  = 1/offsetRadiusI;

    Surreal<RealType> lambda(lambda_primal, lambda_tangent);

    for(int round = 0; round < 32; round++) {

        Surreal<RealType> dxs[3];
        Surreal<RealType> r2(0, 0);
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
            r2 += dxs[d]*dxs[d];
        }

        // Surreal<RealType> delta_lambda = lambda_i - lambda_j;
        Surreal<RealType> delta_lambda = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*lambda;
        r2 += delta_lambda * delta_lambda;

        if (atom_j_idx != atom_i_idx && r2.real < cutoff*cutoff && atom_i_idx < N && atom_j_idx < N) {
            
            Surreal<RealType> r = sqrt(r2);
            Surreal<RealType> rInverse = 1/r;

            RealType offsetRadiusJ      = radiusJ - dielectricOffset;
            RealType offsetRadiusJ2     = offsetRadiusJ*offsetRadiusJ;

            RealType scaledRadiusJ      = offsetRadiusJ*scaleFactorJ;
            RealType scaledRadiusJ2     = scaledRadiusJ*scaledRadiusJ;
            Surreal<RealType> rScaledRadiusJ     = r + scaledRadiusJ;

            if (offsetRadiusI < rScaledRadiusJ.real) {

                // (ytz): yes I was insane enough to derive and optimize all of this bullshit by hand

                Surreal<RealType> rSubScaledRadiusJ = r - scaledRadiusJ;
                Surreal<RealType> rSubScaledRadiusJ2 = rSubScaledRadiusJ*rSubScaledRadiusJ;
                Surreal<RealType> rSubScaledRadiusJ3 = rSubScaledRadiusJ2*rSubScaledRadiusJ;

                Surreal<RealType> rSSRJ = abs(rSubScaledRadiusJ);

                Surreal<RealType> l_ij = offsetRadiusI > rSSRJ.real ? Surreal<RealType>(offsetRadiusI, 0) : rSSRJ;
                l_ij = 1/l_ij;
                Surreal<RealType> l_ij2 = l_ij*l_ij;

                Surreal<RealType> u_ij = 1/rScaledRadiusJ;
                Surreal<RealType> u_ij2 = u_ij*u_ij;

                Surreal<RealType> ll_uu = l_ij2 - u_ij2;

                Surreal<RealType> l2rss = l_ij2*rSubScaledRadiusJ/rSSRJ;

                Surreal<RealType> dl_dr = offsetRadiusI > rSSRJ.real ? Surreal<RealType>(0, 0) : -l2rss;
                Surreal<RealType> du_dr = -u_ij*u_ij*rScaledRadiusJ/abs(rScaledRadiusJ);

                Surreal<RealType> t1 = r - scaledRadiusJ2*rInverse;
                Surreal<RealType> dt1_dr = 1 + scaledRadiusJ2*rInverse*rInverse;

                Surreal<RealType> ratio = log(u_ij/l_ij);

                Surreal<RealType> de1 = dl_dr - du_dr;
                Surreal<RealType> de2 = (u_ij*du_dr - l_ij*dl_dr)*t1;
                Surreal<RealType> de3 = -ll_uu*dt1_dr/2;
                Surreal<RealType> de4 = -rInverse*(rInverse*ratio - (du_dr/u_ij - dl_dr/l_ij));
                Surreal<RealType> de = de1 + (de2 + de3 + de4)/2;

                Surreal<RealType> dl_dsj = offsetRadiusI > rSSRJ.real ? Surreal<RealType>(0, 0) : offsetRadiusJ*l2rss;
                Surreal<RealType> du_dsj = -u_ij2*offsetRadiusJ;

                Surreal<RealType> dsj1 = dl_dsj - du_dsj;
                Surreal<RealType> dsjk = u_ij*du_dsj - l_ij*dl_dsj;
                Surreal<RealType> dsj2 = r*dsjk;
                Surreal<RealType> dsj3 = du_dsj/u_ij - dl_dsj/l_ij;
                Surreal<RealType> dsj4 = -scaledRadiusJ2*dsjk;
                Surreal<RealType> dsj5 = scaledRadiusJ*offsetRadiusJ*ll_uu;

                Surreal<RealType> dpsi_dsj = dsj1 + (dsj2 + rInverse*(dsj3 + dsj4 + dsj5))/2;

                Surreal<RealType> dl_dri = offsetRadiusI > rSSRJ.real ? -l_ij2 : Surreal<RealType>(0, 0);

                Surreal<RealType> term = l_ij - u_ij + (ll_uu/4)*(scaledRadiusJ2*rInverse - r) + rInverse*ratio/2;

                Surreal<RealType> dri1 = dl_dri;
                Surreal<RealType> drik = -l_ij*dl_dri;
                Surreal<RealType> dri2 = r*drik;
                Surreal<RealType> dri3 = -scaledRadiusJ2*drik;
                Surreal<RealType> dri4 = -dl_dri/l_ij;

                Surreal<RealType> dterm_dri = dri1 + (dri2 + rInverse*(dri3 + dri4))/2;

                Surreal<RealType> dl_drj = offsetRadiusI > rSSRJ.real ? Surreal<RealType>(0, 0) : scaleFactorJ*l2rss;
                Surreal<RealType> du_drj = -u_ij2*scaleFactorJ;

                Surreal<RealType> drj1 = dl_drj - du_drj;
                Surreal<RealType> drjk = u_ij*du_drj - l_ij*dl_drj;
                Surreal<RealType> drj2 = r*drjk;
                Surreal<RealType> drj3 = scaledRadiusJ*scaleFactorJ*ll_uu;
                Surreal<RealType> drj4 = -scaledRadiusJ2*drjk;
                Surreal<RealType> drj5 = du_drj/u_ij - dl_drj/l_ij;

                Surreal<RealType> dpsi_drj = drj1 + (drj2 + rInverse*(drj3 + drj4 + drj5))/2;

                if (offsetRadiusI < (scaledRadiusJ - r).real) {
                    de += 2*sign(rSubScaledRadiusJ)/rSubScaledRadiusJ2;
                    term += 2*(radiusIInverse - l_ij);
                    dterm_dri += -2*(radiusIInverse*radiusIInverse + dl_dri);
                    dpsi_dsj += -2*dl_dsj;
                    dpsi_drj += -2*dl_drj;
                }

                Surreal<RealType> dpsi_dri = (dterm_dri*offsetRadiusI + term)*born_force_i/2;
                Surreal<RealType> bfor2 = born_force_i*offsetRadiusI/2;
                de *= bfor2;
                dpsi_drj *= bfor2;
                dpsi_dsj *= bfor2;

                // parameter derivatives wrt atomic radii and scale factors

                // (TBD): only shuffle the imaginary parts 
                dPsi_dri += dpsi_dri;
                dPsi_drj += dpsi_drj;
                dPsi_dsj += dpsi_dsj;

                // (TBD): only shuffle the imaginary parts 
                // dPsi_dri_imag += dpsi_dri.imag;
                // dPsi_drj_imag += dpsi_drj.imag;
                // dPsi_dsj_imag += dpsi_dsj.imag;

                for(int d=0; d < D; d++) {
                    dPsi_dx_i[d] += ((dxs[d]/r)*de);
                    dPsi_dx_j[d] -= ((dxs[d]/r)*de);
                }
            }
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE;
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        born_radii_j = __shfl_sync(0xffffffff, born_radii_j, srcLane);
        radiusJ = __shfl_sync(0xffffffff, radiusJ, srcLane);
        scaleFactorJ = __shfl_sync(0xffffffff, scaleFactorJ, srcLane);
        dPsi_drj = __shfl_sync(0xffffffff, dPsi_drj, srcLane);
        dPsi_dsj = __shfl_sync(0xffffffff, dPsi_dsj, srcLane);

        for(int d=0; d < D; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            dPsi_dx_j[d] = __shfl_sync(0xffffffff, dPsi_dx_j[d], srcLane);
        }
        // lambda_j = __shfl_sync(0xffffffff, lambda_j, srcLane);
        lambda_plane_j = __shfl_sync(0xffffffff, lambda_plane_j, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane);
    }

    for(int d=0; d < D; d++) {
        if(atom_i_idx < N) {
            atomicAdd(out_HvP_primals + atom_i_idx*D+d, dPsi_dx_i[d].real);
            atomicAdd(out_HvP_tangents + atom_i_idx*D+d, dPsi_dx_i[d].imag);
        }
        if(atom_j_idx < N) {
            atomicAdd(out_HvP_primals + atom_j_idx*D+d, dPsi_dx_j[d].real);
            atomicAdd(out_HvP_tangents + atom_j_idx*D+d, dPsi_dx_j[d].imag);

        }
    }

    if(atom_i_idx < N) {
        atomicAdd(out_MvP_primals + atomic_radii_idx_i, dPsi_dri.real);
        atomicAdd(out_MvP_tangents + atomic_radii_idx_i, dPsi_dri.imag);
    }

    if(atom_j_idx < N) {
        atomicAdd(out_MvP_primals + atomic_radii_idx_j, dPsi_drj.real);
        atomicAdd(out_MvP_tangents + atomic_radii_idx_j, dPsi_drj.imag);
        atomicAdd(out_MvP_primals + scale_factor_idx_j, dPsi_dsj.real);
        atomicAdd(out_MvP_tangents + scale_factor_idx_j, dPsi_dsj.imag);
    }


}


