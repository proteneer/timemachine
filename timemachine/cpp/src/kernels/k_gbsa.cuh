#include <stdexcept>
#include <iostream> 
#include "../fixed_point.hpp"
#include "surreal.cuh"
#include "kernel_utils.cuh"

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

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;

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