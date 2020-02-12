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


// template <typename RealType, int D>
// void __global__ k_compute_born_first_loop(
//     const int N,
//     const double* coords,
//     const double* params,
//     const int* charge_param_idxs,
//     const double* born_radii,
//     const double prefactor,
//     const double cutoff,
//     double *bornForces,
//     double *out_forces,
//     double *dU_dp) {

//     if(blockIdx.y > blockIdx.x) {
//         return;
//     }

//     // RealType block_d2ij = 0; 
//     // for(int d=0; d < D; d++) {
//     //     RealType block_row_ctr = block_bounds_ctr[blockIdx.x*D+d];
//     //     RealType block_col_ctr = block_bounds_ctr[blockIdx.y*D+d];
//     //     RealType block_row_ext = block_bounds_ext[blockIdx.x*D+d];
//     //     RealType block_col_ext = block_bounds_ext[blockIdx.y*D+d];
//     //     RealType dx = max(0.0, fabs(block_row_ctr-block_col_ctr) - (block_row_ext+block_col_ext));
//     //     block_d2ij += dx*dx;
//     // }

//     // if(block_d2ij > cutoff*cutoff) {
//     //     return;
//     // }

//     int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
//     RealType ci[D];
//     RealType gi[D] = {0};
//     for(int d=0; d < D; d++) {
//         ci[d] = atom_i_idx < N ? coords[atom_i_idx*D+d] : 0;
//     }
//     int charge_param_idx_i = atom_i_idx < N ? charge_param_idxs[atom_i_idx] : 0;
//     int lj_param_idx_sig_i = atom_i_idx < N ? lj_param_idxs[atom_i_idx*2+0] : 0;
//     int lj_param_idx_eps_i = atom_i_idx < N ? lj_param_idxs[atom_i_idx*2+1] : 0;

//     RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
//     RealType sig_i = atom_i_idx < N ? params[lj_param_idx_sig_i] : 1;
//     RealType eps_i = atom_i_idx < N ? params[lj_param_idx_eps_i] : 0;

//     int atom_j_idx = blockIdx.y*32 + threadIdx.x;
//     RealType cj[D];
//     RealType gj[D] = {0};
//     for(int d=0; d < D; d++) {
//         cj[d] = atom_j_idx < N ? coords[atom_j_idx*D+d] : 0;
//     }
//     int charge_param_idx_j = atom_j_idx < N ? charge_param_idxs[atom_j_idx] : 0;
//     int lj_param_idx_sig_j = atom_j_idx < N ? lj_param_idxs[atom_j_idx*2+0] : 0;
//     int lj_param_idx_eps_j = atom_j_idx < N ? lj_param_idxs[atom_j_idx*2+1] : 0;

//     RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
//     RealType sig_j = atom_j_idx < N ? params[lj_param_idx_sig_j] : 1;
//     RealType eps_j = atom_j_idx < N ? params[lj_param_idx_eps_j] : 0;

//     RealType inv_cutoff = 1/cutoff;

//     // In inference mode, we don't care about gradients with respect to parameters.
//     for(int round = 0; round < 32; round++) {

//         RealType dx[D];
//         for(int d=0; d < D; d++) {
//             dx[d] = ci[d] - cj[d];
//         }

//         RealType inv_dij = fast_vec_rnorm<RealType, D>(dx);

//         if(atom_j_idx < atom_i_idx && inv_dij > inv_cutoff && atom_j_idx < N && atom_i_idx < N) {

//             RealType inv_d2ij = inv_dij*inv_dij;
//             RealType inv_d3ij = inv_dij*inv_d2ij;
//             RealType inv_d4ij = inv_d2ij*inv_d2ij;
//             RealType inv_d6ij = inv_d4ij*inv_d2ij;
//             RealType inv_d8ij = inv_d4ij*inv_d4ij;
//             RealType es_grad_prefactor = qi*qj*inv_d3ij;

//             // lennard jones force
//             RealType eps_ij = overloaded_sqrt(eps_i * eps_j);
//             RealType sig_ij = (sig_i + sig_j)/2;

//             RealType sig2 = sig_ij*sig_ij;
//             RealType sig4 = sig2*sig2;
//             RealType sig6 = sig4*sig2;

//             RealType sig6_inv_d6ij = sig6*inv_d6ij;
//             RealType sig6_inv_d8ij = sig6*inv_d8ij;

//             RealType lj_grad_prefactor = 24*eps_ij*sig6_inv_d8ij*(sig6_inv_d6ij*2 - 1);

//             for(int d=0; d < D; d++) {
//                 RealType dx = ci[d]- cj[d];
//                 gi[d] -= (es_grad_prefactor + lj_grad_prefactor) * dx;
//                 gj[d] += (es_grad_prefactor + lj_grad_prefactor) * dx;
//             }
//         }

//         const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
//         atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
//         qj = __shfl_sync(0xffffffff, qj, srcLane);
//         sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
//         eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
//         for(size_t d=0; d < D; d++) {
//             cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane); // needs to support real
//             gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
//         }
//     }

//     for(int d=0; d < D; d++) {
//         if(atom_i_idx < N) {
//             atomicAdd(grad_coords + atom_i_idx*D + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));            
//         }
//         if(atom_j_idx < N) {
//             atomicAdd(grad_coords + atom_j_idx*D + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));            
//         }
//     }

// }
