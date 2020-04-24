#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"

#include "k_periodic_utils.cuh"
// we need to make this fully deterministic if we want to be able to realiably rematerialize (this also only really matters for forward mode)
// reverse mode we don't care at all
#define WARPSIZE 32

template <typename RealType>
void __global__ k_nonbonded_jvp(
    const int N,
    const double *coords, // maybe Surreal or Real
    const double *coords_tangent, // maybe Surreal or Real
    const double *params, // we do *not* support params tangent, ever!
    const double lambda,
    const double lambda_tangent,
    const int *lambda_idxs,
    const int *charge_param_idxs,
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    double *grad_coords_tangents, // *always* int64 for accumulation purposes, but we discard the primals
    double *grad_params_tangents) {

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
    Surreal<RealType> lambda_i;
    if(atom_i_idx < N) {
        if(lambda_idxs[atom_i_idx] == 0) {
            lambda_i = Surreal<RealType>(0, 0);
        } else if(lambda_idxs[atom_i_idx] == 1) {
            lambda_i = Surreal<RealType>(lambda, lambda_tangent);
        } else if(lambda_idxs[atom_i_idx] == -1) {
            lambda_i = Surreal<RealType>(cutoff + lambda, lambda_tangent);
        }        
    } else {
        lambda_i = Surreal<RealType>(0, 0);
    }

    Surreal<RealType> ci[3];
    Surreal<RealType> gi[3];

    for(int d=0; d < 3; d++) {
        gi[d].real = 0.0;
        gi[d].imag = 0.0;
        ci[d].real = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
        ci[d].imag = atom_i_idx < N ? coords_tangent[atom_i_idx*3+d] : 0;
    }
    int charge_param_idx_i = atom_i_idx < N ? charge_param_idxs[atom_i_idx] : 0;
    int lj_param_idx_sig_i = atom_i_idx < N ? lj_param_idxs[atom_i_idx*2+0] : 0;
    int lj_param_idx_eps_i = atom_i_idx < N ? lj_param_idxs[atom_i_idx*2+1] : 0;

    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType sig_i = atom_i_idx < N ? params[lj_param_idx_sig_i] : 1;
    RealType eps_i = atom_i_idx < N ? params[lj_param_idx_eps_i] : 0;

    Surreal<RealType> g_qi(0.0, 0.0);
    Surreal<RealType> g_sigi(0.0, 0.0);
    Surreal<RealType> g_epsi(0.0, 0.0);

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    Surreal<RealType> lambda_j;

    if(atom_j_idx < N) {
        if(lambda_idxs[atom_j_idx] == 0) {
            lambda_j = Surreal<RealType>(0, 0);
        } else if(lambda_idxs[atom_j_idx] == 1) {
            lambda_j = Surreal<RealType>(lambda, lambda_tangent);
        } else if(lambda_idxs[atom_j_idx] == -1) {
            lambda_j = Surreal<RealType>(cutoff + lambda, lambda_tangent);
        }        
    } else {
        lambda_j = Surreal<RealType>(0, 0);
    }

    Surreal<RealType> cj[3];
    Surreal<RealType> gj[3];

    for(int d=0; d < 3; d++) {
        gj[d].real = 0.0;
        gj[d].imag = 0.0;
        cj[d].real = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
        cj[d].imag = atom_j_idx < N ? coords_tangent[atom_j_idx*3+d] : 0;
    }

    int charge_param_idx_j = atom_j_idx < N ? charge_param_idxs[atom_j_idx] : 0;
    int lj_param_idx_sig_j = atom_j_idx < N ? lj_param_idxs[atom_j_idx*2+0] : 0;
    int lj_param_idx_eps_j = atom_j_idx < N ? lj_param_idxs[atom_j_idx*2+1] : 0;

    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    RealType sig_j = atom_j_idx < N ? params[lj_param_idx_sig_j] : 1;
    RealType eps_j = atom_j_idx < N ? params[lj_param_idx_eps_j] : 0;

    Surreal<RealType> g_qj(0.0, 0.0);
    Surreal<RealType> g_sigj(0.0, 0.0);
    Surreal<RealType> g_epsj(0.0, 0.0);

    for(int round = 0; round < 32; round++) {

        Surreal<RealType> dxs[4];
        Surreal<RealType> d2ij(0,0);
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
            d2ij += dxs[d]*dxs[d];
        }

        Surreal<RealType> delta_lambda = apply_delta(lambda_i - lambda_j, 2*cutoff);
        dxs[3] = delta_lambda; 
        d2ij += delta_lambda * delta_lambda;

        if(atom_j_idx < atom_i_idx && d2ij.real < cutoff*cutoff && atom_j_idx < N && atom_i_idx < N) {

            Surreal<RealType> inv_dij = rsqrt(d2ij);
            Surreal<RealType> inv_d2ij = 1/d2ij;
            Surreal<RealType> inv_d4ij = inv_d2ij*inv_d2ij;
            Surreal<RealType> inv_d6ij = inv_d4ij*inv_d2ij;
            Surreal<RealType> inv_d8ij = inv_d4ij*inv_d4ij;
            Surreal<RealType> inv_d14ij = inv_d8ij*inv_d6ij;
            Surreal<RealType> inv_d3ij = inv_d2ij*inv_dij;
            Surreal<RealType> es_grad_prefactor = qi*qj*inv_d3ij;

            // lennard jones force
            RealType eps_ij = overloaded_sqrt(eps_i*eps_j);
            RealType sig_ij = (sig_i+sig_j)/2;

            RealType sig2 = sig_ij*sig_ij;
            RealType sig4 = sig2*sig2;
            RealType sig5 = sig4*sig_ij;
            RealType sig6 = sig4*sig2;

            Surreal<RealType> sig6_inv_d6ij = sig6*inv_d6ij;
            Surreal<RealType> sig6_inv_d8ij = sig6*inv_d8ij;

            Surreal<RealType> lj_grad_prefactor = 24*eps_ij*sig6_inv_d8ij*(sig6_inv_d6ij*2 - 1);

            for(int d=0; d < 3; d++) {
                gi[d] -= (es_grad_prefactor + lj_grad_prefactor) * dxs[d];
                gj[d] += (es_grad_prefactor + lj_grad_prefactor) * dxs[d];
            }

            // Charge
            g_qi += qj*inv_dij;
            g_qj += qi*inv_dij;

            // vDw
            Surreal<RealType> eps_grad = 2*sig6_inv_d6ij*(sig6_inv_d6ij-1)/eps_ij;
            g_epsi += eps_grad*eps_j;
            g_epsj += eps_grad*eps_i;

            Surreal<RealType> sig_grad = 12*eps_ij*sig5*inv_d6ij*(2*sig6_inv_d6ij-1);
            g_sigi += sig_grad;
            g_sigj += sig_grad;
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        g_qj = __shfl_sync(0xffffffff, g_qj, srcLane);
        g_sigj = __shfl_sync(0xffffffff, g_sigj, srcLane);
        g_epsj = __shfl_sync(0xffffffff, g_epsj, srcLane);
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        for(size_t d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
        lambda_j = __shfl_sync(0xffffffff, lambda_j, srcLane);
    }

    // we should always accumulate in double precision

    // (ytz): we don't care about deterministic atomics that much when
    // doing reverse mode since we only ever have to do it once.
    for(int d=0; d < 3; d++) {
        if(atom_i_idx < N) {
            atomicAdd(grad_coords_tangents + atom_i_idx*3 + d, gi[d].imag);            
        }
        if(atom_j_idx < N) {
            atomicAdd(grad_coords_tangents + atom_j_idx*3 + d, gj[d].imag);            
        }
    }  

    if(atom_i_idx < N) {
        atomicAdd(grad_params_tangents + charge_param_idx_i, g_qi.imag);        
        atomicAdd(grad_params_tangents + lj_param_idx_sig_i, g_sigi.imag);
        atomicAdd(grad_params_tangents + lj_param_idx_eps_i, g_epsi.imag);
    }

    if(atom_j_idx < N) {
        atomicAdd(grad_params_tangents + charge_param_idx_j, g_qj.imag);
        atomicAdd(grad_params_tangents + lj_param_idx_sig_j, g_sigj.imag);
        atomicAdd(grad_params_tangents + lj_param_idx_eps_j, g_epsj.imag);
    }
}



template<typename RealType>
void __global__ k_nonbonded_exclusion_jvp(
    const int E, // number of exclusions
    const double *coords,
    const double *coords_tangent,
    const double *params,
    const double lambda,
    const double lambda_tangent,
    const int *lambda_idxs,
    const int *exclusion_idxs, // [E, 2]pair-list of atoms to be excluded
    const int *charge_scale_idxs, // [E]
    const int *lj_scale_idxs, // [E] 
    const int *charge_param_idxs, // [N]
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    double *grad_coords_tangents, // *always* int64 for accumulation purposes, but we discard the primals
    double *grad_params_tangents) {

    const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(e_idx >= E) {
        return;
    }

    int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
    Surreal<RealType> lambda_i;

    if(lambda_idxs[atom_i_idx] == 0) {
        lambda_i = Surreal<RealType>(0, 0);
    } else if(lambda_idxs[atom_i_idx] == 1) {
        lambda_i = Surreal<RealType>(lambda, lambda_tangent);
    } else if(lambda_idxs[atom_i_idx] == -1) {
        lambda_i = Surreal<RealType>(cutoff + lambda, lambda_tangent);
    }        

    Surreal<RealType> ci[3];
    Surreal<RealType> gi[3] = {Surreal<RealType>(0.0, 0.0)};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        gi[d].real = 0;
        gi[d].imag = 0;
        ci[d].real = coords[atom_i_idx*3+d];
        ci[d].imag = coords_tangent[atom_i_idx*3+d];
    }
    int charge_param_idx_i = charge_param_idxs[atom_i_idx];
    int lj_param_idx_sig_i = lj_param_idxs[atom_i_idx*2+0];
    int lj_param_idx_eps_i = lj_param_idxs[atom_i_idx*2+1];

    RealType qi = params[charge_param_idx_i];
    RealType sig_i = params[lj_param_idx_sig_i];
    RealType eps_i = params[lj_param_idx_eps_i];

    Surreal<RealType> g_qi(0.0, 0.0);
    Surreal<RealType> g_sigi(0.0, 0.0);
    Surreal<RealType> g_epsi(0.0, 0.0);

    int atom_j_idx = exclusion_idxs[e_idx*2 + 1];
    Surreal<RealType> lambda_j;

    if(lambda_idxs[atom_j_idx] == 0) {
        lambda_j = Surreal<RealType>(0, 0);
    } else if(lambda_idxs[atom_j_idx] == 1) {
        lambda_j = Surreal<RealType>(lambda, lambda_tangent);
    } else if(lambda_idxs[atom_j_idx] == -1) {
        lambda_j = Surreal<RealType>(cutoff + lambda, lambda_tangent);
    }        

    Surreal<RealType> cj[3];
    Surreal<RealType> gj[3] = {Surreal<RealType>(0.0, 0.0)};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        gj[d].real = 0;
        gj[d].imag = 0;
        cj[d].real = coords[atom_j_idx*3+d];
        cj[d].imag = coords_tangent[atom_j_idx*3+d];
    }

    int charge_param_idx_j = charge_param_idxs[atom_j_idx];
    int lj_param_idx_sig_j = lj_param_idxs[atom_j_idx*2+0];
    int lj_param_idx_eps_j = lj_param_idxs[atom_j_idx*2+1];

    RealType qj = params[charge_param_idx_j];
    RealType sig_j = params[lj_param_idx_sig_j];
    RealType eps_j = params[lj_param_idx_eps_j];

    Surreal<RealType> g_qj(0.0, 0.0);
    Surreal<RealType> g_sigj(0.0, 0.0);
    Surreal<RealType> g_epsj(0.0, 0.0);

    int charge_scale_idx = charge_scale_idxs[e_idx];
    RealType charge_scale = params[charge_scale_idx];
    
    int lj_scale_idx = lj_scale_idxs[e_idx];
    RealType lj_scale = params[lj_scale_idx];

    Surreal<RealType> dxs[4];
    Surreal<RealType> d2ij(0.0, 0.0);
    #pragma unroll
    for(int d=0; d < 3; d++) {
        Surreal<RealType> dx = ci[d] - cj[d];
        dxs[d] = dx;
        d2ij += dx*dx;
    }

    Surreal<RealType> delta_lambda = apply_delta(lambda_i - lambda_j, 2*cutoff);
    dxs[3] = delta_lambda; 
    d2ij += delta_lambda * delta_lambda;

    if(d2ij.real < cutoff*cutoff) {

        Surreal<RealType> inv_dij = rsqrt(d2ij);
        Surreal<RealType> inv_d2ij = 1/d2ij;
        Surreal<RealType> inv_d3ij = inv_dij*inv_d2ij;
        Surreal<RealType> inv_d4ij = inv_d2ij*inv_d2ij;
        Surreal<RealType> inv_d6ij = inv_d4ij*inv_d2ij;
        Surreal<RealType> inv_d8ij = inv_d4ij*inv_d4ij;
        Surreal<RealType> es_grad_prefactor = qi*qj*inv_d3ij;

        // lennard jones force
        RealType eps_ij = sqrt(eps_i * eps_j);
        RealType sig_ij = (sig_i + sig_j)/2;

        RealType sig2 = sig_ij*sig_ij;
        RealType sig4 = sig2*sig2;
        RealType sig5 = sig4*sig_ij;
        RealType sig6 = sig4*sig2;

        Surreal<RealType> sig6_inv_d6ij = sig6*inv_d6ij;
        Surreal<RealType> sig6_inv_d8ij = sig6*inv_d8ij;

        Surreal<RealType> lj_grad_prefactor = 24*eps_ij*sig6_inv_d8ij*(sig6_inv_d6ij*2 - 1.0);

        #pragma unroll
        for(int d=0; d < 3; d++) {
            Surreal<RealType> dx = ci[d] - cj[d];
            gi[d] += (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor)*dx;
            gj[d] -= (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor)*dx;
        }

        for(int d=0; d < 3; d++) {
            atomicAdd(grad_coords_tangents + atom_i_idx*3 + d, gi[d].imag);
            atomicAdd(grad_coords_tangents + atom_j_idx*3 + d, gj[d].imag);
        }  

        // dE_dp 
        // Charge
        g_qi += qj*inv_dij;
        g_qj += qi*inv_dij;

        // vDw
        Surreal<RealType> eps_grad = 4*(sig6*inv_d6ij-1.0)*sig6*inv_d6ij;
        g_epsi += eps_grad*eps_j/(2*eps_ij);
        g_epsj += eps_grad*eps_i/(2*eps_ij);
        Surreal<RealType> sig_grad = 24*eps_ij*(2*sig6*inv_d6ij-1.0)*(sig5*inv_d6ij);
        g_sigi += sig_grad/2;
        g_sigj += sig_grad/2;

        atomicAdd(grad_params_tangents + charge_param_idx_i, -charge_scale*g_qi.imag);
        atomicAdd(grad_params_tangents + charge_param_idx_j, -charge_scale*g_qj.imag);

        atomicAdd(grad_params_tangents + lj_param_idx_sig_i, -lj_scale*g_sigi.imag);
        atomicAdd(grad_params_tangents + lj_param_idx_sig_j, -lj_scale*g_sigj.imag);

        atomicAdd(grad_params_tangents + lj_param_idx_eps_i, -lj_scale*g_epsi.imag);
        atomicAdd(grad_params_tangents + lj_param_idx_eps_j, -lj_scale*g_epsj.imag);

        // now do derivatives of the scales, which are just the negative unscaled energies!
        Surreal<RealType> charge_scale_grad = qi*qj*inv_dij; 
        Surreal<RealType> lj_scale_grad = 4*eps_ij*(sig6*inv_d6ij-1.0)*sig6*inv_d6ij;

        atomicAdd(grad_params_tangents + charge_scale_idx, -charge_scale_grad.imag);
        atomicAdd(grad_params_tangents + lj_scale_idx, -lj_scale_grad.imag);

    }

}


// assume D = 3
// template <typename RealType, int D>
template <typename RealType>
void __global__ k_nonbonded_inference(
    const int N,
    const double *coords,
    const double *params,
    const double lambda,
    const int *lambda_idxs,
    const int *charge_param_idxs, // [N]
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    unsigned long long *grad_coords,
    double *du_dl) {

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

    int atom_i_idx = blockIdx.x*32 + threadIdx.x;
    RealType lambda_i;

    if(atom_i_idx < N) {
        if(lambda_idxs[atom_i_idx] == 0) {
            lambda_i = 0;
        } else if(lambda_idxs[atom_i_idx] == 1) {
            lambda_i = lambda;
        } else if(lambda_idxs[atom_i_idx] == -1) {
            lambda_i = cutoff + lambda;
        }        
    } else {
        lambda_i = 0;
    }

    RealType ci[3];
    RealType gi[3] = {0};
    RealType du_dl_i = 0;
    for(int d=0; d < 3; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
    }
    
    int charge_param_idx_i = atom_i_idx < N ? charge_param_idxs[atom_i_idx] : 0;
    int lj_param_idx_sig_i = atom_i_idx < N ? lj_param_idxs[atom_i_idx*2+0] : 0;
    int lj_param_idx_eps_i = atom_i_idx < N ? lj_param_idxs[atom_i_idx*2+1] : 0;

    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType sig_i = atom_i_idx < N ? params[lj_param_idx_sig_i] : 1;
    RealType eps_i = atom_i_idx < N ? params[lj_param_idx_eps_i] : 0;

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    RealType lambda_j;

    if(atom_j_idx < N) {
        if(lambda_idxs[atom_j_idx] == 0) {
            lambda_j = 0;
        } else if(lambda_idxs[atom_j_idx] == 1) {
            lambda_j = lambda;
        } else if(lambda_idxs[atom_j_idx] == -1) {
            lambda_j = cutoff + lambda;
        }   
    } else {
        lambda_j = 0;
    }

    RealType cj[3];
    RealType gj[3] = {0};
    RealType du_dl_j = 0;
    for(int d=0; d < 3; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
    }
    int charge_param_idx_j = atom_j_idx < N ? charge_param_idxs[atom_j_idx] : 0;
    int lj_param_idx_sig_j = atom_j_idx < N ? lj_param_idxs[atom_j_idx*2+0] : 0;
    int lj_param_idx_eps_j = atom_j_idx < N ? lj_param_idxs[atom_j_idx*2+1] : 0;

    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    RealType sig_j = atom_j_idx < N ? params[lj_param_idx_sig_j] : 1;
    RealType eps_j = atom_j_idx < N ? params[lj_param_idx_eps_j] : 0;

    RealType inv_cutoff = 1/cutoff;

    // In inference mode, we don't care about gradients with respect to parameters.
    for(int round = 0; round < 32; round++) {

        RealType dxs[4];
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
        }

        RealType delta_lambda = lambda_i - lambda_j;
        dxs[3] = apply_delta(delta_lambda, 2*cutoff); 

        RealType inv_dij = fast_vec_rnorm<RealType, 4>(dxs);

        if(atom_j_idx < atom_i_idx && inv_dij > inv_cutoff && atom_j_idx < N && atom_i_idx < N) {

            RealType inv_d2ij = inv_dij*inv_dij;
            RealType inv_d3ij = inv_dij*inv_d2ij;
            RealType inv_d4ij = inv_d2ij*inv_d2ij;
            RealType inv_d6ij = inv_d4ij*inv_d2ij;
            RealType inv_d8ij = inv_d4ij*inv_d4ij;
            RealType es_grad_prefactor = qi*qj*inv_d3ij;

            // lennard jones force
            RealType eps_ij = overloaded_sqrt(eps_i * eps_j);
            RealType sig_ij = (sig_i + sig_j)/2;

            RealType sig2 = sig_ij*sig_ij;
            RealType sig4 = sig2*sig2;
            RealType sig6 = sig4*sig2;

            RealType sig6_inv_d6ij = sig6*inv_d6ij;
            RealType sig6_inv_d8ij = sig6*inv_d8ij;

            RealType lj_grad_prefactor = 24*eps_ij*sig6_inv_d8ij*(sig6_inv_d6ij*2 - 1);

            for(int d=0; d < 3; d++) {
                gi[d] -= (es_grad_prefactor + lj_grad_prefactor) *  dxs[d];
                gj[d] += (es_grad_prefactor + lj_grad_prefactor) *  dxs[d];
            }

            // this technically should be if lambda_idxs[i] == 0 and lamba_idxs[j] == 0
            // however, they both imply that delta_lambda = 0, so dxs[3] == 0, simplifying the equation
            RealType dw_i = (lambda_i == 0) ? 0 : 1;
            RealType dw_j = (lambda_j == 0) ? 0 : 1;

            du_dl_i -= (es_grad_prefactor + lj_grad_prefactor) * dxs[3] * dw_i;
            du_dl_j += (es_grad_prefactor + lj_grad_prefactor) * dxs[3] * dw_j;
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        for(size_t d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane); // needs to support real
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
        lambda_j = __shfl_sync(0xffffffff, lambda_j, srcLane);
        du_dl_j = __shfl_sync(0xffffffff, du_dl_j, srcLane);
    }

    for(int d=0; d < 3; d++) {
        if(atom_i_idx < N) {
            atomicAdd(grad_coords + atom_i_idx*3 + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));            
        }
        if(atom_j_idx < N) {
            atomicAdd(grad_coords + atom_j_idx*3 + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));            
        }
    }

    atomicAdd(du_dl, du_dl_i + du_dl_j);

}

template<typename RealType>
void __global__ k_nonbonded_exclusion_inference(
    const int E, // number of exclusions
    const double *coords,
    const double *params,
    const double lambda,
    const int *lambda_idxs,
    const int *exclusion_idxs, // [E, 2]pair-list of atoms to be excluded
    const int *charge_scale_idxs, // [E]
    const int *lj_scale_idxs, // [E] 
    const int *charge_param_idxs, // [N]
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    unsigned long long *grad_coords,
    double *du_dl) {

    const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(e_idx >= E) {
        return;
    }

    int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
    RealType du_dl_i = 0;
    RealType lambda_i;

    if(lambda_idxs[atom_i_idx] == 0) {
        lambda_i = 0;
    } else if(lambda_idxs[atom_i_idx] == 1) {
        lambda_i = lambda;
    } else if(lambda_idxs[atom_i_idx] == -1) {
        lambda_i = cutoff + lambda;
    }

    RealType ci[3];
    double gi[3] = {0};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        ci[d] = coords[atom_i_idx*3+d];
    }
    int charge_param_idx_i = charge_param_idxs[atom_i_idx];
    int lj_param_idx_sig_i = lj_param_idxs[atom_i_idx*2+0];
    int lj_param_idx_eps_i = lj_param_idxs[atom_i_idx*2+1];

    RealType qi = params[charge_param_idx_i];
    RealType sig_i = params[lj_param_idx_sig_i];
    RealType eps_i = params[lj_param_idx_eps_i];

    int atom_j_idx = exclusion_idxs[e_idx*2 + 1];
    RealType du_dl_j = 0;
    RealType lambda_j;

    if(lambda_idxs[atom_j_idx] == 0) {
        lambda_j = 0;
    } else if(lambda_idxs[atom_j_idx] == 1) {
        lambda_j = lambda;
    } else if(lambda_idxs[atom_j_idx] == -1) {
        lambda_j = cutoff + lambda;
    }

    RealType cj[3];
    double gj[3] = {0};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        cj[d] = coords[atom_j_idx*3+d];
    }

    int charge_param_idx_j = charge_param_idxs[atom_j_idx];
    int lj_param_idx_sig_j = lj_param_idxs[atom_j_idx*2+0];
    int lj_param_idx_eps_j = lj_param_idxs[atom_j_idx*2+1];

    RealType qj = params[charge_param_idx_j];
    RealType sig_j = params[lj_param_idx_sig_j];
    RealType eps_j = params[lj_param_idx_eps_j];

    RealType charge_scale = params[charge_scale_idxs[e_idx]];
    RealType lj_scale = params[lj_scale_idxs[e_idx]];

    RealType dxs[4];
    for(int d=0; d < 3; d++) {
        dxs[d] = ci[d] - cj[d];
    }

    RealType delta_lambda = lambda_i - lambda_j;
    dxs[3] = apply_delta(delta_lambda, 2*cutoff); 

    RealType inv_dij = fast_vec_rnorm<RealType, 4>(dxs);
    RealType inv_cutoff = 1/cutoff;

    if(inv_dij > inv_cutoff) {

        RealType inv_d2ij = inv_dij*inv_dij;
        RealType inv_d3ij = inv_dij*inv_d2ij;
        RealType es_grad_prefactor = qi*qj*inv_d3ij;

        // lennard jones force
        RealType eps_ij = sqrt(eps_i * eps_j);
        RealType sig_ij = (sig_i + sig_j)/2;

        RealType sig2_inv_d2ij = sig_ij*sig_ij*inv_d2ij; // avoid using inv_dij as much as we can due to loss of precision
        RealType sig4_inv_d4ij = sig2_inv_d2ij*sig2_inv_d2ij;
        RealType sig6_inv_d6ij = sig4_inv_d4ij*sig2_inv_d2ij;
        RealType sig6_inv_d8ij = sig6_inv_d6ij*inv_d2ij;
        RealType sig8_inv_d8ij = sig4_inv_d4ij*sig4_inv_d4ij;
        RealType sig12_inv_d12ij = sig8_inv_d8ij*sig4_inv_d4ij;
        RealType sig12_inv_d14ij = sig12_inv_d12ij*inv_d2ij;

        // RealType lj_grad_prefactor = 24*eps_ij*(sig12rij7*2 - sig6rij4);
        RealType lj_grad_prefactor = 24*eps_ij*sig12_inv_d14ij*2 - 24*eps_ij*sig6_inv_d8ij;

        #pragma unroll
        for(int d=0; d < 3; d++) {
            gi[d] += (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor)*dxs[d];
            gj[d] -= (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor)*dxs[d];
        }

        RealType dw_i = (lambda_i == 0) ? 0 : 1;
        RealType dw_j = (lambda_j == 0) ? 0 : 1;

        du_dl_i += (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor) * dxs[3] * dw_i;
        du_dl_j -= (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor) * dxs[3] * dw_j;

        for(int d=0; d < 3; d++) {
            atomicAdd(grad_coords + atom_i_idx*3 + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));
            atomicAdd(grad_coords + atom_j_idx*3 + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));
        }  

        atomicAdd(du_dl, du_dl_i + du_dl_j);

    }

}
