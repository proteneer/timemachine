#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
// we need to make this fully deterministic if we want to be able to realiably rematerialize (this also only really matters for forward mode)
// reverse mode we don't care at all
#define WARPSIZE 32

template <typename RealType, int D>
void __global__ k_nonbonded_jvp(
    const int N,
    const double *coords, // maybe Surreal or Real
    const double *coords_tangent, // maybe Surreal or Real
    const double *params, // we do *not* support params tangent, ever!
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

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    Surreal<RealType> ci[D];
    Surreal<RealType> gi[D];
    for(int d=0; d < D; d++) {
        gi[d].real = 0.0;
        gi[d].imag = 0.0;
        ci[d].real = atom_i_idx < N ? coords[atom_i_idx*D+d] : 0;
        ci[d].imag = atom_i_idx < N ? coords_tangent[atom_i_idx*D+d] : 0;
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
    Surreal<RealType> cj[D];
    Surreal<RealType> gj[D];

    for(int d=0; d < D; d++) {
        gj[d].real = 0.0;
        gj[d].imag = 0.0;
        cj[d].real = atom_j_idx < N ? coords[atom_j_idx*D+d] : 0;
        cj[d].imag = atom_j_idx < N ? coords_tangent[atom_j_idx*D+d] : 0;
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

    RealType inv_cutoff = 1/cutoff;

    for(int round = 0; round < 32; round++) {

        RealType dx_real[D];
        for(int d=0; d < D; d++) {
            dx_real[d] = ci[d].real - cj[d].real;
        }

        RealType inv_dij_real = fast_vec_rnorm<RealType, D>(dx_real);

        if(atom_j_idx < atom_i_idx && inv_dij_real > inv_cutoff && atom_j_idx < N && atom_i_idx < N) {

            RealType dx_imag = 0;
            #pragma unroll
            for(int d=0; d < D; d++) {
                dx_imag += dx_real[d]*(ci[d].imag - cj[d].imag);
            }

            Surreal<RealType> inv_dij(inv_dij_real, -dx_imag*inv_dij_real*inv_dij_real*inv_dij_real);

            Surreal<RealType> inv_d2ij = inv_dij*inv_dij;
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

            for(int d=0; d < D; d++) {
                Surreal<RealType> dx = ci[d]-cj[d];
                gi[d] -= (es_grad_prefactor + lj_grad_prefactor)*dx;
                gj[d] += (es_grad_prefactor + lj_grad_prefactor)*dx;
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
        for(size_t d=0; d < D; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
    }

    // we should always accumulate in double precision

    // (ytz): we don't care about deterministic atomics that much when
    // doing reverse mode since we only ever have to do it once.
    for(int d=0; d < D; d++) {
        if(atom_i_idx < N) {
            atomicAdd(grad_coords_tangents + atom_i_idx*D + d, gi[d].imag);            
        }
        if(atom_j_idx < N) {
            atomicAdd(grad_coords_tangents + atom_j_idx*D + d, gj[d].imag);            
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

template <typename RealType, int D>
void __global__ k_nonbonded_inference(
    const int N,
    const double *coords,
    const double *params,
    const int *charge_param_idxs, // [N]
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    unsigned long long *grad_coords) {

    if(blockIdx.y > blockIdx.x) {
        return;
    }

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

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    RealType ci[D];
    RealType gi[D] = {0};
    for(int d=0; d < D; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*D+d] : 0;
    }
    int charge_param_idx_i = atom_i_idx < N ? charge_param_idxs[atom_i_idx] : 0;
    int lj_param_idx_sig_i = atom_i_idx < N ? lj_param_idxs[atom_i_idx*2+0] : 0;
    int lj_param_idx_eps_i = atom_i_idx < N ? lj_param_idxs[atom_i_idx*2+1] : 0;

    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType sig_i = atom_i_idx < N ? params[lj_param_idx_sig_i] : 1;
    RealType eps_i = atom_i_idx < N ? params[lj_param_idx_eps_i] : 0;

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    RealType cj[D];
    RealType gj[D] = {0};
    for(int d=0; d < D; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*D+d] : 0;
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

        RealType dx[D];
        for(int d=0; d < D; d++) {
            dx[d] = ci[d] - cj[d];
        }

        RealType inv_dij = fast_vec_rnorm<RealType, D>(dx);

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

            for(int d=0; d < D; d++) {
                RealType dx = ci[d]- cj[d];
                gi[d] -= (es_grad_prefactor + lj_grad_prefactor) * dx;
                gj[d] += (es_grad_prefactor + lj_grad_prefactor) * dx;
            }
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        for(size_t d=0; d < D; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane); // needs to support real
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
    }

    for(int d=0; d < D; d++) {
        if(atom_i_idx < N) {
            atomicAdd(grad_coords + atom_i_idx*D + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));            
        }
        if(atom_j_idx < N) {
            atomicAdd(grad_coords + atom_j_idx*D + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));            
        }
    }

}


template<typename RealType, int D>
void __global__ k_nonbonded_exclusion_jvp(
    const int E, // number of exclusions
    const double *coords,
    const double *coords_tangent,
    const double *params,
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
    Surreal<RealType> ci[D];
    Surreal<RealType> gi[D] = {Surreal<RealType>(0.0, 0.0)};
    #pragma unroll
    for(int d=0; d < D; d++) {
        gi[d].real = 0;
        gi[d].imag = 0;
        ci[d].real = coords[atom_i_idx*D+d];
        ci[d].imag = coords_tangent[atom_i_idx*D+d];
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
    Surreal<RealType> cj[D];
    Surreal<RealType> gj[D] = {Surreal<RealType>(0.0, 0.0)};
    #pragma unroll
    for(int d=0; d < D; d++) {
        gj[d].real = 0;
        gj[d].imag = 0;
        cj[d].real = coords[atom_j_idx*D+d];
        cj[d].imag = coords_tangent[atom_j_idx*D+d];
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

    Surreal<RealType> d2ij(0.0, 0.0);
    #pragma unroll
    for(int d=0; d < D; d++) {
        Surreal<RealType> dx = ci[d] - cj[d];
        d2ij += dx*dx;
    }

    if(d2ij.real < cutoff*cutoff) {
        // Surreal<RealType> dij = sqrt(d2ij);
        // Surreal<RealType> inv_dij = Surreal<RealType>(1.0)/dij;
        // Surreal<RealType> inv_d2ij = inv_dij*inv_dij;
        // Surreal<RealType> inv_d3ij = inv_d2ij*inv_dij;
        // Surreal<RealType> inv_d4ij = inv_d3ij*inv_dij;
        // Surreal<RealType> inv_d6ij = inv_d4ij*inv_d2ij;
        // Surreal<RealType> inv_d7ij = inv_d4ij*inv_d3ij;

        // Surreal<RealType> es_grad_prefactor = qi*qj*inv_d3ij;

        // // lennard jones force
        // RealType eps_ij = sqrt(eps_i*eps_j);
        // RealType sig_ij = (sig_i+sig_j)/2;

        // RealType sig2 = sig_ij*sig_ij;
        // RealType sig4 = sig2*sig2;
        // RealType sig5 = sig4*sig_ij;
        // RealType sig6 = sig4*sig2;
        // RealType sig12 = sig6*sig6;

        // Surreal<RealType> sig12_rij7 = sig12*inv_d7ij*inv_d7ij;
        // Surreal<RealType> sig6_rij4 = sig6*inv_d4ij*inv_d4ij;
        // Surreal<RealType> lj_grad_prefactor = 24*eps_ij*(sig12_rij7*2 - sig6_rij4);

        Surreal<RealType> inv_dij = rsqrt(d2ij);
        Surreal<RealType> inv_d2ij = 1.0/d2ij;
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
        for(int d=0; d < D; d++) {
            Surreal<RealType> dx = ci[d] - cj[d];
            gi[d] += (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor)*dx;
            gj[d] -= (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor)*dx;
        }

        for(int d=0; d < D; d++) {
            atomicAdd(grad_coords_tangents + atom_i_idx*D + d, gi[d].imag);
            atomicAdd(grad_coords_tangents + atom_j_idx*D + d, gj[d].imag);
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


template<typename RealType, int D>
void __global__ k_nonbonded_exclusion_inference(
    const int E, // number of exclusions
    const double *coords,
    const double *params,
    const int *exclusion_idxs, // [E, 2]pair-list of atoms to be excluded
    const int *charge_scale_idxs, // [E]
    const int *lj_scale_idxs, // [E] 
    const int *charge_param_idxs, // [N]
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    unsigned long long *grad_coords) {

    const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(e_idx >= E) {
        return;
    }

    int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
    RealType ci[D];
    double gi[D] = {0};
    #pragma unroll
    for(int d=0; d < D; d++) {
        ci[d] = coords[atom_i_idx*D+d];
    }
    int charge_param_idx_i = charge_param_idxs[atom_i_idx];
    int lj_param_idx_sig_i = lj_param_idxs[atom_i_idx*2+0];
    int lj_param_idx_eps_i = lj_param_idxs[atom_i_idx*2+1];

    RealType qi = params[charge_param_idx_i];
    RealType sig_i = params[lj_param_idx_sig_i];
    RealType eps_i = params[lj_param_idx_eps_i];

    int atom_j_idx = exclusion_idxs[e_idx*2 + 1];
    RealType cj[D];
    double gj[D] = {0};
    #pragma unroll
    for(int d=0; d < D; d++) {
        cj[d] = coords[atom_j_idx*D+d];
    }

    int charge_param_idx_j = charge_param_idxs[atom_j_idx];
    int lj_param_idx_sig_j = lj_param_idxs[atom_j_idx*2+0];
    int lj_param_idx_eps_j = lj_param_idxs[atom_j_idx*2+1];

    RealType qj = params[charge_param_idx_j];
    RealType sig_j = params[lj_param_idx_sig_j];
    RealType eps_j = params[lj_param_idx_eps_j];

    RealType charge_scale = params[charge_scale_idxs[e_idx]];
    RealType lj_scale = params[lj_scale_idxs[e_idx]];

    RealType d2ij = 0;
    #pragma unroll
    for(int d=0; d < D; d++) {
        RealType dx = ci[d] - cj[d];
        d2ij += dx*dx;
    }

    if(d2ij < cutoff*cutoff) {

        RealType inv_dij = rsqrt(d2ij);
        RealType inv_d2ij = 1/d2ij;
        RealType inv_d3ij = inv_dij*inv_d2ij;
        RealType es_grad_prefactor = qi*qj*inv_d3ij;

        // lennard jones force
        RealType eps_ij = sqrt(eps_i * eps_j);
        RealType sig_ij = (sig_i + sig_j)/2;

        RealType sig2_inv_d2ij = sig_ij*sig_ij/d2ij; // avoid using inv_dij as much as we can due to loss of precision
        RealType sig4_inv_d4ij = sig2_inv_d2ij*sig2_inv_d2ij;
        RealType sig6_inv_d6ij = sig4_inv_d4ij*sig2_inv_d2ij;
        RealType sig6_inv_d8ij = sig6_inv_d6ij*inv_d2ij;
        RealType sig8_inv_d8ij = sig4_inv_d4ij*sig4_inv_d4ij;
        RealType sig12_inv_d12ij = sig8_inv_d8ij*sig4_inv_d4ij;
        RealType sig12_inv_d14ij = sig12_inv_d12ij*inv_d2ij;

        // RealType lj_grad_prefactor = 24*eps_ij*(sig12rij7*2 - sig6rij4);
        RealType lj_grad_prefactor = 24*eps_ij*sig12_inv_d14ij*2 - 24*eps_ij*sig6_inv_d8ij;

        #pragma unroll
        for(int d=0; d < D; d++) {
            RealType dx = ci[d] - cj[d];
            gi[d] += (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor)*dx;
            gj[d] -= (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor)*dx;
        }

        for(int d=0; d < D; d++) {
            atomicAdd(grad_coords + atom_i_idx*D + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));
            atomicAdd(grad_coords + atom_j_idx*D + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));
        }  

    }

}
