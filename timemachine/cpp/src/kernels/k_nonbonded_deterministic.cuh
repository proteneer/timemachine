#include "surreal.cuh"
#include "../fixed_point.hpp"
// we need to make this fully deterministic if we want to be able to realiably rematerialize (this also only really matters for forward mode)
// reverse mode we don't care at all
#define WARPSIZE 32


template <typename RealType>
void __global__ k_find_block_bounds(
    const int N,
    const int D,
    const int T,
    const RealType *coords,
    RealType *block_bounds_ctr,
    RealType *block_bounds_ext) {

    const int tile_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(tile_idx >= T) {
        return;
    }

    for(int d=0; d < D; d++) {
        RealType ci_min =  9999999;
        RealType ci_max = -9999999;
        for(int i=0; i < WARPSIZE; i++) {
            int atom_i_idx = tile_idx*WARPSIZE + i;
            if(atom_i_idx < N) {
                RealType ci = coords[atom_i_idx*D + d];
                ci_min = ci < ci_min ? ci : ci_min;
                ci_max = ci > ci_max ? ci : ci_max;                
            }
        }
        block_bounds_ctr[tile_idx*D+d] = (ci_max + ci_min)/2.0;
        block_bounds_ext[tile_idx*D+d] = ci_max - ci_min;
    }

}

template <typename RealType, int D>
void __global__ k_nonbonded_jvp(
    const int N,
    const RealType *coords, // maybe Surreal or Real
    const RealType *coords_tangent, // maybe Surreal or Real
    const RealType *params, // we do *not* support params tangent, ever!
    const int *charge_param_idxs,
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    const RealType *block_bounds_ctr,
    const RealType *block_bounds_ext,
    RealType *grad_coords_tangents, // *always* int64 for accumulation purposes, but we discard the primals
    RealType *grad_params_tangents) {

    if(blockIdx.y > blockIdx.x) {
        return;
    }

    RealType block_d2ij = 0; 
    for(int d=0; d < D; d++) {
        RealType block_row_ctr = block_bounds_ctr[blockIdx.x*D+d];
        RealType block_col_ctr = block_bounds_ctr[blockIdx.y*D+d];
        RealType block_row_ext = block_bounds_ext[blockIdx.x*D+d];
        RealType block_col_ext = block_bounds_ext[blockIdx.y*D+d];
        RealType dx = max(0.0, abs(block_row_ctr-block_col_ctr) - (block_row_ext+block_col_ext));
        block_d2ij += dx*dx;
    }

    if(block_d2ij > cutoff*cutoff) {
        return;
    }

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    Surreal<RealType> ci[D] = {0};
    Surreal<RealType> gi[D] = {0};
    #pragma unroll
    for(int d=0; d < D; d++) {
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
    Surreal<RealType> cj[D] = {0};
    Surreal<RealType> gj[D] = {0};

    #pragma unroll
    for(int d=0; d < D; d++) {
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

    for(int round = 0; round < 32; round++) {

        Surreal<RealType> d2ij = 0;
        #pragma unroll
        for(int d=0; d < D; d++) {
            Surreal<RealType> dx = ci[d] - cj[d];
            d2ij += dx*dx;
        }

        if(atom_j_idx < atom_i_idx && d2ij.real < cutoff*cutoff && atom_j_idx < N && atom_i_idx < N) {

            // this steaming pile is unrolled for speed, and for the fact that we do not overload
            // pow for complex numbers as they get very very tricky
            Surreal<RealType> dij = sqrt(d2ij);
            Surreal<RealType> inv_dij = Surreal<RealType>(1.0)/dij;
            Surreal<RealType> inv_d2ij = inv_dij*inv_dij;
            Surreal<RealType> inv_d3ij = inv_d2ij*inv_dij;
            Surreal<RealType> inv_d4ij = inv_d3ij*inv_dij;
            Surreal<RealType> inv_d6ij = inv_d4ij*inv_d2ij;
            Surreal<RealType> inv_d7ij = inv_d4ij*inv_d3ij;

            Surreal<RealType> es_grad_prefactor = qi*qj*inv_d3ij;

            // lennard jones force
            RealType eps_ij = sqrt(eps_i*eps_j);
            RealType sig_ij = (sig_i+sig_j)/2;

            RealType sig2 = sig_ij*sig_ij;
            RealType sig4 = sig2*sig2;
            RealType sig5 = sig4*sig_ij;
            RealType sig6 = sig4*sig2;
            RealType sig12 = sig6*sig6;

            Surreal<RealType> sig12_rij7 = sig12*inv_d7ij*inv_d7ij;
            Surreal<RealType> sig6_rij4 = sig6*inv_d4ij*inv_d4ij;
            Surreal<RealType> lj_grad_prefactor = 24*eps_ij*(sig12_rij7*2 - sig6_rij4);

            #pragma unroll
            for(int d=0; d < D; d++) {
                gi[d] -= (es_grad_prefactor + lj_grad_prefactor) * (ci[d]-cj[d]);
                gj[d] += (es_grad_prefactor + lj_grad_prefactor) * (ci[d]-cj[d]);
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
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        g_qj = __shfl_sync(0xffffffff, g_qj, srcLane);
        g_sigj = __shfl_sync(0xffffffff, g_sigj, srcLane);
        g_epsj = __shfl_sync(0xffffffff, g_epsj, srcLane);
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        #pragma unroll
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
    const RealType *coords,
    const RealType *params,
    const int *charge_param_idxs, // [N]
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    const RealType *block_bounds_ctr,
    const RealType *block_bounds_ext,
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
        RealType dx = max(0.0, abs(block_row_ctr-block_col_ctr) - (block_row_ext+block_col_ext));
        block_d2ij += dx*dx;
    }

    if(block_d2ij > cutoff*cutoff) {
        return;
    }

    int atom_i_idx =  blockIdx.x*32 + threadIdx.x;
    RealType ci[D] = {0};
    RealType gi[D] = {0};
    #pragma unroll
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
    RealType cj[D] = {0};
    RealType gj[D] = {0};
    #pragma unroll
    for(int d=0; d < D; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*D+d] : 0;
    }
    int charge_param_idx_j = atom_j_idx < N ? charge_param_idxs[atom_j_idx] : 0;
    int lj_param_idx_sig_j = atom_j_idx < N ? lj_param_idxs[atom_j_idx*2+0] : 0;
    int lj_param_idx_eps_j = atom_j_idx < N ? lj_param_idxs[atom_j_idx*2+1] : 0;

    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    RealType sig_j = atom_j_idx < N ? params[lj_param_idx_sig_j] : 1;
    RealType eps_j = atom_j_idx < N ? params[lj_param_idx_eps_j] : 0;

    // In inference mode, we don't care about gradients with respect to parameters.
    for(int round = 0; round < 32; round++) {

        RealType d2ij = 0;
        #pragma unroll
        for(int d=0; d < D; d++) {
            RealType dx = ci[d] - cj[d];
            d2ij += dx*dx;
        }

        if(atom_j_idx < atom_i_idx && d2ij < cutoff*cutoff && atom_j_idx < N && atom_i_idx < N) {



            RealType dij = sqrt(d2ij);
            // electrostatics force
            RealType inv_dij = 1/dij;
            RealType inv_d3ij = inv_dij*inv_dij*inv_dij;
            RealType es_grad_prefactor = qi*qj*inv_d3ij;

            // lennard jones force
            RealType eps_ij = sqrt(eps_i * eps_j);
            RealType sig_ij = (sig_i + sig_j)/2;

            RealType sig = sig_ij;
            RealType sig2 = sig*sig;
            RealType sig4 = sig2*sig2;
            RealType sig6 = sig4*sig2;
            RealType sig12 = sig6*sig6;

            RealType d4ij = d2ij*d2ij;
            RealType d8ij = d4ij*d4ij;
            RealType d14ij = d8ij*d4ij*d2ij;

            RealType sig6rij4 = sig6/d8ij;
            RealType sig12rij7 = sig12/d14ij;
            RealType lj_grad_prefactor = 24*eps_ij*(sig12rij7*2 - sig6rij4);

            #pragma unroll
            for(int d=0; d < D; d++) {
                // this loses precision, but we may have to deal with it for now (esp in single precision)
                RealType val = abs((es_grad_prefactor + lj_grad_prefactor) * (ci[d]-cj[d]));
                gi[d] -= (es_grad_prefactor + lj_grad_prefactor) * (ci[d]-cj[d]);
                gj[d] += (es_grad_prefactor + lj_grad_prefactor) * (ci[d]-cj[d]);
            }
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        #pragma unroll
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
void __global__ k_nonbonded_inference_exclusion_jvp(
    const int E, // number of exclusions
    const RealType *coords,
    const RealType *coords_tangent,
    const RealType *params,
    const int *exclusion_idxs, // [E, 2]pair-list of atoms to be excluded
    const int *charge_scale_idxs, // [E]
    const int *lj_scale_idxs, // [E] 
    const int *charge_param_idxs, // [N]
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    RealType *grad_coords_tangents, // *always* int64 for accumulation purposes, but we discard the primals
    RealType *grad_params_tangents) {

    const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(e_idx >= E) {
        return;
    }

    int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
    Surreal<RealType> ci[D] = {0};
    Surreal<RealType> gi[D] = {0};
    #pragma unroll
    for(int d=0; d < D; d++) {
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
    Surreal<RealType> cj[D] = {0};
    Surreal<RealType> gj[D] = {0};
    #pragma unroll
    for(int d=0; d < D; d++) {
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

    Surreal<RealType> d2ij = 0;
    #pragma unroll
    for(int d=0; d < D; d++) {
        Surreal<RealType> dx = ci[d] - cj[d];
        d2ij += dx*dx;
    }

    if(d2ij.real < cutoff*cutoff) {
        Surreal<RealType> dij = sqrt(d2ij);
        // electrostatics force
        Surreal<RealType> inv_dij = 1/dij;
        Surreal<RealType> inv_d3ij = inv_dij*inv_dij*inv_dij;
        Surreal<RealType> es_grad_prefactor = qi*qj*inv_d3ij;

        // lennard jones force
        RealType eps_ij = sqrt(eps_i * eps_j);
        RealType sig_ij = (sig_i + sig_j)/2;

        Surreal<RealType> d8ij = d2ij*d2ij*d2ij*d2ij;
        Surreal<RealType> d14ij = d8ij*d2ij*d2ij*d2ij;

        Surreal<RealType> sig6rij4 = pow(sig_ij, 6)/d8ij;
        Surreal<RealType> sig12rij7 = pow(sig_ij, 12)/d14ij;
        Surreal<RealType> lj_grad_prefactor = 24*eps_ij*(sig12rij7*2 - sig6rij4);

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
        RealType sig6 = pow(sig_ij, 6);
        RealType sig5 = pow(sig_ij, 5);
        Surreal<RealType> inv_d6ij = inv_d3ij*inv_d3ij;
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
void __global__ k_nonbonded_inference_exclusion(
    const int E, // number of exclusions
    const RealType *coords,
    const RealType *params,
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
    RealType ci[D] = {0};
    RealType gi[D] = {0};
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
    RealType cj[D] = {0};
    RealType gj[D] = {0};
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
        RealType dij = sqrt(d2ij);
        // electrostatics force
        RealType inv_dij = 1/dij;
        RealType inv_d3ij = inv_dij*inv_dij*inv_dij;
        RealType es_grad_prefactor = qi*qj*inv_d3ij;

        // lennard jones force
        RealType eps_ij = sqrt(eps_i * eps_j);
        RealType sig_ij = (sig_i + sig_j)/2;

        RealType sig6rij4 = pow(sig_ij, 6)/pow(d2ij, 4);
        RealType sig12rij7 = pow(sig_ij, 12)/pow(d2ij, 7);
        RealType lj_grad_prefactor = 24*eps_ij*(sig12rij7*2 - sig6rij4);

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
