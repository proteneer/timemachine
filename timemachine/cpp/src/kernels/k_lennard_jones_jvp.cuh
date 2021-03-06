#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
#include "k_periodic_utils.cuh"

template <typename RealType>
void __global__ k_lennard_jones_jvp(
    const int N,
    const double *coords, // maybe Surreal or Real
    const double *coords_tangent, // maybe Surreal or Real
    const double lambda_primal,
    const double lambda_tangent,
    const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const int *lambda_group_idxs,
    const double *lj_params, // [N,2]
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    double *grad_coords_primals,
    double *grad_coords_tangents, // *always* int64 for accumulation purposes, but we discard the primals
    double *grad_lj_params_primals,
    double *grad_lj_params_tangents) {

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
    int lambda_group_i = 0;

    if(atom_i_idx < N) {
        lambda_plane_i = lambda_plane_idxs[atom_i_idx];
        lambda_offset_i = lambda_offset_idxs[atom_i_idx];
        lambda_group_i = lambda_group_idxs[atom_i_idx];
    }

    Surreal<RealType> ci[3];
    Surreal<RealType> gi[3];

    for(int d=0; d < 3; d++) {
        gi[d].real = 0.0;
        gi[d].imag = 0.0;
        ci[d].real = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
        ci[d].imag = atom_i_idx < N ? coords_tangent[atom_i_idx*3+d] : 0;
    }

    int lj_param_idx_sig_i = atom_i_idx < N ? atom_i_idx*2+0 : 0;
    int lj_param_idx_eps_i = atom_i_idx < N ? atom_i_idx*2+1 : 0;

    RealType sig_i = atom_i_idx < N ? lj_params[lj_param_idx_sig_i] : 1;
    RealType eps_i = atom_i_idx < N ? lj_params[lj_param_idx_eps_i] : 0;

    Surreal<RealType> g_sigi(0.0, 0.0);
    Surreal<RealType> g_epsi(0.0, 0.0);

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    int lambda_plane_j = 0;
    int lambda_offset_j = 0;
    int lambda_group_j = 0;

    if(atom_j_idx < N) {
        lambda_plane_j = lambda_plane_idxs[atom_j_idx];
        lambda_offset_j = lambda_offset_idxs[atom_j_idx];
        lambda_group_j = lambda_group_idxs[atom_j_idx];
    }

    Surreal<RealType> cj[3];
    Surreal<RealType> gj[3];

    for(int d=0; d < 3; d++) {
        gj[d].real = 0.0;
        gj[d].imag = 0.0;
        cj[d].real = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
        cj[d].imag = atom_j_idx < N ? coords_tangent[atom_j_idx*3+d] : 0;
    }

    int lj_param_idx_sig_j = atom_j_idx < N ? atom_j_idx*2+0 : 0;
    int lj_param_idx_eps_j = atom_j_idx < N ? atom_j_idx*2+1 : 0;

    RealType sig_j = atom_j_idx < N ? lj_params[lj_param_idx_sig_j] : 1;
    RealType eps_j = atom_j_idx < N ? lj_params[lj_param_idx_eps_j] : 0;

    Surreal<RealType> g_sigj(0.0, 0.0);
    Surreal<RealType> g_epsj(0.0, 0.0);

    Surreal<RealType> lambda(lambda_primal, lambda_tangent);

    for(int round = 0; round < 32; round++) {

        Surreal<RealType> dxs[4]; // tbd don't need 4 of these
        Surreal<RealType> d2ij(0,0);
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
            d2ij += dxs[d]*dxs[d];
        }

        if((lambda_group_j & lambda_group_i) > 0) {
            // do nothing
            // 3D
        } else {
            // 4D
            Surreal<RealType> delta_lambda = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*lambda;
            d2ij += delta_lambda * delta_lambda;
        }

        if(atom_j_idx < atom_i_idx && d2ij.real < cutoff*cutoff && atom_j_idx < N && atom_i_idx < N) {

            Surreal<RealType> inv_d2ij = 1/d2ij;
            Surreal<RealType> inv_d4ij = inv_d2ij*inv_d2ij;
            Surreal<RealType> inv_d6ij = inv_d4ij*inv_d2ij;
            Surreal<RealType> inv_d8ij = inv_d4ij*inv_d4ij;
            Surreal<RealType> inv_d14ij = inv_d8ij*inv_d6ij;

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
                gi[d] -= lj_grad_prefactor * dxs[d];
                gj[d] += lj_grad_prefactor * dxs[d];
            }

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
        g_sigj = __shfl_sync(0xffffffff, g_sigj, srcLane);
        g_epsj = __shfl_sync(0xffffffff, g_epsj, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        for(size_t d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane);
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
        lambda_plane_j = __shfl_sync(0xffffffff, lambda_plane_j, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane);
        lambda_group_j = __shfl_sync(0xffffffff, lambda_group_j, srcLane);
    }

    // we should always accumulate in double precision

    // (ytz): we don't care about deterministic atomics that much when
    // doing reverse mode since we only ever have to do it once.
    for(int d=0; d < 3; d++) {
        if(atom_i_idx < N) {
            atomicAdd(grad_coords_primals + atom_i_idx*3 + d, gi[d].real);
            atomicAdd(grad_coords_tangents + atom_i_idx*3 + d, gi[d].imag);
        }
        if(atom_j_idx < N) {
            atomicAdd(grad_coords_primals + atom_j_idx*3 + d, gj[d].real);
            atomicAdd(grad_coords_tangents + atom_j_idx*3 + d, gj[d].imag);
        }
    }  

    if(atom_i_idx < N) {
        atomicAdd(grad_lj_params_primals + lj_param_idx_sig_i, g_sigi.real);
        atomicAdd(grad_lj_params_primals + lj_param_idx_eps_i, g_epsi.real);

        atomicAdd(grad_lj_params_tangents + lj_param_idx_sig_i, g_sigi.imag);
        atomicAdd(grad_lj_params_tangents + lj_param_idx_eps_i, g_epsi.imag);
    }

    if(atom_j_idx < N) {
        atomicAdd(grad_lj_params_primals + lj_param_idx_sig_j, g_sigj.real);
        atomicAdd(grad_lj_params_primals + lj_param_idx_eps_j, g_epsj.real);

        atomicAdd(grad_lj_params_tangents + lj_param_idx_sig_j, g_sigj.imag);
        atomicAdd(grad_lj_params_tangents + lj_param_idx_eps_j, g_epsj.imag);
    }
}


template<typename RealType>
void __global__ k_lennard_jones_exclusion_jvp(
    const int E, // number of exclusions
    const double *coords,
    const double *coords_tangent,
    const double lambda_primal,
    const double lambda_tangent,
    const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const int *lambda_group_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const int *exclusion_idxs, // [E, 2]pair-list of atoms to be excluded
    const double *lj_scales, // [E] 
    const double *lj_params, // [N,2]
    const double cutoff,
    double *grad_coords_primals,
    double *grad_coords_tangents, // *always* int64 for accumulation purposes, but we discard the primals
    double *grad_lj_params_primals,
    double *grad_lj_params_tangents) {

    const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(e_idx >= E) {
        return;
    }

    int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
    int lambda_plane_i = lambda_plane_idxs[atom_i_idx];
    int lambda_offset_i = lambda_offset_idxs[atom_i_idx];
    int lambda_group_i = lambda_group_idxs[atom_i_idx];

    Surreal<RealType> ci[3];
    Surreal<RealType> gi[3] = {Surreal<RealType>(0.0, 0.0)};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        gi[d].real = 0;
        gi[d].imag = 0;
        ci[d].real = coords[atom_i_idx*3+d];
        ci[d].imag = coords_tangent[atom_i_idx*3+d];
    }

    int lj_param_idx_sig_i = atom_i_idx*2+0;
    int lj_param_idx_eps_i = atom_i_idx*2+1;

    RealType sig_i = lj_params[lj_param_idx_sig_i];
    RealType eps_i = lj_params[lj_param_idx_eps_i];

    Surreal<RealType> g_sigi(0.0, 0.0);
    Surreal<RealType> g_epsi(0.0, 0.0);

    int atom_j_idx = exclusion_idxs[e_idx*2 + 1];
    int lambda_plane_j = lambda_plane_idxs[atom_j_idx];
    int lambda_offset_j = lambda_offset_idxs[atom_j_idx];
    int lambda_group_j = lambda_group_idxs[atom_j_idx];

    Surreal<RealType> cj[3];
    Surreal<RealType> gj[3] = {Surreal<RealType>(0.0, 0.0)};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        gj[d].real = 0;
        gj[d].imag = 0;
        cj[d].real = coords[atom_j_idx*3+d];
        cj[d].imag = coords_tangent[atom_j_idx*3+d];
    }

    int charge_param_idx_j = atom_j_idx;
    int lj_param_idx_sig_j = atom_j_idx*2+0;
    int lj_param_idx_eps_j = atom_j_idx*2+1;

    RealType sig_j = lj_params[lj_param_idx_sig_j];
    RealType eps_j = lj_params[lj_param_idx_eps_j];

    Surreal<RealType> g_sigj(0.0, 0.0);
    Surreal<RealType> g_epsj(0.0, 0.0);

    RealType lj_scale = lj_scales[e_idx];

    Surreal<RealType> d2ij(0.0, 0.0);
    #pragma unroll
    for(int d=0; d < 3; d++) {
        Surreal<RealType> dx = ci[d] - cj[d];
        d2ij += dx*dx;
    }

    Surreal<RealType> lambda(lambda_primal, lambda_tangent);

    if((lambda_group_j & lambda_group_i) > 0) {
        // 3D do nothing
    } else {
        // 4D
        Surreal<RealType> delta_lambda = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*lambda;
        d2ij += delta_lambda * delta_lambda;
    }

    if(d2ij.real < cutoff*cutoff) {

        Surreal<RealType> inv_dij = rsqrt(d2ij);
        Surreal<RealType> inv_d2ij = 1/d2ij;
        Surreal<RealType> inv_d4ij = inv_d2ij*inv_d2ij;
        Surreal<RealType> inv_d6ij = inv_d4ij*inv_d2ij;
        Surreal<RealType> inv_d8ij = inv_d4ij*inv_d4ij;

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
            gi[d] += lj_scale * lj_grad_prefactor * dx;
            gj[d] -= lj_scale * lj_grad_prefactor * dx;
        }

        for(int d=0; d < 3; d++) {
            atomicAdd(grad_coords_primals + atom_i_idx*3 + d, gi[d].real);
            atomicAdd(grad_coords_tangents + atom_i_idx*3 + d, gi[d].imag);
            atomicAdd(grad_coords_primals + atom_j_idx*3 + d, gj[d].real);
            atomicAdd(grad_coords_tangents + atom_j_idx*3 + d, gj[d].imag);
        }  

        // vDw
        Surreal<RealType> eps_grad = 4*(sig6*inv_d6ij-1.0)*sig6*inv_d6ij;
        g_epsi += eps_grad*eps_j/(2*eps_ij);
        g_epsj += eps_grad*eps_i/(2*eps_ij);
        Surreal<RealType> sig_grad = 24*eps_ij*(2*sig6*inv_d6ij-1.0)*(sig5*inv_d6ij);
        g_sigi += sig_grad/2;
        g_sigj += sig_grad/2;

        atomicAdd(grad_lj_params_primals + lj_param_idx_sig_i, -lj_scale*g_sigi.real);
        atomicAdd(grad_lj_params_primals + lj_param_idx_sig_j, -lj_scale*g_sigj.real);

        atomicAdd(grad_lj_params_tangents + lj_param_idx_sig_i, -lj_scale*g_sigi.imag);
        atomicAdd(grad_lj_params_tangents + lj_param_idx_sig_j, -lj_scale*g_sigj.imag);

        atomicAdd(grad_lj_params_primals + lj_param_idx_eps_i, -lj_scale*g_epsi.real);
        atomicAdd(grad_lj_params_primals + lj_param_idx_eps_j, -lj_scale*g_epsj.real);

        atomicAdd(grad_lj_params_tangents + lj_param_idx_eps_i, -lj_scale*g_epsi.imag);
        atomicAdd(grad_lj_params_tangents + lj_param_idx_eps_j, -lj_scale*g_epsj.imag);

    }

}
