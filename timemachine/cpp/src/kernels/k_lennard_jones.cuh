#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
#define WARPSIZE 32

template <typename RealType>
void __global__ k_lennard_jones_inference(
    const int N,
    const double *coords,
    const double lambda,
    const int *lambda_plane_idxs, // -1 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // -1 or 1, how much we offset from the plane by cutoff
    // const int *lambda_group_idxs, // 0 or 1, how we do the group mixing
    const double *lj_params, // [N,2]
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    unsigned long long *du_dx,
    double *du_dp,
    double *du_dl,
    double *u) {
    // double *out_du_dl,
    // double *out_energy) {

    if(blockIdx.y > blockIdx.x) {
        return;
    }

    // TBD restore me.
    // RealType block_d2ij = 0; 
    // for(int d=0; d < 3; d++) {
    //     RealType block_row_ctr = block_bounds_ctr[blockIdx.x*3+d];
    //     RealType block_col_ctr = block_bounds_ctr[blockIdx.y*3+d];
    //     RealType block_row_ext = block_bounds_ext[blockIdx.x*3+d];
    //     RealType block_col_ext = block_bounds_ext[blockIdx.y*3+d];
    //     RealType dx = max(0.0, fabs(block_row_ctr-block_col_ctr) - (block_row_ext+block_col_ext));
    //     block_d2ij += dx*dx;
    // }

    // if(block_d2ij > cutoff*cutoff) {
    //     return;
    // }

    int atom_i_idx = blockIdx.x*32 + threadIdx.x;
    int lambda_plane_i = 0;
    int lambda_offset_i = 0;
    // int lambda_group_i = 0;

    if(atom_i_idx < N) {
        lambda_plane_i = lambda_plane_idxs[atom_i_idx];
        lambda_offset_i = lambda_offset_idxs[atom_i_idx];
    }

    RealType ci[3];
    RealType gi[3] = {0};
    RealType du_dl_i = 0;
    for(int d=0; d < 3; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
    }
    
    int lj_param_idx_sig_i = atom_i_idx < N ? atom_i_idx*2+0 : 0;
    int lj_param_idx_eps_i = atom_i_idx < N ? atom_i_idx*2+1 : 0;

    RealType sig_i = atom_i_idx < N ? lj_params[lj_param_idx_sig_i] : 1;
    RealType eps_i = atom_i_idx < N ? lj_params[lj_param_idx_eps_i] : 0;

    RealType g_sigi = 0.0;
    RealType g_epsi = 0.0;

    int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    int lambda_plane_j = 0;
    int lambda_offset_j = 0;

    if(atom_j_idx < N) {
        lambda_plane_j = lambda_plane_idxs[atom_j_idx];
        lambda_offset_j = lambda_offset_idxs[atom_j_idx];
    }

    RealType cj[3];
    RealType gj[3] = {0};
    RealType du_dl_j = 0;
    for(int d=0; d < 3; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
    }

    int lj_param_idx_sig_j = atom_j_idx < N ? atom_j_idx*2+0 : 0;
    int lj_param_idx_eps_j = atom_j_idx < N ? atom_j_idx*2+1 : 0;

    RealType sig_j = atom_j_idx < N ? lj_params[lj_param_idx_sig_j] : 1;
    RealType eps_j = atom_j_idx < N ? lj_params[lj_param_idx_eps_j] : 0;

    RealType g_sigj = 0.0;
    RealType g_epsj = 0.0;

    RealType inv_cutoff = 1/cutoff;

    // revert this to RealType

    // tbd: deprecate this when we don't need energies any more.
    double energy = 0; // spit this into three parts? (es, lj close, lj far?)

    // In inference mode, we don't care about gradients with respect to parameters.
    for(int round = 0; round < 32; round++) {

        RealType dxs[4];
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
        }

        // we can optimize this later if need be
        int dw_i = 0;
        int dw_j = 0;
    
        RealType delta_lambda = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*lambda;
        dxs[3] = delta_lambda;
        dw_i = lambda_offset_i;
        dw_j = lambda_offset_j;

        RealType inv_dij = fast_vec_rnorm<RealType, 4>(dxs);

        if(atom_j_idx < atom_i_idx && inv_dij > inv_cutoff && atom_j_idx < N && atom_i_idx < N) {

            RealType inv_d2ij = inv_dij*inv_dij;
            RealType inv_d3ij = inv_dij*inv_d2ij;
            RealType inv_d4ij = inv_d2ij*inv_d2ij;
            RealType inv_d6ij = inv_d4ij*inv_d2ij;
            RealType inv_d8ij = inv_d4ij*inv_d4ij;

            // lennard jones force
            RealType eps_ij = overloaded_sqrt(eps_i * eps_j);
            RealType sig_ij = (sig_i + sig_j)/2;

            RealType sig2 = sig_ij*sig_ij;
            RealType sig4 = sig2*sig2;
            RealType sig5 = sig4*sig_ij;
            RealType sig6 = sig4*sig2;

            RealType sig6_inv_d6ij = sig6*inv_d6ij;
            RealType sig6_inv_d8ij = sig6*inv_d8ij;

            RealType lj_grad_prefactor = 24*eps_ij*sig6_inv_d8ij*(sig6_inv_d6ij*2 - 1);

            for(int d=0; d < 3; d++) {

                RealType force_i = (lj_grad_prefactor) *  dxs[d];
                RealType force_j = (lj_grad_prefactor) *  dxs[d];

                gi[d] -= force_i;
                gj[d] += force_j;

            }

            du_dl_i -= (lj_grad_prefactor) * dxs[3] * dw_i;
            du_dl_j += (lj_grad_prefactor) * dxs[3] * dw_j;

            energy += 4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;

            RealType eps_grad = 2*sig6_inv_d6ij*(sig6_inv_d6ij-1)/eps_ij;
            g_epsi += eps_grad*eps_j;
            g_epsj += eps_grad*eps_i;

            RealType sig_grad = 12*eps_ij*sig5*inv_d6ij*(2*sig6_inv_d6ij-1);
            g_sigi += sig_grad;
            g_sigj += sig_grad;
        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        g_sigj = __shfl_sync(0xffffffff, g_sigj, srcLane);
        g_epsj = __shfl_sync(0xffffffff, g_epsj, srcLane);
        for(size_t d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane); // needs to support real
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }
        lambda_plane_j = __shfl_sync(0xffffffff, lambda_plane_j, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane);
        du_dl_j = __shfl_sync(0xffffffff, du_dl_j, srcLane);
    }

    for(int d=0; d < 3; d++) {

        if(atom_i_idx < N) {
            atomicAdd(du_dx + atom_i_idx*3 + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));            
        }

        if(atom_j_idx < N) {
            atomicAdd(du_dx + atom_j_idx*3 + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));            
        }
    }

    atomicAdd(du_dl, du_dl_i + du_dl_j);

    if(u) {
        atomicAdd(u, energy);        
    }

    if(atom_i_idx < N && du_dp) {
        atomicAdd(du_dp + lj_param_idx_sig_i, g_sigi);
        atomicAdd(du_dp + lj_param_idx_eps_i, g_epsi);
    }

    if(atom_j_idx < N && du_dp) {
        atomicAdd(du_dp + lj_param_idx_sig_j, g_sigj);
        atomicAdd(du_dp + lj_param_idx_eps_j, g_epsj);
    }


}

template<typename RealType>
void __global__ k_lennard_jones_exclusion_inference(
    const int E, // number of exclusions
    const double *coords,
    const double lambda,
    const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    // const int *lambda_group_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const int *exclusion_idxs, // [E, 2]pair-list of atoms to be excluded
    const double *lj_scales, // [E] 
    const double *lj_params, // [N,2]
    const double cutoff,
    unsigned long long *du_dx,
    double *du_dp,
    double *du_dl,
    double *u) {

    const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(e_idx >= E) {
        return;
    }

    int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
    RealType du_dl_i = 0;

    int lambda_plane_i = lambda_plane_idxs[atom_i_idx];
    int lambda_offset_i = lambda_offset_idxs[atom_i_idx];
    // int lambda_group_i = lambda_group_idxs[atom_i_idx];

    RealType ci[3];
    double gi[3] = {0};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        ci[d] = coords[atom_i_idx*3+d];
    }

    int lj_param_idx_sig_i = atom_i_idx*2+0;
    int lj_param_idx_eps_i = atom_i_idx*2+1;

    RealType sig_i = lj_params[lj_param_idx_sig_i];
    RealType eps_i = lj_params[lj_param_idx_eps_i];

    RealType g_sigi = 0.0;
    RealType g_epsi = 0.0;

    int atom_j_idx = exclusion_idxs[e_idx*2 + 1];

    RealType du_dl_j = 0;

    int lambda_plane_j = lambda_plane_idxs[atom_j_idx];
    int lambda_offset_j = lambda_offset_idxs[atom_j_idx];
    // int lambda_group_j = lambda_group_idxs[atom_j_idx];

    RealType cj[3];
    double gj[3] = {0};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        cj[d] = coords[atom_j_idx*3+d];
    }

    int lj_param_idx_sig_j = atom_j_idx*2+0;
    int lj_param_idx_eps_j = atom_j_idx*2+1;

    RealType sig_j = lj_params[lj_param_idx_sig_j];
    RealType eps_j = lj_params[lj_param_idx_eps_j];

    RealType g_sigj = 0.0;
    RealType g_epsj = 0.0;

    RealType lj_scale = lj_scales[e_idx];

    RealType dxs[4];
    for(int d=0; d < 3; d++) {
        dxs[d] = ci[d] - cj[d];
    }

    int dw_i = 0;
    int dw_j = 0;

    // if((lambda_group_j & lambda_group_i) > 0) {
    //     // 3D
    //     dxs[3] = 0;
    // } else {
        // 4D
    RealType delta_lambda = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*lambda;
    dxs[3] = delta_lambda;
    dw_i = lambda_offset_i;
    dw_j = lambda_offset_j;
    // }

    RealType inv_dij = fast_vec_rnorm<RealType, 4>(dxs);
    RealType inv_cutoff = 1/cutoff;

    if(inv_dij > inv_cutoff) {

        RealType inv_d2ij = inv_dij*inv_dij;
        RealType inv_d3ij = inv_dij*inv_d2ij;

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
            gi[d] += (lj_scale * lj_grad_prefactor)*dxs[d];
            gj[d] -= (lj_scale * lj_grad_prefactor)*dxs[d];
        }

        du_dl_i += (lj_scale * lj_grad_prefactor) * dxs[3] * dw_i;
        du_dl_j -= (lj_scale * lj_grad_prefactor) * dxs[3] * dw_j;

        for(int d=0; d < 3; d++) {

            atomicAdd(du_dx + atom_i_idx*3 + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));
            atomicAdd(du_dx + atom_j_idx*3 + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));

        }  

        atomicAdd(du_dl, du_dl_i + du_dl_j);

        if(u) {
            RealType energy = lj_scale*4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;
            atomicAdd(u, -energy);
        }

        if(du_dp) {

            RealType eps_grad = 4*(sig6_inv_d6ij-1.0)*sig6_inv_d6ij;
            g_epsi += eps_grad*eps_j/(2*eps_ij);
            g_epsj += eps_grad*eps_i/(2*eps_ij);
            RealType sig_grad = 24*eps_ij*(2*sig6_inv_d6ij-1.0)*(sig6_inv_d6ij/sig_ij);
            g_sigi += sig_grad/2;
            g_sigj += sig_grad/2;

            atomicAdd(du_dp + lj_param_idx_sig_i, -lj_scale*g_sigi);
            atomicAdd(du_dp + lj_param_idx_sig_j, -lj_scale*g_sigj);

            atomicAdd(du_dp + lj_param_idx_eps_i, -lj_scale*g_epsi);
            atomicAdd(du_dp + lj_param_idx_eps_j, -lj_scale*g_epsj);
        }

    }

}