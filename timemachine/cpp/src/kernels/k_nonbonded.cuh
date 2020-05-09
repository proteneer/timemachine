#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"

#include "k_periodic_utils.cuh"
// we need to make this fully deterministic if we want to be able to realiably rematerialize (this also only really matters for forward mode)
// reverse mode we don't care at all
#define WARPSIZE 32

#define NB_EXP 8

// assume D = 3
// template <typename RealType, int D>
template <typename RealType>
void __global__ k_nonbonded_inference(
    const int N,
    const double *coords,
    const double *params,
    const double lambda,
    const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const int *charge_param_idxs, // [N]
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    unsigned long long *grad_coords,
    double *out_du_dl,
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

    int atom_i_idx = blockIdx.x*32 + threadIdx.x;
    RealType lambda_i = lambda;
    RealType dlambda_i = 0;
    if(atom_i_idx < N) {
        lambda_i = cutoff*(lambda_plane_idxs[atom_i_idx] + lambda_offset_idxs[atom_i_idx]*lambda_i);
        dlambda_i = cutoff*lambda_offset_idxs[atom_i_idx];
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
    RealType lambda_j = lambda;
    RealType dlambda_j = 0;
    if(atom_j_idx < N) {
        lambda_j = cutoff*(lambda_plane_idxs[atom_j_idx] + lambda_offset_idxs[atom_j_idx]*lambda_j);
        dlambda_j = cutoff*lambda_offset_idxs[atom_j_idx];
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
    RealType energy = 0; // spit this into three parts? (es, lj close, lj far?)

    // In inference mode, we don't care about gradients with respect to parameters.
    for(int round = 0; round < 32; round++) {

        RealType dxs[4];
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
        }

        RealType delta_lambda = lambda_i - lambda_j;
        dxs[3] = apply_delta(delta_lambda, 2*cutoff); 

        RealType inv_dij = fast_vec_rnorm<RealType, 4>(dxs);
        RealType dij = 1/inv_dij;

        if(atom_j_idx < atom_i_idx && inv_dij > inv_cutoff && atom_j_idx < N && atom_i_idx < N) {

            RealType inv_d2ij = inv_dij*inv_dij;
            RealType inv_d3ij = inv_dij*inv_d2ij;
            RealType inv_d4ij = inv_d2ij*inv_d2ij;
            RealType inv_d6ij = inv_d4ij*inv_d2ij;
            RealType inv_d7ij = inv_d6ij*inv_dij;
            RealType inv_d8ij = inv_d4ij*inv_d4ij;
            RealType es_grad_prefactor = -qi*qj*inv_d2ij;

            // lennard jones force
            RealType eps_ij = overloaded_sqrt(eps_i * eps_j);
            RealType sig_ij = (sig_i + sig_j)/2;

            RealType sig2 = sig_ij*sig_ij;
            RealType sig4 = sig2*sig2;
            RealType sig6 = sig4*sig2;

            RealType sig6_inv_d6ij = sig6*inv_d6ij;
            RealType sig6_inv_d7ij = sig6*inv_d7ij;
            // RealType sig6_inv_d8ij = sig6*inv_d8ij;

            RealType lj_grad_prefactor = -24*eps_ij*sig6_inv_d7ij*(sig6_inv_d6ij*2 - 1);

            // smooth rescale switch using cosine rule
            RealType inner = (PI*pow(dij, NB_EXP))/(2*cutoff);
            RealType sw = cos(inner);
            sw = sw*sw;

            // faster alternate form exists
            RealType dsw_dr = -(NB_EXP)*pow(dij, NB_EXP-1)*(PI/cutoff)*sin(inner)*cos(inner);
            RealType es_energy = qi*qj*inv_dij;
            RealType lj_energy = 4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;
            RealType energy_sum = es_energy + lj_energy;
            RealType grad_sum = es_grad_prefactor + lj_grad_prefactor;

            energy += sw*energy_sum;

            RealType product_rule = dsw_dr*energy_sum + sw*grad_sum;

            for(int d=0; d < 3; d++) {
                RealType force_i = product_rule * (-dxs[d]/dij);
                RealType force_j = product_rule * (-dxs[d]/dij);
                gi[d] -= force_i; // flip this fucking sign later so it's correct derivatives
                gj[d] += force_j;
            }

            RealType dw_i = dlambda_i;
            RealType dw_j = dlambda_j;

            du_dl_i -= product_rule * (-dxs[3]/dij) * dw_i;
            du_dl_j += product_rule * (-dxs[3]/dij) * dw_j;

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
        dlambda_j = __shfl_sync(0xffffffff, dlambda_j, srcLane);
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

    atomicAdd(out_du_dl, du_dl_i + du_dl_j);
    atomicAdd(out_energy, energy);

}

template<typename RealType>
void __global__ k_nonbonded_exclusion_inference(
    const int E, // number of exclusions
    const double *coords,
    const double *params,
    const double lambda,
    const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const int *exclusion_idxs, // [E, 2]pair-list of atoms to be excluded
    const int *charge_scale_idxs, // [E]
    const int *lj_scale_idxs, // [E] 
    const int *charge_param_idxs, // [N]
    const int *lj_param_idxs, // [N,2]
    const double cutoff,
    unsigned long long *grad_coords,
    double *out_du_dl,
    double *out_energy) {

    const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(e_idx >= E) {
        return;
    }

    int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
    RealType du_dl_i = 0;
    RealType lambda_i = lambda;
    lambda_i = cutoff*(lambda_plane_idxs[atom_i_idx] + lambda_offset_idxs[atom_i_idx]*lambda_i);
    RealType dlambda_i = cutoff*lambda_offset_idxs[atom_i_idx];

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
    RealType lambda_j = lambda;
    lambda_j = cutoff*(lambda_plane_idxs[atom_j_idx] + lambda_offset_idxs[atom_j_idx]*lambda_j);
    RealType dlambda_j = cutoff*lambda_offset_idxs[atom_j_idx];

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

    RealType es_scale = params[charge_scale_idxs[e_idx]];
    RealType lj_scale = params[lj_scale_idxs[e_idx]];

    RealType dxs[4];
    for(int d=0; d < 3; d++) {
        dxs[d] = ci[d] - cj[d];
    }

    RealType delta_lambda = lambda_i - lambda_j;
    dxs[3] = apply_delta(delta_lambda, 2*cutoff); 

    RealType inv_dij = fast_vec_rnorm<RealType, 4>(dxs);
    RealType inv_cutoff = 1/cutoff;
    RealType dij = 1/inv_dij;

    if(inv_dij > inv_cutoff) {

        RealType inv_d2ij = inv_dij*inv_dij;
        RealType inv_d3ij = inv_dij*inv_d2ij;
        RealType es_grad_prefactor = -qi*qj*inv_d2ij;

        // lennard jones force
        RealType eps_ij = sqrt(eps_i * eps_j);
        RealType sig_ij = (sig_i + sig_j)/2;

        RealType sig2_inv_d2ij = sig_ij*sig_ij*inv_d2ij; // avoid using inv_dij as much as we can due to loss of precision
        RealType sig4_inv_d4ij = sig2_inv_d2ij*sig2_inv_d2ij;
        RealType sig6_inv_d6ij = sig4_inv_d4ij*sig2_inv_d2ij;
        RealType sig6_inv_d8ij = sig6_inv_d6ij*inv_d2ij;
        RealType sig6_inv_d7ij = sig6_inv_d6ij*inv_dij;
        RealType sig8_inv_d8ij = sig4_inv_d4ij*sig4_inv_d4ij;
        RealType sig12_inv_d12ij = sig8_inv_d8ij*sig4_inv_d4ij;
        RealType sig12_inv_d14ij = sig12_inv_d12ij*inv_d2ij;
        RealType lj_grad_prefactor = -24*eps_ij*sig6_inv_d7ij*(sig6_inv_d6ij*2 - 1);

        RealType inner = (PI*pow(dij, NB_EXP))/(2*cutoff);
        RealType sw = cos(inner);
        sw = sw*sw;

        RealType dsw_dr = -(NB_EXP)*pow(dij, NB_EXP-1)*(PI/cutoff)*sin(inner)*cos(inner);
        RealType es_energy = qi*qj*inv_dij;
        RealType lj_energy = 4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;
        RealType energy_sum = es_scale*es_energy + lj_scale*lj_energy;
        RealType grad_sum = es_scale*es_grad_prefactor + lj_scale*lj_grad_prefactor;

        RealType energy = sw*energy_sum;
        atomicAdd(out_energy, -energy);
        RealType product_rule = dsw_dr*energy_sum + sw*grad_sum;

        for(int d=0; d < 3; d++) {
            gi[d] += product_rule * (-dxs[d]/dij);
            gj[d] -= product_rule * (-dxs[d]/dij);
        }

        RealType dw_i = dlambda_i;
        RealType dw_j = dlambda_j;

        du_dl_i += product_rule * (-dxs[3]/dij) * dw_i;
        du_dl_j -= product_rule * (-dxs[3]/dij) * dw_j;

        for(int d=0; d < 3; d++) {
            atomicAdd(grad_coords + atom_i_idx*3 + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));
            atomicAdd(grad_coords + atom_j_idx*3 + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));
        }  

        atomicAdd(out_du_dl, du_dl_i + du_dl_j);

    }

}
