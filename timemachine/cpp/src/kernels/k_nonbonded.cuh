#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
// we need to make this fully deterministic if we want to be able to realiably rematerialize (this also only really matters for forward mode)
// reverse mode we don't care at all
#define WARPSIZE 32

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
    // RealType lambda_i = lambda;
    int lambda_plane_i = 0;
    int lambda_offset_i = 0;

    // RealType dlambda_i = 0;
    if(atom_i_idx < N) {
        // lambda_i = cutoff*lambda_plane_idxs[atom_i_idx] + lambda_offset_idxs[atom_i_idx]*lambda_i;

        lambda_plane_i = lambda_plane_idxs[atom_i_idx];
        lambda_offset_i = lambda_offset_idxs[atom_i_idx];

        // dlambda_i = lambda_offset_idxs[atom_i_idx];
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
    // RealType lambda_i = lambda;
    int lambda_plane_j = 0;
    int lambda_offset_j = 0;

    // RealType dlambda_i = 0;
    if(atom_j_idx < N) {
        // lambda_i = cutoff*lambda_plane_idxs[atom_i_idx] + lambda_offset_idxs[atom_i_idx]*lambda_i;

        lambda_plane_j = lambda_plane_idxs[atom_j_idx];
        lambda_offset_j = lambda_offset_idxs[atom_j_idx];

        // dlambda_i = lambda_offset_idxs[atom_i_idx];
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

    // revert this to RealType
    double energy = 0; // spit this into three parts? (es, lj close, lj far?)

    // In inference mode, we don't care about gradients with respect to parameters.
    for(int round = 0; round < 32; round++) {

        RealType dxs[4];
        for(int d=0; d < 3; d++) {
            dxs[d] = ci[d] - cj[d];
        }

        // we can optimize this later if need be
        RealType delta_lambda = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*lambda;

        dxs[3] = delta_lambda;

        RealType inv_dij = fast_vec_rnorm<RealType, 4>(dxs);

        if(atom_j_idx < atom_i_idx && inv_dij > inv_cutoff && atom_j_idx < N && atom_i_idx < N) {


            // if(atom_j_idx < N && atom_i_idx < N) {
                // printf("atom_i_idx %d atom_j_idx %d dij %f pi %d pj %d oi %d oj %d\n", atom_i_idx, atom_j_idx, 1/inv_dij, lambda_plane_i, lambda_plane_j, lambda_offset_i, lambda_offset_j);
            // }

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

                RealType force_i = (es_grad_prefactor + lj_grad_prefactor) *  dxs[d];
                RealType force_j = (es_grad_prefactor + lj_grad_prefactor) *  dxs[d];

                gi[d] -= force_i;
                gj[d] += force_j;

            }

            // this technically should be if lambda_idxs[i] == 0 and lamba_idxs[j] == 0
            // however, they both imply that delta_lambda = 0, so dxs[3] == 0, simplifying the equation
            int dw_i = lambda_offset_i;
            int dw_j = lambda_offset_j;

            du_dl_i -= (es_grad_prefactor + lj_grad_prefactor) * dxs[3] * dw_i;
            du_dl_j += (es_grad_prefactor + lj_grad_prefactor) * dxs[3] * dw_j;

            energy += qi*qj*inv_dij + 4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;
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
        lambda_plane_j = __shfl_sync(0xffffffff, lambda_plane_j, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane);
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

    // if(atom_i_idx >= 1758 && abs(du_dl_i) > 100) {
    //     printf("i %d du_dl %f\n", atom_i_idx, du_dl_i); 
    // }

    // if(atom_j_idx >= 1758 && abs(du_dl_j) > 100) {
    //     printf("j %d du_dl %f\n", atom_j_idx, du_dl_j); 
    // }

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
    // lambda_i = cutoff*lambda_plane_idxs[atom_i_idx] + lambda_offset_idxs[atom_i_idx]*lambda_i;
    int lambda_plane_i = lambda_plane_idxs[atom_i_idx];
    int lambda_offset_i = lambda_offset_idxs[atom_i_idx];

    // RealType dlambda_i = lambda_offset_idxs[atom_i_idx];

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

        // if(d == 2) {
            // if(atom_i_idx >= 1758 && atom_j_idx >= 1758) {

            // }
        // }

    RealType du_dl_j = 0;
    RealType lambda_j = lambda;

    int lambda_plane_j = lambda_plane_idxs[atom_j_idx];
    int lambda_offset_j = lambda_offset_idxs[atom_j_idx];
    // lambda_j = cutoff*lambda_plane_idxs[atom_j_idx] + lambda_offset_idxs[atom_j_idx]*lambda_j;
    // RealType dlambda_j = lambda_offset_idxs[atom_j_idx];

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

    RealType delta_lambda = cutoff*(lambda_plane_i - lambda_plane_j) + lambda*(lambda_offset_i - lambda_offset_j);
    dxs[3] = delta_lambda;

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

        int dw_i = lambda_offset_i;
        int dw_j = lambda_offset_j;

        du_dl_i += (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor) * dxs[3] * dw_i;
        du_dl_j -= (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor) * dxs[3] * dw_j;

        for(int d=0; d < 3; d++) {

            // if(d == 2) {
            //     if(atom_i_idx - 1758 == 26 || atom_j_idx - 1758 == 26) {
            //         printf("exclusion processing ixn %d %d force gi %f gj %f\n", atom_i_idx - 1758, atom_j_idx - 1758, gi[d], gj[d]);
            //     }                   
            // }
             

            atomicAdd(grad_coords + atom_i_idx*3 + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));
            atomicAdd(grad_coords + atom_j_idx*3 + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));

        }  


        // if(atom_i_idx > 1758 && atom_j_idx > 1758) {
        //     RealType dd_i = (lj_scale*lj_grad_prefactor) * dxs[3] * dw_i;
        //     RealType dd_j = (lj_scale*lj_grad_prefactor) * dxs[3] * dw_j;
        //     if(atom_i_idx == 1819 && abs(dd_i) > 1000) {
        //         printf("exc ixn between %d %d value %f with dxs[3] %f lambda_i %f lambda_j %f \n", atom_i_idx-1758, atom_j_idx-1758, dd_i, dxs[3], lambda_i, lambda_j);
        //     } else if(atom_j_idx == 1819 && abs(dd_j) > 1000) {
        //         printf("exc ixn between %d %d value %f with dxs[3] %f lambda_i %f lambda_j %f \n", atom_i_idx-1758, atom_j_idx-1758, dd_j, dxs[3], lambda_i, lambda_j);
        //     }
        // }



        // if(atom_i_idx >= 1758 && abs(du_dl_i) > 100) {
        //     printf("exc i %d du_dl %f\n", atom_i_idx, du_dl_i); 
        // }

        // if(atom_j_idx >= 1758 && abs(du_dl_j) > 100) {
        //     printf("exc j %d du_dl %f\n", atom_j_idx, du_dl_j); 
        // }

        atomicAdd(out_du_dl, du_dl_i + du_dl_j);
        RealType energy = charge_scale*qi*qj*inv_dij + lj_scale*4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;
        atomicAdd(out_energy, -energy);
    }

}