#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
#define WARPSIZE 32

#define PI 3.141592653589793115997963468544185161

template <typename RealType>
void __global__ k_electrostatics(
    const int N,
    const double *coords,
    const double *charge_params, // [N]
    const double *box,
    const double lambda,
    const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const double beta,
    const double cutoff,
    const double *block_bounds_ctr,
    const double *block_bounds_ext,
    const int *perm,
    unsigned long long *du_dx,
    double *du_dp,
    double *du_dl,
    double *u) {

    if(blockIdx.y > blockIdx.x) {
        return;
    }

    RealType block_d2ij = 0; 
    // needs to be done periodically

    RealType bx[3] = {box[0*3+0], box[1*3+1], box[2*3+2]};

    for(int d=0; d < 3; d++) {
        RealType block_row_ctr = block_bounds_ctr[blockIdx.x*3+d];
        RealType block_col_ctr = block_bounds_ctr[blockIdx.y*3+d];
        RealType block_row_ext = block_bounds_ext[blockIdx.x*3+d];
        RealType block_col_ext = block_bounds_ext[blockIdx.y*3+d];


        RealType dx = max(0.0, fabs(block_row_ctr - block_col_ctr) - (block_row_ext + block_col_ext));

        // if(blockIdx.x == 21 && blockIdx.y == 13 && threadIdx.x == 0) {
            // printf("dx %f\n", dx);
        // }
        // dx -= floor(dx/bx[d]+static_cast<RealType>(0.5))*bx[d];
        block_d2ij += dx*dx;
    }

    // if(blockIdx.x == 21 && blockIdx.y == 13 && threadIdx.x == 0) {
    //     printf("Tile X Center %f %f %f Extent %f %f %f\n", block_bounds_ctr[blockIdx.x*3+0], block_bounds_ctr[blockIdx.x*3+1], block_bounds_ctr[blockIdx.x*3+2], block_bounds_ext[blockIdx.x*3+0], block_bounds_ext[blockIdx.x*3+1], block_bounds_ext[blockIdx.x*3+2]);
    //     printf("Tile Y Center %f %f %f Extent %f %f %f\n", block_bounds_ctr[blockIdx.y*3+0], block_bounds_ctr[blockIdx.y*3+1], block_bounds_ctr[blockIdx.y*3+2], block_bounds_ext[blockIdx.y*3+0], block_bounds_ext[blockIdx.y*3+1], block_bounds_ext[blockIdx.y*3+2]);
    // }

    if(block_d2ij > cutoff*cutoff) {
        // if(threadIdx.x == 0) {
            // printf("tile %d %d is skipped\n", blockIdx.x, blockIdx.y);
        // }
        return;
    }

    int tid = blockIdx.x*32 + threadIdx.x;
    int atom_i_idx = tid < N ? perm[tid] : tid;
    int lambda_offset_i = 0;

    if(atom_i_idx < N) {
        lambda_offset_i = lambda_offset_idxs[atom_i_idx];
    }

    RealType ci[3];
    RealType gi[3] = {0};
    RealType du_dl_i = 0;
    for(int d=0; d < 3; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
    }
    
    int charge_param_idx_i = atom_i_idx < N ? atom_i_idx : 0;

    RealType qi = atom_i_idx < N ? charge_params[charge_param_idx_i] : 0;
    RealType g_qi = 0;

    // int atom_j_idx = blockIdx.y*32 + threadIdx.x;
    int tjd = blockIdx.y*32 + threadIdx.x;
    int atom_j_idx = tjd < N ? perm[tjd] : tjd;

    int lambda_offset_j = 0;

    if(atom_j_idx < N) {
        lambda_offset_j = lambda_offset_idxs[atom_j_idx];
    }

    RealType cj[3];
    RealType gj[3] = {0};
    RealType du_dl_j = 0;
    for(int d=0; d < 3; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
    }

    int charge_param_idx_j = atom_j_idx < N ? atom_j_idx : 0;

    RealType qj = atom_j_idx < N ? charge_params[charge_param_idx_j] : 0;
    RealType g_qj = 0;
    RealType inv_cutoff = 1/cutoff;

    // revert this to RealType

    // tbd: deprecate this when we don't need energies any more.
    RealType energy = 0;

    // In inference mode, we don't care about gradients with respect to parameters.

    // int has_interaction = 0;
    for(int round = 0; round < 32; round++) {

        RealType dxs[4];
        for(int d=0; d < 3; d++) {
            RealType delta = ci[d] - cj[d];
            // delta -= floor(delta/bx[d]+static_cast<RealType>(0.5))*bx[d];
            dxs[d] = delta;
        }

        // we can optimize this later if need be
        RealType delta_lambda = (lambda_offset_i - lambda_offset_j)*lambda;
        dxs[3] = delta_lambda;

        RealType inv_dij = fast_vec_rnorm<RealType, 4>(dxs);

        // if(inv_dij > inv_cutoff) {
        //     if(blockIdx.x == 21 && blockIdx.y == 13) {
        //         printf("FAILED atom_i_idx %d (%f %f %f %f) atom_j_idx %d (%f %f %f %f) dist %f\n", atom_i_idx, ci[0], ci[1], ci[2], wi, atom_j_idx, cj[0], cj[1], cj[2], wj, dij);
        //     }
        //     has_interaction += 1;
        // }

        if(tjd < tid && inv_dij > inv_cutoff && atom_j_idx < N && atom_i_idx < N) {

            RealType dij = 1/inv_dij;

            RealType inv_d2ij = inv_dij*inv_dij;
            RealType ebd = real_erfc(beta*dij);
            RealType qij = qi*qj;

            RealType prefactor = qij*(-2*beta*real_exp(-beta*beta*dij*dij)/(real_sqrt(PI)*dij) - ebd*inv_d2ij)*inv_dij;

            for(int d=0; d < 3; d++) {

                RealType force = prefactor * dxs[d];
                // note switch here
                gi[d] += force;
                gj[d] -= force;

            }

            du_dl_i += prefactor * dxs[3] * lambda_offset_i;
            du_dl_j -= prefactor * dxs[3] * lambda_offset_j;

            energy += qij*inv_dij*ebd;

            g_qi += qj*inv_dij*ebd;
            g_qj += qi*inv_dij*ebd;

        }

        const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
        tjd = __shfl_sync(0xffffffff, tjd, srcLane);

        qj = __shfl_sync(0xffffffff, qj, srcLane);
        for(size_t d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane); // needs to support real
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }

        g_qj = __shfl_sync(0xffffffff, g_qj, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane);
        du_dl_j = __shfl_sync(0xffffffff, du_dl_j, srcLane);
    }

    // bool tile_has_interaction = __any_sync(0xffffffff, has_interaction);

    // if(threadIdx.x == 0 && tile_has_interaction && block_d2ij > cutoff*cutoff) {
    //     printf("tile %d %d should not have been skipped\n", blockIdx.x, blockIdx.y);
    // }

    if(du_dx) {
        for(int d=0; d < 3; d++) {
            if(atom_i_idx < N) {
                atomicAdd(du_dx + atom_i_idx*3 + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));            
            }
            if(atom_j_idx < N) {
                atomicAdd(du_dx + atom_j_idx*3 + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));            
            }
        }   
    }

    if(du_dp) {

        if(atom_i_idx < N) {
            atomicAdd(du_dp + charge_param_idx_i, g_qi);
        }

        if(atom_j_idx < N) {
            atomicAdd(du_dp + charge_param_idx_j, g_qj);
        }

    }

    if(du_dl) {
        atomicAdd(du_dl, du_dl_i + du_dl_j);        
    }

    if(u) {
        atomicAdd(u, energy);        
    }


}

template<typename RealType>
void __global__ k_electrostatics_exclusion_inference(
    const int E, // number of exclusions
    const double *coords,
    const double *charge_params,
    const double *box,
    const double lambda,
    const int *lambda_offset_idxs, // 0 or 1, if we alolw this atom to be decoupled
    const int *exclusion_idxs, // [E, 2] pair-list of atoms to be excluded
    const double *charge_scales, // [E]
    const double beta,
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

    int lambda_offset_i = lambda_offset_idxs[atom_i_idx];

    RealType ci[3];
    double gi[3] = {0};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        ci[d] = coords[atom_i_idx*3+d];
    }

    int charge_param_idx_i = atom_i_idx;

    RealType qi = charge_params[charge_param_idx_i];
    RealType g_qi = 0;

    int atom_j_idx = exclusion_idxs[e_idx*2 + 1];

    RealType du_dl_j = 0;

    int lambda_offset_j = lambda_offset_idxs[atom_j_idx];

    RealType cj[3];
    double gj[3] = {0};
    #pragma unroll
    for(int d=0; d < 3; d++) {
        cj[d] = coords[atom_j_idx*3+d];
    }

    int charge_param_idx_j = atom_j_idx;
    RealType qj = charge_params[charge_param_idx_j];
    RealType g_qj = 0;
    RealType charge_scale = charge_scales[e_idx];

    RealType bx[3] = {box[0*3+0], box[1*3+1], box[2*3+2]};

    RealType dxs[4];
    for(int d=0; d < 3; d++) {
        RealType delta = ci[d] - cj[d];
        // delta -= floor(delta/bx[d]+static_cast<RealType>(0.5))*bx[d];
        dxs[d] = delta;
    }

    RealType delta_lambda = lambda*(lambda_offset_i - lambda_offset_j);
    dxs[3] = delta_lambda;

    RealType inv_dij = fast_vec_rnorm<RealType, 4>(dxs);
    RealType inv_cutoff = 1/cutoff;
    RealType dij = 1/inv_dij;

    if(inv_dij > inv_cutoff) {

        RealType inv_d2ij = inv_dij*inv_dij;
        RealType es_grad_prefactor = qi*qj*(-2*beta*exp(-beta*beta*dij*dij)/(sqrt(PI)*dij) - erfc(beta*dij)*inv_d2ij);

        #pragma unroll
        for(int d=0; d < 3; d++) {
            gi[d] -= charge_scale * es_grad_prefactor * (dxs[d]/dij);
            gj[d] += charge_scale * es_grad_prefactor * (dxs[d]/dij);
        }

        du_dl_i -= charge_scale * es_grad_prefactor * (dxs[3]/dij) * lambda_offset_i;
        du_dl_j += charge_scale * es_grad_prefactor * (dxs[3]/dij) * lambda_offset_j;

        if(du_dx) {
            for(int d=0; d < 3; d++) {
                atomicAdd(du_dx + atom_i_idx*3 + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));
                atomicAdd(du_dx + atom_j_idx*3 + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));
            }
        }

        if(du_dp) {

            g_qi += charge_scale*qj*inv_dij*erfc(beta*dij);
            g_qj += charge_scale*qi*inv_dij*erfc(beta*dij);

            atomicAdd(du_dp + charge_param_idx_i, -g_qi);
            atomicAdd(du_dp + charge_param_idx_j, -g_qj);

        }

        if(du_dl) {
            atomicAdd(du_dl, du_dl_i + du_dl_j);            
        }

        if(u) {
            RealType energy = charge_scale*qi*qj*inv_dij*erfc(beta*dij);
            atomicAdd(u, -energy);            
        }

    }

}