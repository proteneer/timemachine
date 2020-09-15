#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
#define WARPSIZE 32

#define PI 3.141592653589793115997963468544185161

template <typename RealType>
void __global__ k_electrostatics(
    const int N,
    const int T,
    const double * __restrict__ coords,
    const double * __restrict__ charge_params, // [N]
    const double * __restrict__ box,
    const double lambda,
    const int * __restrict__ lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const double beta,
    const double cutoff,
    // const double *block_bounds_ctr,
    // const double *block_bounds_ext,
    const int * __restrict__ tiles_x,
    const int * __restrict__ tiles_y,
    const int *perm,
    unsigned long long * __restrict__ du_dx,
    double * __restrict__ du_dp,
    double * __restrict__ du_dl,
    double * __restrict__ u,
    int * __restrict__ total_ixn_count,
    int * __restrict__ total_empty_tiles) {

    // if(blockIdx.y > blockIdx.x) {
        // return;
    // }

    // RealType block_d2ij = 0; 

    RealType bx[3] = {
        static_cast<RealType>(box[0*3+0]),
        static_cast<RealType>(box[1*3+1]),
        static_cast<RealType>(box[2*3+2])
    };


    // for(int d=0; d < 3; d++) {
    //     RealType block_row_ctr = block_bounds_ctr[blockIdx.x*3+d];
    //     RealType block_col_ctr = block_bounds_ctr[blockIdx.y*3+d];
    //     RealType block_row_ext = block_bounds_ext[blockIdx.x*3+d];
    //     RealType block_col_ext = block_bounds_ext[blockIdx.y*3+d];

    //     RealType dx = block_row_ctr - block_col_ctr;
    //     dx -= bx[d]*floor(dx/bx[d]+static_cast<RealType>(0.5));
    //     dx = max(static_cast<RealType>(0.0), fabs(dx) - (block_row_ext + block_col_ext));
    //     block_d2ij += dx*dx;
    // }

    // if(block_d2ij > cutoff*cutoff) {
    //     return;
    // }


    int gid = blockIdx.x*blockDim.x + threadIdx.x; // threadblock idx
    int tid = gid / 32; // which tile we're processing

    if(tid >= T) {
        return;
    }

    int bid = tiles_x[tid];
    int bjd = tiles_y[tid];

    int oid = bid*32+threadIdx.x%32; // original row index

    // int tid = bid*32 + threadIdx.x;
    int atom_i_idx = oid < N ? perm[oid] : oid;
    int lambda_offset_i = atom_i_idx < N ? lambda_offset_idxs[atom_i_idx] : 0;

    RealType ci[3];
    RealType gi[3] = {0};
    RealType du_dl_i = 0;
    for(int d=0; d < 3; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
    }
    
    int charge_param_idx_i = atom_i_idx < N ? atom_i_idx : 0;

    RealType qi = atom_i_idx < N ? charge_params[charge_param_idx_i] : 0;
    RealType g_qi = 0;

    int ojd = bjd*32 + threadIdx.x%32;
    int atom_j_idx = ojd < N ? perm[ojd] : ojd;
    int lambda_offset_j = atom_j_idx < N ? lambda_offset_idxs[atom_j_idx] : 0;

    // printf("thread %d processing bijd (%d %d) oid ojd (%d %d) \n", gid, tid, bid, bjd, oid, ojd);

    // printf("thread %d processing oijd %d %d atom_ij_idx %d %d\n", threadIdx.x, oid, ojd, atom_i_idx, atom_j_idx);

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

    RealType energy = 0;

    // In inference mode, we don't care about gradients with respect to parameters.

    RealType real_lambda = static_cast<RealType>(lambda);
    RealType real_beta = static_cast<RealType>(beta);

    int ixn_count = 0;
    for(int round = 0; round < 32; round++) {

        RealType dxs[4];
        #pragma unroll
        for(int d=0; d < 3; d++) {
            RealType delta = ci[d] - cj[d];
            delta -= bx[d]*floor(delta/bx[d]+static_cast<RealType>(0.5));
            dxs[d] = delta;
        }

        RealType delta_lambda = (lambda_offset_i - lambda_offset_j)*real_lambda;
        dxs[3] = delta_lambda;

        // apparently rnorm4d itself is not overloaded correctly like the rest of the math functions
        RealType inv_dij = real_rnorm4d(dxs[0], dxs[1], dxs[2], dxs[3]);

        // some warps will skip this entirely
        if(inv_dij > inv_cutoff  && ojd < oid && atom_j_idx < N && atom_i_idx < N) {

            ixn_count += 1;

            RealType dij = 1/inv_dij;

            RealType inv_d2ij = inv_dij*inv_dij;
            RealType ebd = erfc(real_beta*dij);
            RealType qij = qi*qj;

            RealType prefactor = qij*(-2*real_beta*exp(-real_beta*real_beta*dij*dij)/(sqrt(static_cast<RealType>(PI))*dij) - ebd*inv_d2ij)*inv_dij;

            #pragma unroll
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

        const int srcLane = (threadIdx.x/WARPSIZE)*WARPSIZE + (threadIdx.x+1)%WARPSIZE;
        // const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
        ojd = __shfl_sync(0xffffffff, ojd, srcLane);
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        #pragma unroll
        for(size_t d=0; d < 3; d++) {
            cj[d] = __shfl_sync(0xffffffff, cj[d], srcLane); // needs to support real
            gj[d] = __shfl_sync(0xffffffff, gj[d], srcLane);
        }

        g_qj = __shfl_sync(0xffffffff, g_qj, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane);
        du_dl_j = __shfl_sync(0xffffffff, du_dl_j, srcLane);
    }

    // simply have every tile add their contrib
    atomicAdd(total_ixn_count, ixn_count);


    bool tile_is_empty = __all_sync(0xffffffff, ixn_count == 0);
    if(threadIdx.x == 0) {
        atomicAdd(total_empty_tiles, tile_is_empty);        
    }

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
        delta -= floor(delta/bx[d]+static_cast<RealType>(0.5))*bx[d];
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