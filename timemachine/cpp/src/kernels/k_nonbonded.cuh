#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
#define WARPSIZE 32

#define PI 3.141592653589793115997963468544185161

template <typename RealType>
void __global__ k_nonbonded(
    const int N,
    const double * __restrict__ coords,
    const double * __restrict__ params, // [N]
    const double * __restrict__ box,
    const double lambda,
    const int * __restrict__ lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const double beta,
    const double cutoff,
    // const unsigned int * __restrict__ ixn_count,
    const int * __restrict__ ixn_tiles,
    const unsigned int * __restrict__ ixn_atoms,
    unsigned long long * __restrict__ du_dx,
    double * __restrict__ du_dp,
    double * __restrict__ du_dl,
    double * __restrict__ u) {


    RealType bx[3] = {
        static_cast<RealType>(box[0*3+0]),
        static_cast<RealType>(box[1*3+1]),
        static_cast<RealType>(box[2*3+2])
    };


    int tile_idx = blockIdx.x;

    int row_block_idx = ixn_tiles[tile_idx];

    int atom_i_idx = row_block_idx*32 + threadIdx.x;
    int lambda_offset_i = atom_i_idx < N ? lambda_offset_idxs[atom_i_idx] : 0;

    RealType ci[3];
    RealType gi[3] = {0};
    RealType du_dl_i = 0;
    for(int d=0; d < 3; d++) {
        ci[d] = atom_i_idx < N ? coords[atom_i_idx*3+d] : 0;
    }
    
    int charge_param_idx_i = atom_i_idx*3 + 0;

    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType g_qi = 0;

    int atom_j_idx = ixn_atoms[tile_idx*32 + threadIdx.x];

    // printf("threadIdx.x %d tile_idx %d i %d j %d\n", threadIdx.x, tile_idx, atom_i_idx, atom_j_idx);

    int lambda_offset_j = atom_j_idx < N ? lambda_offset_idxs[atom_j_idx] : 0;

    RealType cj[3];
    RealType gj[3] = {0};
    RealType du_dl_j = 0;
    for(int d=0; d < 3; d++) {
        cj[d] = atom_j_idx < N ? coords[atom_j_idx*3+d] : 0;
    }

    int charge_param_idx_j = atom_j_idx*3 + 0;

    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    RealType g_qj = 0;
    RealType inv_cutoff = 1/cutoff;

    RealType energy = 0;

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
        // important to flip atom_j_idx with atom_i_idx
        if(inv_dij > inv_cutoff  && atom_j_idx > atom_i_idx && atom_j_idx < N && atom_i_idx < N) {

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

        // const int srcLane = (threadIdx.x/WARPSIZE)*WARPSIZE + (threadIdx.x+1)%WARPSIZE;
        const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane); // we can pre-compute this
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



// template<typename RealType>
// void __global__ k_nonbonded_exclusion_inference(
//     const int E, // number of exclusions
//     const double *coords,
//     const double lambda,
//     const int *lambda_plane_idxs, // 0 or 1, which non-interacting plane we're on
//     const int *lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
//     const int *exclusion_idxs, // [E, 2]pair-list of atoms to be excluded
//     const double *charge_scales, // [E]
//     const double *lj_scales, // [E] 
//     const double *charge_params, // [N]
//     const double *lj_params, // [N,2]
//     const double cutoff,
//     unsigned long long *grad_coords,
//     double *out_du_dl,
//     double *out_energy) {

//     const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
//     if(e_idx >= E) {
//         return;
//     }

//     int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
//     RealType du_dl_i = 0;

//     int lambda_plane_i = lambda_plane_idxs[atom_i_idx];
//     int lambda_offset_i = lambda_offset_idxs[atom_i_idx];

//     RealType ci[3];
//     double gi[3] = {0};
//     #pragma unroll
//     for(int d=0; d < 3; d++) {
//         ci[d] = coords[atom_i_idx*3+d];
//     }

//     int charge_param_idx_i = atom_i_idx;
//     int lj_param_idx_sig_i = atom_i_idx*2+0;
//     int lj_param_idx_eps_i = atom_i_idx*2+1;

//     RealType qi = charge_params[charge_param_idx_i];
//     RealType sig_i = lj_params[lj_param_idx_sig_i];
//     RealType eps_i = lj_params[lj_param_idx_eps_i];

//     int atom_j_idx = exclusion_idxs[e_idx*2 + 1];

//     RealType du_dl_j = 0;

//     int lambda_plane_j = lambda_plane_idxs[atom_j_idx];
//     int lambda_offset_j = lambda_offset_idxs[atom_j_idx];

//     RealType cj[3];
//     double gj[3] = {0};
//     #pragma unroll
//     for(int d=0; d < 3; d++) {
//         cj[d] = coords[atom_j_idx*3+d];
//     }

//     int charge_param_idx_j = atom_j_idx;
//     int lj_param_idx_sig_j = atom_j_idx*2+0;
//     int lj_param_idx_eps_j = atom_j_idx*2+1;

//     RealType qj = charge_params[charge_param_idx_j];
//     RealType sig_j = lj_params[lj_param_idx_sig_j];
//     RealType eps_j = lj_params[lj_param_idx_eps_j];

//     RealType charge_scale = charge_scales[e_idx];
//     RealType lj_scale = lj_scales[e_idx];

//     RealType dxs[4];
//     for(int d=0; d < 3; d++) {
//         dxs[d] = ci[d] - cj[d];
//     }

//     RealType delta_lambda = cutoff*(lambda_plane_i - lambda_plane_j) + lambda*(lambda_offset_i - lambda_offset_j);
//     dxs[3] = delta_lambda;

//     RealType inv_dij = fast_vec_rnorm<RealType, 4>(dxs);
//     RealType inv_cutoff = 1/cutoff;

//     if(inv_dij > inv_cutoff) {

//         RealType inv_d2ij = inv_dij*inv_dij;
//         RealType inv_d3ij = inv_dij*inv_d2ij;
//         RealType es_grad_prefactor = qi*qj*inv_d3ij;

//         // lennard jones force
//         RealType eps_ij = sqrt(eps_i * eps_j);
//         RealType sig_ij = (sig_i + sig_j)/2;

//         RealType sig2_inv_d2ij = sig_ij*sig_ij*inv_d2ij; // avoid using inv_dij as much as we can due to loss of precision
//         RealType sig4_inv_d4ij = sig2_inv_d2ij*sig2_inv_d2ij;
//         RealType sig6_inv_d6ij = sig4_inv_d4ij*sig2_inv_d2ij;
//         RealType sig6_inv_d8ij = sig6_inv_d6ij*inv_d2ij;
//         RealType sig8_inv_d8ij = sig4_inv_d4ij*sig4_inv_d4ij;
//         RealType sig12_inv_d12ij = sig8_inv_d8ij*sig4_inv_d4ij;
//         RealType sig12_inv_d14ij = sig12_inv_d12ij*inv_d2ij;

//         // RealType lj_grad_prefactor = 24*eps_ij*(sig12rij7*2 - sig6rij4);
//         RealType lj_grad_prefactor = 24*eps_ij*sig12_inv_d14ij*2 - 24*eps_ij*sig6_inv_d8ij;

//         #pragma unroll
//         for(int d=0; d < 3; d++) {
//             gi[d] += (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor)*dxs[d];
//             gj[d] -= (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor)*dxs[d];
//         }

//         int dw_i = lambda_offset_i;
//         int dw_j = lambda_offset_j;

//         du_dl_i += (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor) * dxs[3] * dw_i;
//         du_dl_j -= (charge_scale * es_grad_prefactor + lj_scale * lj_grad_prefactor) * dxs[3] * dw_j;

//         for(int d=0; d < 3; d++) {

//             atomicAdd(grad_coords + atom_i_idx*3 + d, static_cast<unsigned long long>((long long) (gi[d]*FIXED_EXPONENT)));
//             atomicAdd(grad_coords + atom_j_idx*3 + d, static_cast<unsigned long long>((long long) (gj[d]*FIXED_EXPONENT)));

//         }  

//         atomicAdd(out_du_dl, du_dl_i + du_dl_j);
//         RealType energy = charge_scale*qi*qj*inv_dij + lj_scale*4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;
//         atomicAdd(out_energy, -energy);
//     }

// }