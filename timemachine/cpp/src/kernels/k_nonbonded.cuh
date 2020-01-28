#pragma once

#include "stdio.h"

#define NDIMS 4 // do *not* change this under any circumstance
#define WARPSIZE 32

#define SHARED_OFF_DIAG_HESS(i,j,di,dj) ((i%8)*4*8*4 + di*8*4 + (j%8)*4 + dj)

// #define SHARED_ON_DIAG_HESS(i,di,dj) (i*4*4 + di*4 + dj)
// #define SHARED_ON_DIAG_HESS(i,di,dj) (((i%8)*4+di)*4 + dj)

#define SHARED_ON_DIAG_HESS(i,di,dj) (di*8*4 + (i%8)*4 + dj)
// #define SHARED_ON_DIAG_HESS(i,di,dj) ((di*8 + i%8)*4 + dj)

#define GLOBAL_HESS(i,j,di,dj,N) (((i*4 + di)*N + j)*4 + dj)


template <typename RealType>
void __global__ k_find_block_bounds(
    const int N,
    const RealType *coords,
    RealType* block_bounds_min,
    RealType* block_bounds_max) {

    const int gid = blockDim.x*blockIdx.x + threadIdx.x;
    const int atom_i_idx = gid / 4;
    const int dim_i_idx = gid % 4;

    RealType ci = coords[atom_i_idx*NDIMS + dim_i_idx];
    RealType ci_max = ci;
    RealType nb_max;

    nb_max = __shfl_xor_sync(0xffffffff, ci_max, 4);
    ci_max = nb_max > ci_max ? nb_max : ci_max;
    nb_max = __shfl_xor_sync(0xffffffff, ci_max, 8);
    ci_max = nb_max > ci_max ? nb_max : ci_max;
    nb_max = __shfl_xor_sync(0xffffffff, ci_max, 16);
    ci_max = nb_max > ci_max ? nb_max : ci_max;

    RealType ci_min = ci;
    RealType nb_min;

    nb_min = __shfl_xor_sync(0xffffffff, ci_min, 4);
    ci_min = nb_min < ci_min ? nb_min : ci_min;
    nb_min = __shfl_xor_sync(0xffffffff, ci_min, 8);
    ci_min = nb_min < ci_min ? nb_min : ci_min;
    nb_min = __shfl_xor_sync(0xffffffff, ci_min, 16);
    ci_min = nb_min < ci_min ? nb_min : ci_min;

    if(threadIdx.x < NDIMS) {
        block_bounds_min[blockIdx.x*4 + dim_i_idx] = ci_min;
        block_bounds_max[blockIdx.x*4 + dim_i_idx] = ci_max;
    }

}

template <typename RealType>
void __global__ k_mark_dx_dp(
    const int N,
    const int DP,
    const RealType *dx_dp,
    int *block_markers) {
    // this kernels processes dx_dp to see if we need to compute the dx_dp
    // we can skip A_sub and A_sub.T if the matching dx_dp and dx_dp subs are zsero
    const int gid = blockDim.x*blockIdx.x + threadIdx.x;
    const int atom_i_idx = gid / 4;
    const int dim_i_idx = gid % 4;

    int count = 0;
    for(int p=0; p < DP; p++) {
        if(dx_dp[p*N*4+atom_i_idx*4+dim_i_idx] != 0.0) {
            count += 1;
        };
    }

    int is_dense = __any_sync(0xffffffff, count);

    if(threadIdx.x == 0) {
        block_markers[blockIdx.x] = is_dense;
    }
}

template <typename RealType>
void __global__ k_nonbonded(
    const int N,
    const int DP,
    const RealType *coords, // [N,4]
    const RealType *params, // [N]
    const int *dp_idxs, // [N], can be -1 to N-1, indicating where to scatter the index
    const int *block_markers,
    const RealType *dx_dp, // [N]
    const RealType *block_bounds_min,
    const RealType *block_bounds_max,
    const RealType cutoff,
    RealType *out_hess,
    RealType *out_dx_dp,
    RealType *out_mp) {

    const int gid = blockDim.x*blockIdx.x + threadIdx.x;

    const int atom_i_idx = gid / 4;
    const int dim_i_idx = gid % 4;

    const RealType ci = coords[atom_i_idx*NDIMS + dim_i_idx];
    const RealType qi = params[atom_i_idx];
    const int dp_idx_i = dp_idxs[atom_i_idx];

    __shared__ RealType row_on_diag[4*8*4];
    __shared__ RealType col_on_diag[4*8*4];

    __shared__ RealType off_diag[WARPSIZE*WARPSIZE];
    __shared__ RealType mmul_buf[WARPSIZE];

    const int tile_row = blockIdx.x;
    const int num_y_tiles = blockIdx.x + 1;

    const RealType block_row_min = block_bounds_min[tile_row*4+dim_i_idx];
    const RealType block_row_max = block_bounds_max[tile_row*4+dim_i_idx];
    const RealType block_row_ext = block_row_max - block_row_min;
    const RealType block_row_ctr = block_row_ext + block_row_min;

    const int row_dp_flag = __any_sync(0xffffffff, dp_idx_i >= 0);
    const int tr = block_markers[tile_row] == 0;

    for(int tile_col = 0; tile_col < num_y_tiles; tile_col++) {

        int gjd = tile_col * WARPSIZE + threadIdx.x;
        int shfl_atom_j_idx = gjd / 4;
        int shfl_dp_idx_j = dp_idxs[shfl_atom_j_idx];
        const int col_dp_flag = __any_sync(0xffffffff, shfl_dp_idx_j >= 0);
        const int tc = block_markers[tile_col] == 0;

        // markers for dxdp and p_n
        if((tr && tc) && (row_dp_flag == 0 && col_dp_flag == 0)) {
            continue;
        } 

        const RealType block_col_min = block_bounds_min[tile_col*4+dim_i_idx];
        const RealType block_col_max = block_bounds_max[tile_col*4+dim_i_idx];
        const RealType block_col_ext = block_col_max - block_col_min;
        const RealType block_col_ctr = block_col_ext + block_col_min;

        const RealType block_dij = max(0.0, abs(block_col_ctr-block_row_ctr) - (block_col_ext+block_row_ext));
        RealType block_d2ij = block_dij*block_dij;

        block_d2ij += __shfl_xor_sync(0xffffffff, block_d2ij, 1);
        block_d2ij += __shfl_xor_sync(0xffffffff, block_d2ij, 2);

        if(block_d2ij > cutoff*cutoff) {
            continue;
        }

        // (TODO): reset the off_diag buffer each round
        // don't need to initialize this due to the unique add guarantees
        // so we can eventually remove this
        for(int i=0; i < WARPSIZE; i++) {
            off_diag[i*WARPSIZE+threadIdx.x] = 0;
        }

        for(int d=0; d < 4; d++) {
            row_on_diag[4*threadIdx.x+d] = 0;
            col_on_diag[4*threadIdx.x+d] = 0;
        }

        RealType shfl_cj = coords[gjd];
        RealType shfl_qj = params[shfl_atom_j_idx];

        for(int round = 0; round < 8; round++) {

            const RealType cij = ci - shfl_cj;
            RealType d2ij = cij*cij; // this will be warp-reduced

            // we can probably replace this with a more opportunistic method later if it becomes the bottleneck.
            d2ij += __shfl_xor_sync(0xffffffff, d2ij, 1);
            d2ij += __shfl_xor_sync(0xffffffff, d2ij, 2);

            const RealType qiqj = qi*shfl_qj;

            RealType dij = sqrt(d2ij);
            RealType d3ij = d2ij*dij;
            RealType d5ij = d3ij*d2ij;

            // hessian
            for(int d=0; d < 4; d++) {

                // array([0, 1, 2, 3, 4, 5, 6, 7]) d == 0 
                // array([1, 2, 3, 0, 5, 6, 7, 4]) d == 1
                // array([2, 3, 0, 1, 6, 7, 4, 5]) d == 2
                const int srcLane = (threadIdx.x + d) % NDIMS + (threadIdx.x / NDIMS) * NDIMS;
                const int j_dim = srcLane % 4;
                const RealType shfl_cij = __shfl_sync(0xffffffff, cij, srcLane); // whoa what the fuck 
                const RealType d2u = (3*cij*shfl_cij)/d5ij;

                // we use the if statements here to avoid synchronization issues
                if(shfl_atom_j_idx < atom_i_idx && d2ij < cutoff*cutoff) {
                    off_diag[SHARED_OFF_DIAG_HESS(atom_i_idx, shfl_atom_j_idx, dim_i_idx, j_dim)] = ((dim_i_idx==j_dim)/d3ij - d2u)*qiqj;
                    row_on_diag[SHARED_ON_DIAG_HESS(atom_i_idx, dim_i_idx, j_dim)] += (d2u - (dim_i_idx==j_dim)/d3ij)*qiqj;
                    col_on_diag[SHARED_ON_DIAG_HESS(shfl_atom_j_idx, j_dim, dim_i_idx)] += (d2u - (dim_i_idx==j_dim)/d3ij)*qiqj;
                }
            }

            // mixed_partial derivatives
            // (TODO): move on-diagonals into shared memory for much better performance.
            if(shfl_atom_j_idx < atom_i_idx && d2ij < cutoff*cutoff) {
                if(dp_idx_i >= 0) {
                    atomicAdd(out_mp + dp_idx_i*N*4+atom_i_idx*4+dim_i_idx,  -cij*shfl_qj/d3ij); // d2u/dxi dqi
                    atomicAdd(out_mp + dp_idx_i*N*4+shfl_atom_j_idx*4+dim_i_idx, cij*shfl_qj/d3ij); //d2u/dxj dqi
                }

                if(shfl_dp_idx_j >= 0) {
                    atomicAdd(out_mp + shfl_dp_idx_j*N*4+shfl_atom_j_idx*4+dim_i_idx, cij*qi/d3ij); // d2u/dxj dqj
                    atomicAdd(out_mp + shfl_dp_idx_j*N*4+atom_i_idx*4+dim_i_idx, -cij*qi/d3ij); // d2u/dxi dqj
                }
            }


            // __shfl_sync will modulo warp_size automatically
            shfl_atom_j_idx = __shfl_sync(0xffffffff, shfl_atom_j_idx, threadIdx.x + NDIMS);
            shfl_dp_idx_j = __shfl_sync(0xffffffff, shfl_dp_idx_j, threadIdx.x + NDIMS);
            shfl_cj = __shfl_sync(0xffffffff, shfl_cj, threadIdx.x + NDIMS);
            shfl_qj = __shfl_sync(0xffffffff, shfl_qj, threadIdx.x + NDIMS);

        }


        if(out_hess) {

            // turn on for debugging
            // for(int j=0; j < WARPSIZE; j++) {
            //     int jj_atom = (tile_col*32 + j) / 4;
            //     int jj_dim = j % 4;

            //     if(jj_atom < atom_i_idx) {
            //         if(tile_row == tile_col) {
            //             // on-diagonals need special treatment 
            //             atomicAdd(out_hess + GLOBAL_HESS(atom_i_idx, jj_atom, dim_i_idx, jj_dim, N), off_diag[threadIdx.x*WARPSIZE+j]);
            //             atomicAdd(out_hess + GLOBAL_HESS(jj_atom, atom_i_idx, jj_dim, dim_i_idx, N), off_diag[threadIdx.x*WARPSIZE+j]);
            //         } else {
            //             out_hess[GLOBAL_HESS(atom_i_idx, jj_atom, dim_i_idx, jj_dim, N)] += off_diag[threadIdx.x*WARPSIZE+j];
            //             out_hess[GLOBAL_HESS(jj_atom, atom_i_idx, jj_dim, dim_i_idx, N)] += off_diag[threadIdx.x*WARPSIZE+j];
            //         }
            //     }
            // }

            // on-diagonal writes, accumulate and do this once at the very very end.
            for(int d=0; d < 4; d++) {

                const int srcLane = (threadIdx.x + d) % NDIMS + (threadIdx.x / NDIMS) * NDIMS;
                const int j_dim = srcLane % 4;

                atomicAdd(out_hess + GLOBAL_HESS(atom_i_idx, atom_i_idx, dim_i_idx, j_dim, N), row_on_diag[SHARED_ON_DIAG_HESS(atom_i_idx, dim_i_idx, j_dim)]);
                atomicAdd(out_hess + GLOBAL_HESS(shfl_atom_j_idx, shfl_atom_j_idx, j_dim, dim_i_idx, N), col_on_diag[SHARED_ON_DIAG_HESS(atom_i_idx, j_dim, dim_i_idx)]);

            }

        }

        // off-diagonal matrix multiply
        // (optimization) we need to execute this only and only if dij < cutoff across the board.
        if(dx_dp && out_dx_dp) {
            // for each parameter set
            for(int p=0; p < DP; p++) {

                    mmul_buf[threadIdx.x] = dx_dp[p*N*4+shfl_atom_j_idx*4+dim_i_idx];
                    RealType accum = 0;
                    #pragma unroll
                    for(int i=0; i < WARPSIZE; i++) {
                        accum += off_diag[threadIdx.x*WARPSIZE+i]*mmul_buf[i];
                    }
                    atomicAdd(out_dx_dp + p*N*4+atom_i_idx*4+dim_i_idx, accum);

                    // transpose
                    mmul_buf[threadIdx.x] = dx_dp[p*N*4+atom_i_idx*4+dim_i_idx];
                    accum = 0;
                    #pragma unroll
                    for(int i=0; i < WARPSIZE; i++) {
                        accum += off_diag[i*WARPSIZE+threadIdx.x]*mmul_buf[i];
                    }
                    atomicAdd(out_dx_dp + p*N*4+shfl_atom_j_idx*4+dim_i_idx, accum);
            }
        }

    }
 
}

template <typename RealType>
void __global__ k_on_diag_sgemm(
    const int N,
    const int DP,
    const RealType *hess,
    const RealType *dx_dp,
    RealType *out_dx_dp) {

    __shared__ RealType on_diag[4*8*4];
    __shared__ RealType mmul_buf[WARPSIZE];

    const int gid = blockDim.x*blockIdx.x + threadIdx.x;

    const int atom_i_idx = gid / 4;
    const int dim_i_idx = gid % 4;

    // load diagonal elements into shared memory
    for(int d=0; d < 4; d++) {
        on_diag[(threadIdx.x/4)*4*4+dim_i_idx*4+d] = hess[GLOBAL_HESS(atom_i_idx, atom_i_idx, dim_i_idx, d, N)];
    }

    for(int p=0; p < DP; p++) {
        mmul_buf[threadIdx.x] = dx_dp[p*N*4+atom_i_idx*4+dim_i_idx];
        RealType accum = 0;
        for(int d=0; d < 4; d++) {
            accum += on_diag[(threadIdx.x/4)*4*4+dim_i_idx*4+d]*mmul_buf[(threadIdx.x/4)*4+d];
        }
        atomicAdd(out_dx_dp + p*N*4+atom_i_idx*4+dim_i_idx, accum);
    }
}
