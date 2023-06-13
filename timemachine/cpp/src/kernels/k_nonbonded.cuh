#pragma once

#include "../fixed_point.hpp"
#include "k_nonbonded_common.cuh"
#include "kernel_utils.cuh"

// Divide [0,1]^3 box into HILBERT_GRID_DIM^3 voxels for Hilbert sort
static const int HILBERT_GRID_DIM = 128;

// Encode grid index along each dimension using HILBERT_N_BITS
static const int HILBERT_N_BITS = 8;
static const int HILBERT_MAX_GRID_DIM = 1 << HILBERT_N_BITS;
static_assert(HILBERT_N_BITS == 8);
static_assert(HILBERT_GRID_DIM <= HILBERT_MAX_GRID_DIM);

// generate kv values from coordinates to be radix sorted allowing the selection of a subset of coordinates
void __global__ k_coords_to_kv_gather(
    const int N,
    const unsigned int *__restrict__ atom_idxs,
    const double *__restrict__ coords,
    const double *__restrict__ box,
    const unsigned int *__restrict__ bin_to_idx,
    unsigned int *__restrict__ keys,
    unsigned int *__restrict__ vals);

template <typename RealType>
void __global__ k_check_rebuild_coords_and_box_gather(
    const int N,
    const unsigned int *atom_idxs,
    const double *__restrict__ new_coords,
    const double *__restrict__ old_coords,
    const double *__restrict__ new_box,
    const double *__restrict__ old_box,
    const double padding,
    int *rebuild_flag) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 9) {
        // (ytz): box vectors have exactly 9 components
        // we can probably derive a looser bound later on.
        if (old_box[idx] != new_box[idx]) {
            rebuild_flag[0] = 1;
            return;
        }
    }

    if (idx >= N) {
        return;
    }

    const int atom_idx = atom_idxs[idx];

    RealType xi = old_coords[atom_idx * 3 + 0];
    RealType yi = old_coords[atom_idx * 3 + 1];
    RealType zi = old_coords[atom_idx * 3 + 2];

    RealType xj = new_coords[atom_idx * 3 + 0];
    RealType yj = new_coords[atom_idx * 3 + 1];
    RealType zj = new_coords[atom_idx * 3 + 2];

    RealType dx = xi - xj;
    RealType dy = yi - yj;
    RealType dz = zi - zj;

    RealType d2ij = dx * dx + dy * dy + dz * dz;
    if (d2ij > static_cast<RealType>(0.25) * padding * padding) {
        // (ytz): this is *safe* but technically is a race condition
        rebuild_flag[0] = 1;
    }
}

template <typename RealType>
void __global__ k_gather_coords_and_params(
    const int N,
    const unsigned int *__restrict__ idxs,
    const RealType *__restrict__ coords,
    const RealType *__restrict__ params,
    RealType *__restrict__ gathered_coords,
    RealType *__restrict__ gathered_params) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if (idx >= N) {
        return;
    }

    // Coords have 3 dimensions, params have 4
    if (stride_idx < 3) {
        gathered_coords[idx * 3 + stride_idx] = coords[idxs[idx] * 3 + stride_idx];
    }
    gathered_params[idx * stride + stride_idx] = params[idxs[idx] * stride + stride_idx];
}

template <typename RealType>
void __global__ k_scatter_accum(
    const int N,
    const unsigned int *__restrict__ unique_idxs, // NOTE: race condition possible if there are repeated indices
    const RealType *__restrict__ gathered_array,
    RealType *__restrict__ array) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if (idx >= N) {
        return;
    }

    atomicAdd(array + (unique_idxs[idx] * stride + stride_idx), gathered_array[idx * stride + stride_idx]);
}

template <typename RealType>
void __global__ k_scatter_assign(
    const int N,
    const unsigned int *__restrict__ unique_idxs, // NOTE: race condition possible if there are repeated indices
    const RealType *__restrict__ gathered_array,
    RealType *__restrict__ array) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if (idx >= N) {
        return;
    }

    array[unique_idxs[idx] * stride + stride_idx] = gathered_array[idx * stride + stride_idx];
}

template <typename RealType>
void __global__ k_scatter_assign_2x(
    const int N,
    const unsigned int *__restrict__ unique_idxs, // NOTE: race condition possible if there are repeated indices
    const RealType *__restrict__ gathered_array_1,
    const RealType *__restrict__ gathered_array_2,
    RealType *__restrict__ array_1,
    RealType *__restrict__ array_2) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if (idx >= N) {
        return;
    }

    array_1[unique_idxs[idx] * stride + stride_idx] = gathered_array_1[idx * stride + stride_idx];
    array_2[unique_idxs[idx] * stride + stride_idx] = gathered_array_2[idx * stride + stride_idx];
}

template <typename RealType> void __global__ k_reduce_buffer(int N, RealType *d_buffer, RealType *d_sum) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    RealType elem = idx < N ? d_buffer[idx] : 0;

    atomicAdd(d_sum, elem);
};

template <typename RealType> void __global__ k_reduce_ull_buffer(int N, unsigned long long *d_buffer, RealType *d_sum) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    RealType elem = idx < N ? FIXED_TO_FLOAT<RealType>(d_buffer[idx]) : 0;

    atomicAdd(d_sum, elem);
};

// ALCHEMICAL == false guarantees that the tile's atoms are such that
// 1. src_param and dst_params are equal for every i in R and j in C
// 2. w_i and w_j are identical for every (i,j) in (RxC)
template <typename RealType, bool ALCHEMICAL, bool COMPUTE_U, bool COMPUTE_DU_DX, bool COMPUTE_DU_DP>
// void __device__ __forceinline__ v_nonbonded_unified(
void __device__ v_nonbonded_unified(
    const int tile_idx,
    const int N,
    const int NR,
    const double *__restrict__ coords,
    const double *__restrict__ params, // [N]
    box_cache<RealType> &shared_box,   // [6]
    const double beta,
    const double cutoff,
    const unsigned int *__restrict__ row_idxs,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ u_buffer) {

    RealType box_x = shared_box.x;
    RealType box_y = shared_box.y;
    RealType box_z = shared_box.z;

    RealType inv_box_x = shared_box.inv_x;
    RealType inv_box_y = shared_box.inv_y;
    RealType inv_box_z = shared_box.inv_z;

    int row_block_idx = ixn_tiles[tile_idx];

    const int warp_idx = threadIdx.x % warp_size;

    const int index = row_block_idx * warp_size + warp_idx;
    const unsigned int atom_i_idx = index < NR ? row_idxs[index] : N;

    RealType ci_x = atom_i_idx < N ? coords[atom_i_idx * 3 + 0] : 0;
    RealType ci_y = atom_i_idx < N ? coords[atom_i_idx * 3 + 1] : 0;
    RealType ci_z = atom_i_idx < N ? coords[atom_i_idx * 3 + 2] : 0;

    unsigned long long gi_x = 0;
    unsigned long long gi_y = 0;
    unsigned long long gi_z = 0;

    int params_i_idx = atom_i_idx * PARAMS_PER_ATOM;
    int charge_param_idx_i = params_i_idx + PARAM_OFFSET_CHARGE;
    int lj_param_idx_sig_i = params_i_idx + PARAM_OFFSET_SIG;
    int lj_param_idx_eps_i = params_i_idx + PARAM_OFFSET_EPS;
    int w_param_idx_i = params_i_idx + PARAM_OFFSET_W;

    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType sig_i = atom_i_idx < N ? params[lj_param_idx_sig_i] : 0;
    RealType eps_i = atom_i_idx < N ? params[lj_param_idx_eps_i] : 0;
    RealType w_i = atom_i_idx < N ? params[w_param_idx_i] : 0;

    unsigned long long g_qi = 0;
    unsigned long long g_sigi = 0;
    unsigned long long g_epsi = 0;
    unsigned long long g_wi = 0;

    // i idx is contiguous but j is not, so we should swap them to avoid having to shuffle atom_j_idx
    int atom_j_idx = ixn_atoms[tile_idx * warp_size + warp_idx];

    RealType cj_x = atom_j_idx < N ? coords[atom_j_idx * 3 + 0] : 0;
    RealType cj_y = atom_j_idx < N ? coords[atom_j_idx * 3 + 1] : 0;
    RealType cj_z = atom_j_idx < N ? coords[atom_j_idx * 3 + 2] : 0;

    unsigned long long gj_x = 0;
    unsigned long long gj_y = 0;
    unsigned long long gj_z = 0;

    int params_j_idx = atom_j_idx * PARAMS_PER_ATOM;
    int charge_param_idx_j = params_j_idx + PARAM_OFFSET_CHARGE;
    int lj_param_idx_sig_j = params_j_idx + PARAM_OFFSET_SIG;
    int lj_param_idx_eps_j = params_j_idx + PARAM_OFFSET_EPS;
    int w_param_idx_j = params_j_idx + PARAM_OFFSET_W;

    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    RealType sig_j = atom_j_idx < N ? params[lj_param_idx_sig_j] : 0;
    RealType eps_j = atom_j_idx < N ? params[lj_param_idx_eps_j] : 0;
    RealType w_j = atom_j_idx < N ? params[w_param_idx_j] : 0;

    unsigned long long g_qj = 0;
    unsigned long long g_sigj = 0;
    unsigned long long g_epsj = 0;
    unsigned long long g_wj = 0;

    RealType real_cutoff = static_cast<RealType>(cutoff);
    RealType cutoff_squared = real_cutoff * real_cutoff;

    unsigned long long energy = 0;

    RealType real_beta = static_cast<RealType>(beta);

    const int src_lane = (warp_idx + 1) % warp_size; // fixed
    // #pragma unroll
    for (int round = 0; round < warp_size; round++) {

        RealType delta_x = ci_x - cj_x;
        RealType delta_y = ci_y - cj_y;
        RealType delta_z = ci_z - cj_z;

        delta_x -= box_x * nearbyint(delta_x * inv_box_x);
        delta_y -= box_y * nearbyint(delta_y * inv_box_y);
        delta_z -= box_z * nearbyint(delta_z * inv_box_z);

        RealType d2ij = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
        RealType delta_w;

        if (ALCHEMICAL) {
            // (ytz): we are guaranteed that delta_w is zero if ALCHEMICAL == false
            delta_w = w_i - w_j;
            d2ij += delta_w * delta_w;
        }

        // All idxs must be smaller than N and if N == NR then we are doing upper triangle and thus atom_i_idx
        // must be less than atom_j_idx
        const bool valid_ij = atom_i_idx < N && atom_j_idx < N && (N != NR || atom_i_idx < atom_j_idx);

        // (ytz): note that d2ij must be *strictly* less than cutoff_squared. This is because we set the
        // non-interacting atoms to exactly real_cutoff*real_cutoff. This ensures that atoms who's 4th dimension
        // is set to cutoff are non-interacting.
        if (valid_ij && d2ij < cutoff_squared) {
            // electrostatics
            RealType u;
            RealType es_prefactor;
            RealType ebd;
            RealType dij;
            RealType inv_dij;
            RealType inv_d2ij;
            compute_electrostatics<RealType, COMPUTE_U>(
                1.0, qi, qj, d2ij, beta, dij, inv_dij, inv_d2ij, ebd, es_prefactor, u);

            RealType delta_prefactor = es_prefactor;

            // lennard jones force
            if (eps_i != 0 && eps_j != 0) {
                RealType sig_grad;
                RealType eps_grad;
                compute_lj<RealType, COMPUTE_U>(
                    1.0, eps_i, eps_j, sig_i, sig_j, inv_dij, inv_d2ij, u, delta_prefactor, sig_grad, eps_grad);

                // do chain rule inside loop
                if (COMPUTE_DU_DP) {
                    g_sigi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad);
                    g_sigj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad);
                    g_epsi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(eps_grad * eps_j);
                    g_epsj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(eps_grad * eps_i);
                }
            }

            if (COMPUTE_DU_DX) {
                gi_x += FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_x);
                gi_y += FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_y);
                gi_z += FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_z);

                gj_x += FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_x);
                gj_y += FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_y);
                gj_z += FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_z);
            }

            if (COMPUTE_DU_DP) {
                g_qi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(qj * inv_dij * ebd);
                g_qj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(qi * inv_dij * ebd);

                if (ALCHEMICAL) {
                    g_wi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(delta_prefactor * delta_w);
                    g_wj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(-delta_prefactor * delta_w);
                }
            }

            if (COMPUTE_U) {
                energy += FLOAT_TO_FIXED_NONBONDED(u);
            }
        }

        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, src_lane); // we can pre-compute this probably
        qj = __shfl_sync(0xffffffff, qj, src_lane);
        eps_j = __shfl_sync(0xffffffff, eps_j, src_lane);
        sig_j = __shfl_sync(0xffffffff, sig_j, src_lane);

        cj_x = __shfl_sync(0xffffffff, cj_x, src_lane);
        cj_y = __shfl_sync(0xffffffff, cj_y, src_lane);
        cj_z = __shfl_sync(0xffffffff, cj_z, src_lane);

        if (ALCHEMICAL) {
            w_j = __shfl_sync(0xffffffff, w_j, src_lane); // this also can be optimized away
        }

        if (COMPUTE_DU_DX) {
            gj_x = __shfl_sync(0xffffffff, gj_x, src_lane);
            gj_y = __shfl_sync(0xffffffff, gj_y, src_lane);
            gj_z = __shfl_sync(0xffffffff, gj_z, src_lane);
        }

        if (COMPUTE_DU_DP) {
            g_qj = __shfl_sync(0xffffffff, g_qj, src_lane);
            g_sigj = __shfl_sync(0xffffffff, g_sigj, src_lane);
            g_epsj = __shfl_sync(0xffffffff, g_epsj, src_lane);
            g_wj = __shfl_sync(0xffffffff, g_wj, src_lane);
        }
    }

    if (COMPUTE_DU_DX) {
        if (atom_i_idx < N) {
            atomicAdd(du_dx + atom_i_idx * 3 + 0, gi_x);
            atomicAdd(du_dx + atom_i_idx * 3 + 1, gi_y);
            atomicAdd(du_dx + atom_i_idx * 3 + 2, gi_z);
        }
        if (atom_j_idx < N) {
            atomicAdd(du_dx + atom_j_idx * 3 + 0, gj_x);
            atomicAdd(du_dx + atom_j_idx * 3 + 1, gj_y);
            atomicAdd(du_dx + atom_j_idx * 3 + 2, gj_z);
        }
    }

    if (COMPUTE_DU_DP) {
        if (atom_i_idx < N) {
            atomicAdd(du_dp + charge_param_idx_i, g_qi);
            atomicAdd(du_dp + lj_param_idx_sig_i, g_sigi);
            atomicAdd(du_dp + lj_param_idx_eps_i, g_epsi);
            atomicAdd(du_dp + w_param_idx_i, g_wi);
        }

        if (atom_j_idx < N) {
            atomicAdd(du_dp + charge_param_idx_j, g_qj);
            atomicAdd(du_dp + lj_param_idx_sig_j, g_sigj);
            atomicAdd(du_dp + lj_param_idx_eps_j, g_epsj);
            atomicAdd(du_dp + w_param_idx_j, g_wj);
        }
    }

    if (COMPUTE_U) {
        if (atom_i_idx < N) {
            atomicAdd(u_buffer + atom_i_idx, energy);
        }
    }
}

template <typename RealType, bool COMPUTE_U, bool COMPUTE_DU_DX, bool COMPUTE_DU_DP>
void __global__ k_nonbonded_unified(
    const int N,  // Number of atoms
    const int NR, // Number of row indices
    const unsigned int *ixn_count,
    const double *__restrict__ coords,
    const double *__restrict__ params, // [N]
    const double *__restrict__ box,
    const double beta,
    const double cutoff,
    const unsigned int *__restrict__ row_idxs,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ u_buffer) {
    __shared__ box_cache<RealType> shared_box;
    if (threadIdx.x == 0) {
        shared_box.x = box[0 * 3 + 0];
        shared_box.y = box[1 * 3 + 1];
        shared_box.z = box[2 * 3 + 2];
        shared_box.inv_x = 1 / shared_box.x;
        shared_box.inv_y = 1 / shared_box.y;
        shared_box.inv_z = 1 / shared_box.z;
    }
    __syncthreads();
    // Tiles are 32 x 32, which is the same as the warp size
    const int tile_size = warp_size;

    const int tiles_per_block = blockDim.x / tile_size;
    const int stride = gridDim.x * tiles_per_block;
    int tile_idx = blockIdx.x * tiles_per_block + (threadIdx.x / tile_size);

    const int tile_offset = threadIdx.x % tile_size;

    const unsigned int interactions = ixn_count[0];
    while (tile_idx < interactions) {

        int row_block_idx = ixn_tiles[tile_idx];
        int index = row_block_idx * tile_size + tile_offset;
        int atom_i_idx = index < NR ? row_idxs[index] : N;
        int atom_j_idx = ixn_atoms[tile_idx * tile_size + tile_offset];

        RealType w_i = atom_i_idx < N ? params[atom_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_W] : 0;
        RealType w_j = atom_j_idx < N ? params[atom_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_W] : 0;

        int is_vanilla = w_i == 0 && w_j == 0;

        bool tile_is_vanilla = __all_sync(0xffffffff, is_vanilla);

        if (tile_is_vanilla) {
            v_nonbonded_unified<RealType, 0, COMPUTE_U, COMPUTE_DU_DX, COMPUTE_DU_DP>(
                tile_idx,
                N,
                NR,
                coords,
                params,
                shared_box,
                beta,
                cutoff,
                row_idxs,
                ixn_tiles,
                ixn_atoms,
                du_dx,
                du_dp,
                u_buffer);
        } else {
            v_nonbonded_unified<RealType, 1, COMPUTE_U, COMPUTE_DU_DX, COMPUTE_DU_DP>(
                tile_idx,
                N,
                NR,
                coords,
                params,
                shared_box,
                beta,
                cutoff,
                row_idxs,
                ixn_tiles,
                ixn_atoms,
                du_dx,
                du_dp,
                u_buffer);
        };
        tile_idx += stride;
    }
}
