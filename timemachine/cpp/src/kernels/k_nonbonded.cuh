#pragma once

#include "../fixed_point.hpp"
#include "../gpu_utils.cuh"
#include "k_nonbonded_common.cuh"
#include "kernel_utils.cuh"

namespace timemachine {

template <typename RealType>
void __global__ k_check_rebuild_coords_and_box_gather(
    const int N,
    const unsigned int *__restrict__ atom_idxs,
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

template <typename RealType, int COORDS_DIM, int PARAMS_DIM>
void __global__ k_gather_coords_and_params(
    const int N,
    const unsigned int *__restrict__ idxs,
    const RealType *__restrict__ coords,
    const RealType *__restrict__ params,
    RealType *__restrict__ gathered_coords,
    RealType *__restrict__ gathered_params) {
    static_assert(COORDS_DIM == 3);
    static_assert(PARAMS_DIM == PARAMS_PER_ATOM);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    const unsigned int atom_idx = idxs[idx];

    // Coords have 3 dimensions, params have 4
#pragma unroll COORDS_DIM
    for (int i = 0; i < COORDS_DIM; i++) {
        gathered_coords[idx * COORDS_DIM + i] = coords[atom_idx * COORDS_DIM + i];
    }
#pragma unroll PARAMS_DIM
    for (int i = 0; i < PARAMS_DIM; i++) {
        gathered_params[idx * PARAMS_DIM + i] = params[atom_idx * PARAMS_DIM + i];
    }
}

template <typename T, int D>
void __global__ k_scatter_accum(
    const int N,
    const unsigned int *__restrict__ unique_idxs, // NOTE: race condition possible if there are repeated indices
    const T *__restrict__ gathered_array,
    T *__restrict__ array) {
    static_assert(D <= 5, "More loop unrolling than expected");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }
    const unsigned int dest_idx = unique_idxs[idx];

#pragma unroll D
    for (int i = 0; i < D; i++) {
        atomicAdd(array + (dest_idx * D + i), gathered_array[idx * D + i]);
    }
}

// ALCHEMICAL == false guarantees that the tile's atoms are such that
// 1. src_param and dst_params are equal for every i in R and j in C
// 2. w_i and w_j are identical for every (i,j) in (RxC)
template <typename RealType, bool ALCHEMICAL, bool COMPUTE_U, bool COMPUTE_DU_DX, bool COMPUTE_DU_DP>
// void __device__ __forceinline__ v_nonbonded_unified(
void __device__ v_nonbonded_unified(
    const int tile_idx,
    const int N,
    const int NR,
    const double *__restrict__ coords, // [N * 3]
    const double *__restrict__ params, // [N * PARAMS_PER_ATOM]
    box_cache<RealType> &shared_box,
    __int128 *__restrict__ energy_buffer, // [blockDim.x]
    const double beta,
    const double cutoff,
    const unsigned int *__restrict__ row_idxs,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp) {

    int row_block_idx = ixn_tiles[tile_idx];

    const int warp_idx = threadIdx.x % WARP_SIZE;

    const int index = row_block_idx * WARP_SIZE + warp_idx;
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
    int atom_j_idx = ixn_atoms[tile_idx * WARP_SIZE + warp_idx];

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

    RealType real_beta = static_cast<RealType>(beta);

    const int src_lane = (warp_idx + 1) % WARP_SIZE; // fixed
    // #pragma unroll
    for (int round = 0; round < WARP_SIZE; round++) {

        RealType delta_x = ci_x - cj_x;
        RealType delta_y = ci_y - cj_y;
        RealType delta_z = ci_z - cj_z;

        delta_x -= shared_box.x * nearbyint(delta_x * shared_box.inv_x);
        delta_y -= shared_box.y * nearbyint(delta_y * shared_box.inv_y);
        delta_z -= shared_box.z * nearbyint(delta_z * shared_box.inv_z);

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
                1.0, qi, qj, d2ij, real_beta, dij, inv_dij, inv_d2ij, ebd, es_prefactor, u);

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
                energy_buffer[threadIdx.x] += FLOAT_TO_FIXED_ENERGY<RealType>(u);
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
}

template <typename RealType, int THREADS_PER_BLOCK, bool COMPUTE_U, bool COMPUTE_DU_DX, bool COMPUTE_DU_DP>
void __global__ k_nonbonded_unified(
    const int N,  // Number of atoms
    const int NR, // Number of row indices
    const unsigned int *__restrict__ ixn_count,
    const double *__restrict__ coords, // [N, 3]
    const double *__restrict__ params, // [N, PARAMS_PER_ATOM]
    const double *__restrict__ box,    // [3, 3]
    const double beta,
    const double cutoff,
    const unsigned int *__restrict__ row_idxs,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    __int128 *__restrict__ u_buffer // [blockDim.x]
) {
    static_assert(THREADS_PER_BLOCK <= 256 && (THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0);
    __shared__ box_cache<RealType> shared_box;
    __shared__ __int128 block_energy_buffer[THREADS_PER_BLOCK];
    if (COMPUTE_U) {
        block_energy_buffer[threadIdx.x] = 0; // Zero out the energy buffer
    }
    if (threadIdx.x == 0) {
        shared_box.x = box[0 * 3 + 0];
        shared_box.y = box[1 * 3 + 1];
        shared_box.z = box[2 * 3 + 2];
        shared_box.inv_x = 1 / shared_box.x;
        shared_box.inv_y = 1 / shared_box.y;
        shared_box.inv_z = 1 / shared_box.z;
    }
    __syncthreads();
    // Tile size is the same as warp size but it doesn't have to be.
    // Can be used interchangably at the moment, but in the future we may have different
    // tile sizes.
    const int tile_size = WARP_SIZE;

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
                block_energy_buffer,
                beta,
                cutoff,
                row_idxs,
                ixn_tiles,
                ixn_atoms,
                du_dx,
                du_dp);
        } else {
            v_nonbonded_unified<RealType, 1, COMPUTE_U, COMPUTE_DU_DX, COMPUTE_DU_DP>(
                tile_idx,
                N,
                NR,
                coords,
                params,
                shared_box,
                block_energy_buffer,
                beta,
                cutoff,
                row_idxs,
                ixn_tiles,
                ixn_atoms,
                du_dx,
                du_dp);
        };
        tile_idx += stride;
    }
    if (COMPUTE_U) {
        // Sync to ensure the shared buffers are populated
        __syncthreads();

        block_energy_reduce<THREADS_PER_BLOCK>(block_energy_buffer, threadIdx.x);

        if (threadIdx.x == 0) {
            u_buffer[blockIdx.x] = block_energy_buffer[0];
        }
    }
}

template <typename RealType, int THREADS_PER_BLOCK>
void __global__ k_compute_nonbonded_target_atom_energies(
    const int N,
    const int num_target_atoms,
    const int *__restrict__ target_atoms,        // [num_target_atoms]
    const int *__restrict__ target_mols,         // [num_target_atoms]
    const int *__restrict__ target_mols_offsets, // [num_mols + 1]
    const double *__restrict__ coords,           // [N, 3]
    const double *__restrict__ params,           // [N, PARAMS_PER_ATOM]
    const double *__restrict__ box,              // [3, 3],
    const RealType beta,
    const RealType cutoff_squared,
    __int128 *__restrict__ output_energies // [num_target_atoms, gridDim.x]
) {
    static_assert(THREADS_PER_BLOCK <= 256 && (THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0);
    __shared__ __int128 block_energy_buffer[THREADS_PER_BLOCK];

    const RealType bx = box[0 * 3 + 0];
    const RealType by = box[1 * 3 + 1];
    const RealType bz = box[2 * 3 + 2];

    const RealType inv_bx = 1 / bx;
    const RealType inv_by = 1 / by;
    const RealType inv_bz = 1 / bz;
    int row_idx = blockIdx.y;
    while (row_idx < num_target_atoms) {

        int atom_i_idx = target_atoms[row_idx];
        int mol_i_idx = target_mols[row_idx];

        int min_mol_offset = target_mols_offsets[mol_i_idx];
        int max_mol_offset = target_mols_offsets[mol_i_idx + 1];
        int min_atom_idx = target_atoms[min_mol_offset];
        int max_atom_idx = target_atoms[max_mol_offset - 1];

        int params_i_idx = atom_i_idx * PARAMS_PER_ATOM;
        int charge_param_idx_i = params_i_idx + PARAM_OFFSET_CHARGE;
        int lj_param_idx_sig_i = params_i_idx + PARAM_OFFSET_SIG;
        int lj_param_idx_eps_i = params_i_idx + PARAM_OFFSET_EPS;
        int w_param_idx_i = params_i_idx + PARAM_OFFSET_W;

        RealType qi = params[charge_param_idx_i];
        RealType sig_i = params[lj_param_idx_sig_i];
        RealType eps_i = params[lj_param_idx_eps_i];
        RealType w_i = params[w_param_idx_i];

        RealType ci_x = coords[atom_i_idx * 3 + 0];
        RealType ci_y = coords[atom_i_idx * 3 + 1];
        RealType ci_z = coords[atom_i_idx * 3 + 2];

        int atom_j_idx = blockIdx.x * blockDim.x + threadIdx.x;
        // All threads in the threadblock must loop to allow for __syncthreads() and the row accumulation.
        while (atom_j_idx - threadIdx.x < N) {
            // Zero out the energy buffer
            block_energy_buffer[threadIdx.x] = 0;
            // The two atoms are in the same molecule, don't compute the energies
            // requires that the atom indices in each target mol is consecutive
            if (atom_j_idx < N && (atom_j_idx < min_atom_idx || atom_j_idx > max_atom_idx)) {

                int params_j_idx = atom_j_idx * PARAMS_PER_ATOM;
                int charge_param_idx_j = params_j_idx + PARAM_OFFSET_CHARGE;
                int lj_param_idx_sig_j = params_j_idx + PARAM_OFFSET_SIG;
                int lj_param_idx_eps_j = params_j_idx + PARAM_OFFSET_EPS;
                int w_param_idx_j = params_j_idx + PARAM_OFFSET_W;

                RealType qj = params[charge_param_idx_j];
                RealType sig_j = params[lj_param_idx_sig_j];
                RealType eps_j = params[lj_param_idx_eps_j];
                RealType w_j = params[w_param_idx_j];

                RealType cj_x = coords[atom_j_idx * 3 + 0];
                RealType cj_y = coords[atom_j_idx * 3 + 1];
                RealType cj_z = coords[atom_j_idx * 3 + 2];

                RealType delta_x = ci_x - cj_x;
                RealType delta_y = ci_y - cj_y;
                RealType delta_z = ci_z - cj_z;
                RealType delta_w = w_i - w_j;

                delta_x -= bx * nearbyint(delta_x * inv_bx);
                delta_y -= by * nearbyint(delta_y * inv_by);
                delta_z -= bz * nearbyint(delta_z * inv_bz);

                RealType d2ij = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z + delta_w * delta_w;

                if (d2ij < cutoff_squared) {
                    RealType u;
                    RealType delta_prefactor;
                    RealType ebd;
                    RealType dij;
                    RealType inv_dij;
                    RealType inv_d2ij;
                    compute_electrostatics<RealType, true>(
                        1.0, qi, qj, d2ij, beta, dij, inv_dij, inv_d2ij, ebd, delta_prefactor, u);
                    // lennard jones energy
                    if (eps_i != 0 && eps_j != 0) {
                        RealType sig_grad;
                        RealType eps_grad;
                        compute_lj<RealType, true>(
                            1.0, eps_i, eps_j, sig_i, sig_j, inv_dij, inv_d2ij, u, delta_prefactor, sig_grad, eps_grad);
                    }
                    // Store the atom by atom energy
                    block_energy_buffer[threadIdx.x] = FLOAT_TO_FIXED_ENERGY<RealType>(u);
                }
            }
            // Sync to ensure the shared buffers are populated
            __syncthreads();

            block_energy_reduce<THREADS_PER_BLOCK>(block_energy_buffer, threadIdx.x);

            if (threadIdx.x == 0) {
                output_energies[row_idx * gridDim.x + blockIdx.x] += block_energy_buffer[0];
            }
            // Sync the threads so threads don't move on and stomp on the block energy buffer
            __syncthreads();

            atom_j_idx += gridDim.x * blockDim.x;
        }
        row_idx += gridDim.y * blockDim.y;
    }
}

// NUM_BLOCKS is the number of blocks that k_compute_nonbonded_target_atom_energies is run with. Decides the number
// of values that need to be accumulated per atom. Also dictates the number of threads
template <typename RealType, int NUM_BLOCKS>
void __global__ k_accumulate_atom_energies_to_per_mol_energies(
    const int target_mols,
    const int *__restrict__ mol_offsets,            // [target_mols + 1]
    const __int128 *__restrict__ per_atom_energies, // [target_atoms, NUM_BLOCKS]
    __int128 *__restrict__ per_mol_energies) {

    int mol_idx = blockIdx.x;

    static_assert(NUM_BLOCKS <= 256 && (NUM_BLOCKS & (NUM_BLOCKS - 1)) == 0);
    __shared__ __int128 block_energy_buffer[NUM_BLOCKS];

    while (mol_idx < target_mols) {
        __int128 local_accumulator = 0;
        const int mol_start = mol_offsets[mol_idx];
        const int mol_end = mol_offsets[mol_idx + 1];

        int idx = (mol_start * NUM_BLOCKS) + threadIdx.x;
        while (idx < mol_end * NUM_BLOCKS) {
            local_accumulator += per_atom_energies[idx];

            idx += blockDim.x;
        }
        block_energy_buffer[threadIdx.x] = local_accumulator;
        __syncthreads();
        block_energy_reduce<NUM_BLOCKS>(block_energy_buffer, threadIdx.x);

        if (threadIdx.x == 0) {
            per_mol_energies[mol_idx] = block_energy_buffer[0];
        }

        mol_idx += gridDim.x;
    }
}

// k_atom_by_atom_energies is intended to be used for computing the energies of a subset of atoms
// against all other atoms. The kernel allows changing the positions of the target atoms by passing in an
// array for target_coords, if a nullptr is provided it will use the coords array for the positions of the target
// atoms. This allows modification of the positions of a subset of atoms, avoiding the need to duplicating all of the
// coordinates.
template <typename RealType>
void __global__ k_atom_by_atom_energies(
    const int N,
    const int num_target_atoms,
    const int *__restrict__ target_atoms, // [num_target_atoms]
    const double
        *__restrict__ target_coords,   // [num_target_atoms, 3] Can be nullptr if should use coords for the target atoms
    const double *__restrict__ coords, // [N, 3]
    const double *__restrict__ params, // [N, PARAMS_PER_ATOM]
    const double *__restrict__ box,    // [3, 3],
    const RealType beta,
    const RealType cutoff_squared,
    RealType *__restrict__ output_energies // [num_target_atoms, N]
) {
    const RealType bx = box[0 * 3 + 0];
    const RealType by = box[1 * 3 + 1];
    const RealType bz = box[2 * 3 + 2];

    const RealType inv_bx = 1 / bx;
    const RealType inv_by = 1 / by;
    const RealType inv_bz = 1 / bz;
    int row_idx = blockIdx.y;
    while (row_idx < num_target_atoms) {

        int atom_i_idx = target_atoms[row_idx];

        RealType ci_x = target_coords != nullptr ? target_coords[row_idx * 3 + 0] : coords[atom_i_idx * 3 + 0];
        RealType ci_y = target_coords != nullptr ? target_coords[row_idx * 3 + 1] : coords[atom_i_idx * 3 + 1];
        RealType ci_z = target_coords != nullptr ? target_coords[row_idx * 3 + 2] : coords[atom_i_idx * 3 + 2];

        int params_i_idx = atom_i_idx * PARAMS_PER_ATOM;
        int charge_param_idx_i = params_i_idx + PARAM_OFFSET_CHARGE;
        int lj_param_idx_sig_i = params_i_idx + PARAM_OFFSET_SIG;
        int lj_param_idx_eps_i = params_i_idx + PARAM_OFFSET_EPS;
        int w_param_idx_i = params_i_idx + PARAM_OFFSET_W;

        RealType qi = params[charge_param_idx_i];
        RealType sig_i = params[lj_param_idx_sig_i];
        RealType eps_i = params[lj_param_idx_eps_i];
        RealType w_i = params[w_param_idx_i];

        int atom_j_idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (atom_j_idx < N) {
            RealType u = 0.0;

            int params_j_idx = atom_j_idx * PARAMS_PER_ATOM;
            int charge_param_idx_j = params_j_idx + PARAM_OFFSET_CHARGE;
            int lj_param_idx_sig_j = params_j_idx + PARAM_OFFSET_SIG;
            int lj_param_idx_eps_j = params_j_idx + PARAM_OFFSET_EPS;
            int w_param_idx_j = params_j_idx + PARAM_OFFSET_W;

            RealType qj = params[charge_param_idx_j];
            RealType sig_j = params[lj_param_idx_sig_j];
            RealType eps_j = params[lj_param_idx_eps_j];
            RealType w_j = params[w_param_idx_j];

            RealType cj_x = coords[atom_j_idx * 3 + 0];
            RealType cj_y = coords[atom_j_idx * 3 + 1];
            RealType cj_z = coords[atom_j_idx * 3 + 2];

            RealType delta_x = ci_x - cj_x;
            RealType delta_y = ci_y - cj_y;
            RealType delta_z = ci_z - cj_z;
            RealType delta_w = w_i - w_j;

            delta_x -= bx * nearbyint(delta_x * inv_bx);
            delta_y -= by * nearbyint(delta_y * inv_by);
            delta_z -= bz * nearbyint(delta_z * inv_bz);

            RealType d2ij = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z + delta_w * delta_w;

            if (d2ij < cutoff_squared) {
                RealType delta_prefactor;
                RealType ebd;
                RealType dij;
                RealType inv_dij;
                RealType inv_d2ij;
                compute_electrostatics<RealType, true>(
                    1.0, qi, qj, d2ij, beta, dij, inv_dij, inv_d2ij, ebd, delta_prefactor, u);

                // lennard jones force
                if (eps_i != 0 && eps_j != 0) {
                    RealType sig_grad;
                    RealType eps_grad;
                    compute_lj<RealType, true>(
                        1.0, eps_i, eps_j, sig_i, sig_j, inv_dij, inv_d2ij, u, delta_prefactor, sig_grad, eps_grad);
                }
            }
            // Store the atom by atom energy, can be in floating point since there is no accumulation in the kernel
            output_energies[row_idx * N + atom_j_idx] = u;
            atom_j_idx += gridDim.x * blockDim.x;
        }
        row_idx += gridDim.y * blockDim.y;
    }
}

} // namespace timemachine
