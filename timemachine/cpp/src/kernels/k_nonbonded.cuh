#pragma once

#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
#include "nonbonded_common.cuh"
#include "surreal.cuh"

// Divide [0,1]^3 box into HILBERT_GRID_DIM^3 voxels for Hilbert sort
static const int HILBERT_GRID_DIM = 128;

// Encode grid index along each dimension using HILBERT_N_BITS
static const int HILBERT_N_BITS = 8;
static const int HILBERT_MAX_GRID_DIM = 1 << HILBERT_N_BITS;
static_assert(HILBERT_N_BITS == 8);
static_assert(HILBERT_GRID_DIM <= HILBERT_MAX_GRID_DIM);

void __global__ k_arange(int N, unsigned int *arr);

// generate kv values from coordinates to be radix sorted
void __global__ k_coords_to_kv(
    const int N,
    const double *coords,
    const double *box,
    const unsigned int *bin_to_idx,
    unsigned int *keys,
    unsigned int *vals);

// variant of k_coords_to_kv allowing the selection of a subset of coordinates
void __global__ k_coords_to_kv_gather(
    const int N,                   // number of atoms in selection
    const unsigned int *atom_idxs, // [N] indices of atoms to select
    const double *coords,
    const double *box,
    const unsigned int *bin_to_idx,
    unsigned int *keys,
    unsigned int *vals);

template <typename RealType>
void __global__ k_check_rebuild_box(const int N, const double *new_box, const double *old_box, int *rebuild) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 9) {
        return;
    }

    // (ytz): box vectors have exactly 9 components
    // we can probably derive a looser bound later on.
    if (old_box[idx] != new_box[idx]) {
        rebuild[0] = 1;
    }
}

template <typename RealType>
void __global__ k_check_rebuild_coords_and_box(
    const int N,
    const double *__restrict__ new_coords,
    const double *__restrict__ old_coords,
    const double *__restrict__ new_box,
    const double *__restrict__ old_box,
    const double padding,
    int *rebuild) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 9) {
        // (ytz): box vectors have exactly 9 components
        // we can probably derive a looser bound later on.
        if (old_box[idx] != new_box[idx]) {
            rebuild[0] = 1;
        }
    }

    if (idx >= N) {
        return;
    }

    RealType xi = old_coords[idx * 3 + 0];
    RealType yi = old_coords[idx * 3 + 1];
    RealType zi = old_coords[idx * 3 + 2];

    RealType xj = new_coords[idx * 3 + 0];
    RealType yj = new_coords[idx * 3 + 1];
    RealType zj = new_coords[idx * 3 + 2];

    RealType dx = xi - xj;
    RealType dy = yi - yj;
    RealType dz = zi - zj;

    RealType d2ij = dx * dx + dy * dy + dz * dz;
    if (d2ij > static_cast<RealType>(0.25) * padding * padding) {
        // (ytz): this is *safe* but technically is a race condition
        rebuild[0] = 1;
    }
}

// TODO: DRY with k_check_rebuild_coords_and_box
template <typename RealType>
void __global__ k_check_rebuild_coords_and_box_gather(
    const int N,
    const unsigned int *atom_idxs,
    const double *__restrict__ new_coords,
    const double *__restrict__ old_coords,
    const double *__restrict__ new_box,
    const double *__restrict__ old_box,
    const double padding,
    int *rebuild) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 9) {
        // (ytz): box vectors have exactly 9 components
        // we can probably derive a looser bound later on.
        if (old_box[idx] != new_box[idx]) {
            rebuild[0] = 1;
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
        rebuild[0] = 1;
    }
}

template <typename RealType>
void __global__ k_copy_nblist_coords_and_box(
    const int N,
    const int *__restrict__ rebuild,
    const double *__restrict__ new_coords,
    const double *__restrict__ new_box,
    double *__restrict__ nblist_coords,
    double *__restrict__ nblist_box) {
    if (rebuild[0] <= 0) {
        return;
    }
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }
    if (idx < 9) {
        nblist_box[idx] = new_box[idx];
    }
#pragma unroll 3
    for (int i = 0; i < 3; i++) {
        nblist_coords[idx * 3 + i] = new_coords[idx * 3 + i];
    }
}

template <typename RealType>
void __global__ k_gather(
    const int N,
    const unsigned int *__restrict__ idxs,
    const RealType *__restrict__ array,
    RealType *__restrict__ gathered_array) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if (idx >= N) {
        return;
    }

    gathered_array[idx * stride + stride_idx] = array[idxs[idx] * stride + stride_idx];
}

template <typename RealType>
void __global__ k_gather_2x(
    const int N,
    const unsigned int *__restrict__ idxs,
    const RealType *__restrict__ array_1,
    const RealType *__restrict__ array_2,
    RealType *__restrict__ gathered_array_1,
    RealType *__restrict__ gathered_array_2) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if (idx >= N) {
        return;
    }

    gathered_array_1[idx * stride + stride_idx] = array_1[idxs[idx] * stride + stride_idx];
    gathered_array_2[idx * stride + stride_idx] = array_2[idxs[idx] * stride + stride_idx];
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

    array[unique_idxs[idx] * stride + stride_idx] += gathered_array[idx * stride + stride_idx];
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

// void __global__ k_compute_w_coords(
//     const int N,
//     const double lambda,
//     const double cutoff,
//     const int * __restrict__ lambda_plane_idxs, // 0 or 1, shift
//     const int * __restrict__ lambda_offset_idxs,
//     double * __restrict__ coords_w,
//     double * __restrict__ dw_dl) {

//     int atom_i_idx = blockIdx.x*blockDim.x + threadIdx.x;

//     if(atom_i_idx >= N) {
//         return;
//     }

//     int lambda_offset_i = atom_i_idx < N ? lambda_offset_idxs[atom_i_idx] : 0;
//     int lambda_plane_i = atom_i_idx < N ? lambda_plane_idxs[atom_i_idx] : 0;

//     double coords_w_i = (lambda_plane_i + lambda_offset_i*lambda)*cutoff;
//     double dw_dl_i = lambda_offset_i*cutoff;

//     coords_w[atom_i_idx] = coords_w_i;
//     dw_dl[atom_i_idx] = dw_dl_i;

// } // 0 or 1, how much we offset from the plane by )

// ALCHEMICAL == false guarantees that the tile's atoms are such that
// 1. src_param and dst_params are equal for every i in R and j in C
// 2. w_i and w_j are identical for every (i,j) in (RxC)
// DU_DL_DEPENDS_ON_DU_DP indicates whether or not to compute DU_DP when
// COMPUTE_DU_DL is requested (needed for interpolated potentials)
template <
    typename RealType,
    bool ALCHEMICAL,
    bool COMPUTE_U,
    bool COMPUTE_DU_DX,
    bool COMPUTE_DU_DL,
    bool COMPUTE_DU_DP>
// void __device__ __forceinline__ v_nonbonded_unified(
void __device__ v_nonbonded_unified(
    const int NC,
    const int NR,
    const double *__restrict__ coords,
    const double *__restrict__ params, // [N]
    const double *__restrict__ box,
    const double *__restrict__ dp_dl,
    const double *__restrict__ coords_w, // 4D coords
    const double *__restrict__ dw_dl,    // 4D derivatives
    // const int * __restrict__ lambda_plane_idxs, // 0 or 1, shift
    // const int * __restrict__ lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const double beta,
    const double cutoff,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ du_dl_buffer,
    unsigned long long *__restrict__ u_buffer) {

    int tile_idx = blockIdx.x;

    RealType box_x = box[0 * 3 + 0];
    RealType box_y = box[1 * 3 + 1];
    RealType box_z = box[2 * 3 + 2];

    RealType inv_box_x = 1 / box_x;
    RealType inv_box_y = 1 / box_y;
    RealType inv_box_z = 1 / box_z;

    int row_block_idx = ixn_tiles[tile_idx];

    int atom_i_idx = row_block_idx * 32 + threadIdx.x;
    // int lambda_offset_i = atom_i_idx < N ? lambda_offset_idxs[atom_i_idx] : 0;
    // int lambda_plane_i = atom_i_idx < N ? lambda_plane_idxs[atom_i_idx] : 0;

    const int N = NC + NR;

    if (NR != 0) {
        atom_i_idx += NC;
    }

    RealType ci_x = atom_i_idx < N ? coords[atom_i_idx * 3 + 0] : 0;
    RealType ci_y = atom_i_idx < N ? coords[atom_i_idx * 3 + 1] : 0;
    RealType ci_z = atom_i_idx < N ? coords[atom_i_idx * 3 + 2] : 0;
    RealType ci_w = atom_i_idx < N ? coords_w[atom_i_idx] : 0;

    RealType dq_dl_i = atom_i_idx < N ? dp_dl[atom_i_idx * 3 + 0] : 0;
    RealType dsig_dl_i = atom_i_idx < N ? dp_dl[atom_i_idx * 3 + 1] : 0;
    RealType deps_dl_i = atom_i_idx < N ? dp_dl[atom_i_idx * 3 + 2] : 0;
    RealType dw_dl_i = atom_i_idx < N ? dw_dl[atom_i_idx] : 0;

    unsigned long long gi_x = 0;
    unsigned long long gi_y = 0;
    unsigned long long gi_z = 0;
    unsigned long long du_dl = 0;

    int charge_param_idx_i = atom_i_idx * 3 + 0;
    int lj_param_idx_sig_i = atom_i_idx * 3 + 1;
    int lj_param_idx_eps_i = atom_i_idx * 3 + 2;

    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType sig_i = atom_i_idx < N ? params[lj_param_idx_sig_i] : 0;
    RealType eps_i = atom_i_idx < N ? params[lj_param_idx_eps_i] : 0;

    unsigned long long g_qi = 0;
    unsigned long long g_sigi = 0;
    unsigned long long g_epsi = 0;

    // i idx is contiguous but j is not, so we should swap them to avoid having to shuffle atom_j_idx
    int atom_j_idx = ixn_atoms[tile_idx * 32 + threadIdx.x];
    // int lambda_offset_j = atom_j_idx < N ? lambda_offset_idxs[atom_j_idx] : 0;
    // int lambda_plane_j = atom_j_idx < N ? lambda_plane_idxs[atom_j_idx] : 0;

    RealType cj_x = atom_j_idx < NC ? coords[atom_j_idx * 3 + 0] : 0;
    RealType cj_y = atom_j_idx < NC ? coords[atom_j_idx * 3 + 1] : 0;
    RealType cj_z = atom_j_idx < NC ? coords[atom_j_idx * 3 + 2] : 0;
    RealType cj_w = atom_j_idx < NC ? coords_w[atom_j_idx] : 0;

    RealType dq_dl_j = atom_j_idx < NC ? dp_dl[atom_j_idx * 3 + 0] : 0;
    RealType dsig_dl_j = atom_j_idx < NC ? dp_dl[atom_j_idx * 3 + 1] : 0;
    RealType deps_dl_j = atom_j_idx < NC ? dp_dl[atom_j_idx * 3 + 2] : 0;
    RealType dw_dl_j = atom_j_idx < NC ? dw_dl[atom_j_idx] : 0;

    unsigned long long gj_x = 0;
    unsigned long long gj_y = 0;
    unsigned long long gj_z = 0;

    int charge_param_idx_j = atom_j_idx * 3 + 0;
    int lj_param_idx_sig_j = atom_j_idx * 3 + 1;
    int lj_param_idx_eps_j = atom_j_idx * 3 + 2;

    RealType qj = atom_j_idx < NC ? params[charge_param_idx_j] : 0;
    RealType sig_j = atom_j_idx < NC ? params[lj_param_idx_sig_j] : 0;
    RealType eps_j = atom_j_idx < NC ? params[lj_param_idx_eps_j] : 0;

    unsigned long long g_qj = 0;
    unsigned long long g_sigj = 0;
    unsigned long long g_epsj = 0;

    RealType real_cutoff = static_cast<RealType>(cutoff);
    RealType cutoff_squared = real_cutoff * real_cutoff;

    unsigned long long energy = 0;

    RealType real_beta = static_cast<RealType>(beta);

    const int srcLane = (threadIdx.x + 1) % warp_size; // fixed
    // #pragma unroll
    for (int round = 0; round < 32; round++) {

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
            // delta_w = (lambda_plane_i - lambda_plane_j)*real_cutoff + (lambda_offset_i - lambda_offset_j)*real_lambda*real_cutoff;
            delta_w = ci_w - cj_w;
            d2ij += delta_w * delta_w;
        }

        const bool valid_ij =
            atom_i_idx < N &&
            ((NR == 0) ? atom_i_idx < atom_j_idx && atom_j_idx < N // all-pairs case, only compute the upper tri
                                                                   //   0  <= i < N, i < j < N
                       : atom_j_idx < NC);                         // ixn groups case, compute all pairwise ixns
                                                                   //   NC <= i < N, 0 <= j < NC

        // (ytz): note that d2ij must be *strictly* less than cutoff_squared. This is because we set the
        // non-interacting atoms to exactly real_cutoff*real_cutoff. This ensures that atoms who's 4th dimension
        // is set to cutoff are non-interacting.
        if (d2ij < cutoff_squared && valid_ij) {
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

            RealType real_du_dl = 0;

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

                if (COMPUTE_DU_DL && ALCHEMICAL) {
                    real_du_dl += sig_grad * (dsig_dl_i + dsig_dl_j);
                    RealType term = eps_grad * fix_nvidia_fmad(eps_j, deps_dl_i, eps_i, deps_dl_j);
                    real_du_dl += term;
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
            }

            if (COMPUTE_DU_DL && ALCHEMICAL) {
                // needed for cancellation of nans (if one term blows up)
                real_du_dl += delta_w * delta_prefactor * (dw_dl_i - dw_dl_j);
                real_du_dl += inv_dij * ebd * fix_nvidia_fmad(qj, dq_dl_i, qi, dq_dl_j);
                du_dl += FLOAT_TO_FIXED_NONBONDED(real_du_dl);
            }

            if (COMPUTE_U) {
                energy += FLOAT_TO_FIXED_NONBONDED(u);
            }
        }

        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane); // we can pre-compute this probably
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);

        cj_x = __shfl_sync(0xffffffff, cj_x, srcLane);
        cj_y = __shfl_sync(0xffffffff, cj_y, srcLane);
        cj_z = __shfl_sync(0xffffffff, cj_z, srcLane);

        if (ALCHEMICAL) {
            cj_w = __shfl_sync(0xffffffff, cj_w, srcLane); // this also can be optimized away
            dw_dl_j = __shfl_sync(0xffffffff, dw_dl_j, srcLane);
        }

        if (COMPUTE_DU_DX) {
            gj_x = __shfl_sync(0xffffffff, gj_x, srcLane);
            gj_y = __shfl_sync(0xffffffff, gj_y, srcLane);
            gj_z = __shfl_sync(0xffffffff, gj_z, srcLane);
        }

        if (COMPUTE_DU_DP) {
            g_qj = __shfl_sync(0xffffffff, g_qj, srcLane);
            g_sigj = __shfl_sync(0xffffffff, g_sigj, srcLane);
            g_epsj = __shfl_sync(0xffffffff, g_epsj, srcLane);
        }

        if (COMPUTE_DU_DL && ALCHEMICAL) {
            dsig_dl_j = __shfl_sync(0xffffffff, dsig_dl_j, srcLane);
            deps_dl_j = __shfl_sync(0xffffffff, deps_dl_j, srcLane);
            dq_dl_j = __shfl_sync(0xffffffff, dq_dl_j, srcLane);
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
        }

        if (atom_j_idx < N) {
            atomicAdd(du_dp + charge_param_idx_j, g_qj);
            atomicAdd(du_dp + lj_param_idx_sig_j, g_sigj);
            atomicAdd(du_dp + lj_param_idx_eps_j, g_epsj);
        }
    }

    // these are buffered and then reduced to avoid massive conflicts
    if (COMPUTE_DU_DL && ALCHEMICAL) {
        if (atom_i_idx < N) {
            atomicAdd(du_dl_buffer + atom_i_idx, du_dl);
        }
    }

    if (COMPUTE_U) {
        if (atom_i_idx < N) {
            atomicAdd(u_buffer + atom_i_idx, energy);
        }
    }
}

template <typename RealType, bool COMPUTE_U, bool COMPUTE_DU_DX, bool COMPUTE_DU_DL, bool COMPUTE_DU_DP>
void __global__ k_nonbonded_unified(
    const int NC,
    const int NR,
    const double *__restrict__ coords,
    const double *__restrict__ params, // [N]
    const double *__restrict__ box,
    const double *__restrict__ dp_dl,
    const double *__restrict__ coords_w, // 4D coords
    const double *__restrict__ dw_dl,    // 4D derivatives
    const double beta,
    const double cutoff,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ du_dl_buffer,
    unsigned long long *__restrict__ u_buffer) {

    int tile_idx = blockIdx.x;
    int row_block_idx = ixn_tiles[tile_idx];
    int atom_i_idx = row_block_idx * 32 + threadIdx.x;

    const int N = NC + NR;

    if (NR != 0) {
        atom_i_idx += NC;
    }

    RealType dq_dl_i = atom_i_idx < N ? dp_dl[atom_i_idx * 3 + 0] : 0;
    RealType dsig_dl_i = atom_i_idx < N ? dp_dl[atom_i_idx * 3 + 1] : 0;
    RealType deps_dl_i = atom_i_idx < N ? dp_dl[atom_i_idx * 3 + 2] : 0;
    RealType cw_i = atom_i_idx < N ? coords_w[atom_i_idx] : 0;

    int atom_j_idx = ixn_atoms[tile_idx * 32 + threadIdx.x];

    RealType dq_dl_j = atom_j_idx < N ? dp_dl[atom_j_idx * 3 + 0] : 0;
    RealType dsig_dl_j = atom_j_idx < N ? dp_dl[atom_j_idx * 3 + 1] : 0;
    RealType deps_dl_j = atom_j_idx < N ? dp_dl[atom_j_idx * 3 + 2] : 0;
    RealType cw_j = atom_j_idx < N ? coords_w[atom_j_idx] : 0;

    int is_vanilla =
        (cw_i == 0 && dq_dl_i == 0 && dsig_dl_i == 0 && deps_dl_i == 0 && cw_j == 0 && dq_dl_j == 0 && dsig_dl_j == 0 &&
         deps_dl_j == 0);

    bool tile_is_vanilla = __all_sync(0xffffffff, is_vanilla);

    if (tile_is_vanilla) {
        v_nonbonded_unified<RealType, 0, COMPUTE_U, COMPUTE_DU_DX, COMPUTE_DU_DL, COMPUTE_DU_DP>(
            NC,
            NR,
            coords,
            params,
            box,
            dp_dl,
            coords_w,
            dw_dl,
            beta,
            cutoff,
            ixn_tiles,
            ixn_atoms,
            du_dx,
            du_dp,
            du_dl_buffer,
            u_buffer);
    } else {
        v_nonbonded_unified<RealType, 1, COMPUTE_U, COMPUTE_DU_DX, COMPUTE_DU_DL, COMPUTE_DU_DP>(
            NC,
            NR,
            coords,
            params,
            box,
            dp_dl,
            coords_w,
            dw_dl,
            beta,
            cutoff,
            ixn_tiles,
            ixn_atoms,
            du_dx,
            du_dp,
            du_dl_buffer,
            u_buffer);
    };
}
