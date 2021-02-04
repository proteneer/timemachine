#pragma once

#include "surreal.cuh"
#include "../fixed_point.hpp"
#include "kernel_utils.cuh"
#define WARPSIZE 32

#define PI 3.141592653589793115997963468544185161
#define TWO_OVER_SQRT_PI 1.128379167095512595889238330988549829708

// we need to use a different level of precision for parameter derivatives
#define FIXED_EXPONENT_DU_DCHARGE 0x1000000000
#define FIXED_EXPONENT_DU_DSIG    0x2000000000
#define FIXED_EXPONENT_DU_DEPS    0x4000000000 // this is just getting silly

template<typename RealType, unsigned long long EXPONENT>
unsigned long long __device__ __forceinline__ FLOAT_TO_FIXED_DU_DP(RealType v) {
    return static_cast<unsigned long long>((long long)(v*EXPONENT));
}

template<typename RealType, unsigned long long EXPONENT>
RealType __device__ __forceinline__ FIXED_TO_FLOAT_DU_DP(unsigned long long v) {
    return static_cast<RealType>(static_cast<long long>(v))/EXPONENT;
}

// generate kv values from coordinates to be radix sorted 
void __global__ k_coords_to_kv(
    const int N,
    const double *coords,
    const double *box,
    const unsigned int *bin_to_idx,
    unsigned int *keys,
    unsigned int *vals) {

    const int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(atom_idx >= N) {
        return;
    }

    // these coords have to be centered
    double bx = box[0*3+0];
    double by = box[1*3+1];
    double bz = box[2*3+2];

    double binWidth = max(max(bx, by), bz)/255.0;

    double x = coords[atom_idx*3+0];
    double y = coords[atom_idx*3+1];
    double z = coords[atom_idx*3+2];

    x -= bx*floor(x/bx);
    y -= by*floor(y/by);
    z -= bz*floor(z/bz);

    unsigned int bin_x = x/binWidth;
    unsigned int bin_y = y/binWidth;
    unsigned int bin_z = z/binWidth;

    keys[atom_idx] = bin_to_idx[bin_x*256*256+bin_y*256+bin_z];
    // uncomment below if you want to perserve the atom ordering
    // keys[atom_idx] = atom_idx;
    vals[atom_idx] = atom_idx;

}

template <typename RealType>
void __global__ k_check_rebuild_box(
    const int N,
    const double *new_box,
    const double *old_box,
    int *rebuild) {

    const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= 9) {
        return;
    }

    // (ytz): box vectors have exactly 9 components
    // we can probably derive a looser bound later on.
    if(old_box[idx] != new_box[idx]) {
        rebuild[0] = 1;
    }

}

template <typename RealType>
void __global__ k_check_rebuild_coords(
    const int N,
    const double *new_coords,
    const double *old_coords,
    double padding,
    int *rebuild) {

    const int atom_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(atom_idx >= N) {
        return;
    }

    RealType xi = old_coords[atom_idx*3+0];
    RealType yi = old_coords[atom_idx*3+1];
    RealType zi = old_coords[atom_idx*3+2];

    RealType xj = new_coords[atom_idx*3+0];
    RealType yj = new_coords[atom_idx*3+1];
    RealType zj = new_coords[atom_idx*3+2];

    RealType dx = xi - xj;
    RealType dy = yi - yj;
    RealType dz = zi - zj;

    RealType d2ij = dx*dx + dy*dy + dz*dz;
    if(d2ij > static_cast<RealType>(0.25)*padding*padding) {
        // (ytz): this is *safe* but technically is a race condition
        rebuild[0] = 1;
    }


}


template <typename RealType>
void __global__ k_permute(
    const int N,
    const unsigned int * __restrict__ perm,
    const RealType * __restrict__ array,
    RealType * __restrict__ sorted_array) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if(idx >= N) {
        return;
    }

    sorted_array[idx*stride+stride_idx] = array[perm[idx]*stride+stride_idx];

}

template <typename RealType>
void __global__ k_inv_permute_accum(
    const int N,
    const unsigned int * __restrict__ perm,
    const RealType * __restrict__ sorted_array,
    RealType * __restrict__ array) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if(idx >= N) {
        return;
    }

    array[perm[idx]*stride+stride_idx] += sorted_array[idx*stride+stride_idx];

}

template <typename RealType>
void __global__ k_inv_permute_assign(
    const int N,
    const unsigned int * __restrict__ perm,
    const RealType * __restrict__ sorted_array,
    RealType * __restrict__ array) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if(idx >= N) {
        return;
    }

    array[perm[idx]*stride+stride_idx] = sorted_array[idx*stride+stride_idx];

}

template <typename RealType>
void __global__ k_add_ull_to_real(
    const int N,
    const unsigned long long * __restrict__ ull_array,
    RealType * __restrict__ real_array) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if(idx >= N) {
        return;
    }

    // handle charges, sigmas, epsilons with different exponents
    if(stride_idx == 0) {
        real_array[idx*stride+stride_idx] += FIXED_TO_FLOAT_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(ull_array[idx*stride+stride_idx]);
    } else if(stride_idx == 1) {
        real_array[idx*stride+stride_idx] += FIXED_TO_FLOAT_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(ull_array[idx*stride+stride_idx]);
    } else if(stride_idx == 2) {
        real_array[idx*stride+stride_idx] += FIXED_TO_FLOAT_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(ull_array[idx*stride+stride_idx]);
    }


}

template<typename RealType>
void __global__ k_reduce_buffer(
    int N,
    RealType *d_buffer,
    RealType *d_sum) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    RealType elem = idx < N ? d_buffer[idx] : 0;

    atomicAdd(d_sum, elem);

};

template<typename RealType>
void __global__ k_reduce_ull_buffer(
    int N,
    unsigned long long *d_buffer,
    RealType *d_sum) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    RealType elem = idx < N ? FIXED_TO_FLOAT<RealType>(d_buffer[idx]) : 0;

    atomicAdd(d_sum, elem);

};

void __global__ k_final_add(
    const unsigned long long *ull_array,
    double *double_array) {

    if(threadIdx.x == 0) {
        double_array[0] += FIXED_TO_FLOAT<double>(ull_array[0]);
    }

}

double __device__ __forceinline__ real_es_factor(double real_beta, double dij, double inv_d2ij) {
    double beta_dij = real_beta*dij;
    double exp_beta_dij_2 = exp(-beta_dij*beta_dij);
    double erfc_beta_dij = erfc(beta_dij);
    return -inv_d2ij*(static_cast<double>(TWO_OVER_SQRT_PI)*beta_dij*exp_beta_dij_2 + erfc_beta_dij);
}

float __device__ __forceinline__ real_es_factor(float real_beta, float dij, float inv_d2ij) {
    float beta_dij = real_beta*dij;
    // max ulp error is: 2 + floor(abs(1.16 * x))
    float exp_beta_dij_2 = __expf(-beta_dij*beta_dij);
    // 5th order gaussian polynomial approximation, we need the exp(-x^2) anyways for the chain rule
    // so we use last variant in https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    float t = 1.0f/(1.0f+0.3275911f*beta_dij);
    float erfc_beta_dij = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*exp_beta_dij_2;
    return -inv_d2ij*(static_cast<float>(TWO_OVER_SQRT_PI)*beta_dij*exp_beta_dij_2 + erfc_beta_dij);
}


template <typename RealType, bool COMPUTE_4D>
void __global__ k_nonbonded_du_dx(
    const int N,
    const double * __restrict__ coords,
    const double * __restrict__ params, // [N]
    const double * __restrict__ box,
    const double lambda,
    const int * __restrict__ lambda_plane_idxs, // 0 or 1, shift
    const int * __restrict__ lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const double beta,
    const double cutoff,
    const int * __restrict__ ixn_tiles,
    const unsigned int * __restrict__ ixn_atoms,
    unsigned long long * __restrict__ du_dx) {

    int tile_idx = blockIdx.x;

    RealType box_x = box[0*3+0];
    RealType box_y = box[1*3+1];
    RealType box_z = box[2*3+2];

    RealType inv_box_x = 1/box_x;
    RealType inv_box_y = 1/box_y;
    RealType inv_box_z = 1/box_z;

    int row_block_idx = ixn_tiles[tile_idx];

    int atom_i_idx = row_block_idx*32 + threadIdx.x;

    RealType ci_x = atom_i_idx < N ? coords[atom_i_idx*3+0] : 0;
    RealType ci_y = atom_i_idx < N ? coords[atom_i_idx*3+1] : 0;
    RealType ci_z = atom_i_idx < N ? coords[atom_i_idx*3+2] : 0;

    unsigned long long gi_x = 0;
    unsigned long long gi_y = 0;
    unsigned long long gi_z = 0;

    int charge_param_idx_i = atom_i_idx*3 + 0;
    int lj_param_idx_sig_i = atom_i_idx*3 + 1;
    int lj_param_idx_eps_i = atom_i_idx*3 + 2;

    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType sig_i = atom_i_idx < N ? params[lj_param_idx_sig_i] : 0;
    RealType eps_i = atom_i_idx < N ? params[lj_param_idx_eps_i] : 0;

    // i idx is contiguous but j is not, so we should swap them to avoid having to shuffle atom_j_idx
    int atom_j_idx = ixn_atoms[tile_idx*32 + threadIdx.x];

    RealType cj_x = atom_j_idx < N ? coords[atom_j_idx*3+0] : 0;
    RealType cj_y = atom_j_idx < N ? coords[atom_j_idx*3+1] : 0;
    RealType cj_z = atom_j_idx < N ? coords[atom_j_idx*3+2] : 0;
    unsigned long long gj_x = 0;
    unsigned long long gj_y = 0;
    unsigned long long gj_z = 0;

    int charge_param_idx_j = atom_j_idx*3 + 0;
    int lj_param_idx_sig_j = atom_j_idx*3 + 1;
    int lj_param_idx_eps_j = atom_j_idx*3 + 2;

    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    RealType sig_j = atom_j_idx < N ? params[lj_param_idx_sig_j] : 0;
    RealType eps_j = atom_j_idx < N ? params[lj_param_idx_eps_j] : 0;

    RealType cutoff_squared = cutoff*cutoff;

    RealType real_lambda = static_cast<RealType>(lambda);
    RealType real_beta = static_cast<RealType>(beta);

    // templatized parameter

    int lambda_offset_i = (COMPUTE_4D && atom_i_idx < N) ? lambda_offset_idxs[atom_i_idx] : 0;
    int lambda_plane_i = (COMPUTE_4D && atom_i_idx < N) ? lambda_plane_idxs[atom_i_idx] : 0;
    int lambda_offset_j = (COMPUTE_4D && atom_j_idx < N) ? lambda_offset_idxs[atom_j_idx] : 0;
    int lambda_plane_j = (COMPUTE_4D && atom_j_idx < N) ? lambda_plane_idxs[atom_j_idx] : 0;
    // see if this entire tile can be computed in 3D.
    // bool compute_4d = __any_sync(0xffffffff, lambda_plane_i != 0)  ||
                      // __any_sync(0xffffffff, lambda_plane_j != 0)  ||
                      // __any_sync(0xffffffff, lambda_offset_i != 0) ||
                      // __any_sync(0xffffffff, lambda_offset_j != 0);

    const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
    // #pragma unroll
    for(int round = 0; round < 32; round++) {

        RealType delta_x = ci_x - cj_x;
        RealType delta_y = ci_y - cj_y;
        RealType delta_z = ci_z - cj_z;

        delta_x -= box_x*nearbyint(delta_x*inv_box_x);
        delta_y -= box_y*nearbyint(delta_y*inv_box_y);
        delta_z -= box_z*nearbyint(delta_z*inv_box_z);

        // (ytz): it seems more expensive todo this extra computation on the inside loop but it helps preserves
        // precision. If we computed the 4D distance first, then delta_w = (Large_X + small_X) - (Large_Y + small_Y).
        // However, computing via (Large_X - Large_Y) + (small_X - small_Y) would result in a smaller error. In the worst case
        // d would be zero for Large_X = Large_Y , small_X != small_Y, Large_X >> small_X , Large_Y >> small_Y.
        // RealType delta_w = 0;
        RealType d2ij = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z;

        // compile time evaluates to either 0 or 1
        if(COMPUTE_4D) {
            RealType delta_w = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*real_lambda*cutoff;
            d2ij += delta_w*delta_w;
        }

        // (ytz): note that d2ij must be *strictly* less than cutoff_squared. This is because we set the
        // non-interacting atoms to exactly cutoff*cutoff. This ensures that atoms who's 4th dimension
        // is set to cutoff are non-interacting.
        if(d2ij < cutoff_squared  && atom_j_idx > atom_i_idx && atom_j_idx < N && atom_i_idx < N) {


            // electrostatics
            RealType inv_dij = rsqrt(d2ij);
            RealType dij = d2ij*inv_dij;

            RealType inv_d2ij = inv_dij*inv_dij;
            RealType qij = qi*qj;
            RealType es_prefactor = qij*inv_dij*real_es_factor(real_beta, dij, inv_d2ij);

            // lennard jones force
            RealType delta_prefactor = es_prefactor;
            if(eps_i != 0 && eps_j != 0) {
                RealType eps_ij = eps_i * eps_j;
                RealType sig_ij = sig_i + sig_j;
                RealType sig_inv_dij = sig_ij*inv_dij;
                RealType sig2_inv_d2ij = sig_inv_dij*sig_inv_dij;
                RealType sig6_inv_d6ij = sig2_inv_d2ij*sig2_inv_d2ij*sig2_inv_d2ij;
                RealType sig6_inv_d8ij = sig6_inv_d6ij*inv_d2ij;
                delta_prefactor -= eps_ij*sig6_inv_d8ij*(sig6_inv_d6ij*48 - 24);
            }

            gi_x += FLOAT_TO_FIXED(delta_prefactor*delta_x);
            gi_y += FLOAT_TO_FIXED(delta_prefactor*delta_y);
            gi_z += FLOAT_TO_FIXED(delta_prefactor*delta_z);

            // (ytz): the cast relies on undefined behavior since the exclusion terms will overflow.
            // Thus far, the current gen GPUs are deterministic in the resulting value. So long
            // as the UB stays deterministic then this should be fine. There's no guarantee though that
            // NVIDIA will keep the behavior consistent. Furthermore, do *not* attempt to swap signs via:
            // gj_* -= FLOAT_TO_FIXED(delta_prefactor*delta_*); // this is incorrect and has been shown
            // to introduce subtle bugs!
            gj_x += FLOAT_TO_FIXED(-delta_prefactor*delta_x);
            gj_y += FLOAT_TO_FIXED(-delta_prefactor*delta_y);
            gj_z += FLOAT_TO_FIXED(-delta_prefactor*delta_z);

        }

        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane); // we can pre-compute this probably, but shuffling may be faster
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
        cj_x = __shfl_sync(0xffffffff, cj_x, srcLane);
        cj_y = __shfl_sync(0xffffffff, cj_y, srcLane);
        cj_z = __shfl_sync(0xffffffff, cj_z, srcLane);
        gj_x = __shfl_sync(0xffffffff, gj_x, srcLane);
        gj_y = __shfl_sync(0xffffffff, gj_y, srcLane);
        gj_z = __shfl_sync(0xffffffff, gj_z, srcLane);

        if(COMPUTE_4D) {
            lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane); // this also can be optimized away
            lambda_plane_j = __shfl_sync(0xffffffff, lambda_plane_j, srcLane);
        }
    }

    // these reduction buffers are really tricky
    if(du_dx) {
        if(atom_i_idx < N) {
            atomicAdd(du_dx + atom_i_idx*3 + 0, gi_x);
            atomicAdd(du_dx + atom_i_idx*3 + 1, gi_y);
            atomicAdd(du_dx + atom_i_idx*3 + 2, gi_z);
        }
        if(atom_j_idx < N) {
            atomicAdd(du_dx + atom_j_idx*3 + 0, gj_x);
            atomicAdd(du_dx + atom_j_idx*3 + 1, gj_y);
            atomicAdd(du_dx + atom_j_idx*3 + 2, gj_z);
        }
    }

}


template <typename RealType>
void __global__ k_nonbonded(
    const int N,
    const double * __restrict__ coords,
    const double * __restrict__ params, // [N]
    const double * __restrict__ box,
    const double lambda,
    const int * __restrict__ lambda_plane_idxs, // 0 or 1, shift
    const int * __restrict__ lambda_offset_idxs, // 0 or 1, how much we offset from the plane by cutoff
    const double beta,
    const double cutoff,
    const int * __restrict__ ixn_tiles,
    const unsigned int * __restrict__ ixn_atoms,
    unsigned long long * __restrict__ du_dx,
    unsigned long long * __restrict__ du_dp,
    unsigned long long * __restrict__ du_dl_buffer,
    unsigned long long * __restrict__ u_buffer) {

    int tile_idx = blockIdx.x;

    RealType box_x = box[0*3+0];
    RealType box_y = box[1*3+1];
    RealType box_z = box[2*3+2];

    RealType inv_box_x = 1/box_x;
    RealType inv_box_y = 1/box_y;
    RealType inv_box_z = 1/box_z;

    int row_block_idx = ixn_tiles[tile_idx];

    int atom_i_idx = row_block_idx*32 + threadIdx.x;
    int lambda_offset_i = atom_i_idx < N ? lambda_offset_idxs[atom_i_idx] : 0;
    int lambda_plane_i = atom_i_idx < N ? lambda_plane_idxs[atom_i_idx] : 0;

    RealType ci_x = atom_i_idx < N ? coords[atom_i_idx*3+0] : 0;
    RealType ci_y = atom_i_idx < N ? coords[atom_i_idx*3+1] : 0;
    RealType ci_z = atom_i_idx < N ? coords[atom_i_idx*3+2] : 0;

    unsigned long long gi_x = 0;
    unsigned long long gi_y = 0;
    unsigned long long gi_z = 0;
    unsigned long long du_dl_i = 0;

    int charge_param_idx_i = atom_i_idx*3 + 0;
    int lj_param_idx_sig_i = atom_i_idx*3 + 1;
    int lj_param_idx_eps_i = atom_i_idx*3 + 2;

    RealType qi = atom_i_idx < N ? params[charge_param_idx_i] : 0;
    RealType sig_i = atom_i_idx < N ? params[lj_param_idx_sig_i] : 0;
    RealType eps_i = atom_i_idx < N ? params[lj_param_idx_eps_i] : 0;

    unsigned long long g_qi = 0;
    unsigned long long g_sigi = 0;
    unsigned long long g_epsi = 0;

    // i idx is contiguous but j is not, so we should swap them to avoid having to shuffle atom_j_idx
    int atom_j_idx = ixn_atoms[tile_idx*32 + threadIdx.x];
    int lambda_offset_j = atom_j_idx < N ? lambda_offset_idxs[atom_j_idx] : 0;
    int lambda_plane_j = atom_j_idx < N ? lambda_plane_idxs[atom_j_idx] : 0;

    RealType cj_x = atom_j_idx < N ? coords[atom_j_idx*3+0] : 0;
    RealType cj_y = atom_j_idx < N ? coords[atom_j_idx*3+1] : 0;
    RealType cj_z = atom_j_idx < N ? coords[atom_j_idx*3+2] : 0;
    unsigned long long gj_x = 0;
    unsigned long long gj_y = 0;
    unsigned long long gj_z = 0;
    unsigned long long du_dl_j = 0;

    int charge_param_idx_j = atom_j_idx*3 + 0;
    int lj_param_idx_sig_j = atom_j_idx*3 + 1;
    int lj_param_idx_eps_j = atom_j_idx*3 + 2;

    RealType qj = atom_j_idx < N ? params[charge_param_idx_j] : 0;
    RealType sig_j = atom_j_idx < N ? params[lj_param_idx_sig_j] : 0;
    RealType eps_j = atom_j_idx < N ? params[lj_param_idx_eps_j] : 0;

    unsigned long long g_qj = 0;
    unsigned long long g_sigj = 0;
    unsigned long long g_epsj = 0;

    RealType cutoff_squared = cutoff*cutoff;

    unsigned long long energy = 0;

    RealType real_lambda = static_cast<RealType>(lambda);
    RealType real_beta = static_cast<RealType>(beta);

    const int srcLane = (threadIdx.x + 1) % WARPSIZE; // fixed
    // #pragma unroll
    for(int round = 0; round < 32; round++) {

        RealType delta_x = ci_x - cj_x;
        RealType delta_y = ci_y - cj_y;
        RealType delta_z = ci_z - cj_z;

        delta_x -= box_x*nearbyint(delta_x*inv_box_x);
        delta_y -= box_y*nearbyint(delta_y*inv_box_y);
        delta_z -= box_z*nearbyint(delta_z*inv_box_z);

        RealType delta_w = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*real_lambda*cutoff;

        RealType d2ij = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z + delta_w*delta_w;

        // (ytz): note that d2ij must be *strictly* less than cutoff_squared. This is because we set the
        // non-interacting atoms to exactly cutoff*cutoff. This ensures that atoms who's 4th dimension
        // is set to cutoff are non-interacting.
        if(d2ij < cutoff_squared  && atom_j_idx > atom_i_idx && atom_j_idx < N && atom_i_idx < N) {

            // electrostatics
            RealType inv_dij = rsqrt(d2ij);
            RealType dij = d2ij*inv_dij;

            RealType inv_d2ij = inv_dij*inv_dij;
            RealType beta_dij = real_beta*dij;
            RealType ebd = erfc(beta_dij);

            RealType qij = qi*qj;

            RealType u = qij*inv_dij*ebd;
            RealType es_prefactor = qij*inv_dij*real_es_factor(real_beta, dij, inv_d2ij);

            // lennard jones
            RealType inv_d4ij = inv_d2ij*inv_d2ij;
            RealType inv_d6ij = inv_d4ij*inv_d2ij;

            // lennard jones force
            RealType eps_ij = eps_i * eps_j;

            RealType delta_prefactor = es_prefactor;

            if(eps_i != 0 && eps_j != 0) {
                RealType sig_ij = sig_i + sig_j;

                RealType sig5 = sig_ij*sig_ij*sig_ij*sig_ij*sig_ij;

                RealType sig_inv_dij = sig_ij*inv_dij;
                RealType sig2_inv_d2ij = sig_inv_dij*sig_inv_dij;
                RealType sig6_inv_d6ij = sig2_inv_d2ij*sig2_inv_d2ij*sig2_inv_d2ij;
                RealType sig6_inv_d8ij = sig6_inv_d6ij*inv_d2ij;

                RealType lj_prefactor = eps_ij*sig6_inv_d8ij*(sig6_inv_d6ij*48 - 24);
                u += 4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;
                delta_prefactor -= lj_prefactor;

                RealType sig_grad = 24*eps_ij*sig5*inv_d6ij*(2*sig6_inv_d6ij-1);
                g_sigi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad);
                g_sigj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad);
                RealType eps_grad = 4*(sig6_inv_d6ij-1)*sig6_inv_d6ij;
                g_epsi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(eps_grad*eps_j);
                g_epsj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(eps_grad*eps_i);
            }
         
            gi_x += FLOAT_TO_FIXED(delta_prefactor*delta_x);
            gi_y += FLOAT_TO_FIXED(delta_prefactor*delta_y);
            gi_z += FLOAT_TO_FIXED(delta_prefactor*delta_z);

            gj_x += FLOAT_TO_FIXED(-delta_prefactor*delta_x);
            gj_y += FLOAT_TO_FIXED(-delta_prefactor*delta_y);
            gj_z += FLOAT_TO_FIXED(-delta_prefactor*delta_z);

            du_dl_i += FLOAT_TO_FIXED(delta_prefactor*delta_w*lambda_offset_i*cutoff);
            du_dl_j += FLOAT_TO_FIXED(-delta_prefactor*delta_w*lambda_offset_j*cutoff);

            energy += FLOAT_TO_FIXED(u);
            g_qi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(qj*inv_dij*ebd);
            g_qj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(qi*inv_dij*ebd);


        }

        atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, srcLane); // we can pre-compute this probably
        qj = __shfl_sync(0xffffffff, qj, srcLane);
        eps_j = __shfl_sync(0xffffffff, eps_j, srcLane);
        sig_j = __shfl_sync(0xffffffff, sig_j, srcLane);
        cj_x = __shfl_sync(0xffffffff, cj_x, srcLane); // needs to support real
        cj_y = __shfl_sync(0xffffffff, cj_y, srcLane); // needs to support real
        cj_z = __shfl_sync(0xffffffff, cj_z, srcLane); // needs to support real
        gj_x = __shfl_sync(0xffffffff, gj_x, srcLane);
        gj_y = __shfl_sync(0xffffffff, gj_y, srcLane);
        gj_z = __shfl_sync(0xffffffff, gj_z, srcLane);
        g_qj = __shfl_sync(0xffffffff, g_qj, srcLane);
        g_sigj = __shfl_sync(0xffffffff, g_sigj, srcLane);
        g_epsj = __shfl_sync(0xffffffff, g_epsj, srcLane);
        lambda_offset_j = __shfl_sync(0xffffffff, lambda_offset_j, srcLane); // this also can be optimized away
        lambda_plane_j = __shfl_sync(0xffffffff, lambda_plane_j, srcLane);
        du_dl_j = __shfl_sync(0xffffffff, du_dl_j, srcLane);
    }

    // these reduction buffers are really tricky
    if(du_dx) {
        if(atom_i_idx < N) {
            atomicAdd(du_dx + atom_i_idx*3 + 0, gi_x);
            atomicAdd(du_dx + atom_i_idx*3 + 1, gi_y);
            atomicAdd(du_dx + atom_i_idx*3 + 2, gi_z);
        }
        if(atom_j_idx < N) {
            atomicAdd(du_dx + atom_j_idx*3 + 0, gj_x);
            atomicAdd(du_dx + atom_j_idx*3 + 1, gj_y);
            atomicAdd(du_dx + atom_j_idx*3 + 2, gj_z);
        }
    }

    if(du_dp) {

        if(atom_i_idx < N) {
            atomicAdd(du_dp + charge_param_idx_i, g_qi);
            atomicAdd(du_dp + lj_param_idx_sig_i, g_sigi);
            atomicAdd(du_dp + lj_param_idx_eps_i, g_epsi);
        }

        if(atom_j_idx < N) {
            atomicAdd(du_dp + charge_param_idx_j, g_qj);
            atomicAdd(du_dp + lj_param_idx_sig_j, g_sigj);
            atomicAdd(du_dp + lj_param_idx_eps_j, g_epsj);
        }

    }

    // these are buffered and then reduced to avoid massive conflicts
    if(du_dl_buffer) {
        if(atom_i_idx < N) {
            atomicAdd(du_dl_buffer + atom_i_idx, du_dl_i + du_dl_j);
        }
    }

    if(u_buffer) {
        if(atom_i_idx < N) {
            atomicAdd(u_buffer + atom_i_idx, energy);
        }
    }

}


template<typename RealType>
void __global__ k_nonbonded_exclusions(
    const int E, // number of exclusions
    const double *coords,
    const double *params,
    const double *box,
    const double lambda,
    const int *lambda_plane_idxs, // 0 or 1, shift
    const int *lambda_offset_idxs, // 0 or 1, if we alolw this atom to be decoupled
    const int *exclusion_idxs, // [E, 2] pair-list of atoms to be excluded
    const double *scales, // [E]
    const double beta,
    const double cutoff,
    unsigned long long *du_dx,
    unsigned long long *du_dp,
    unsigned long long *du_dl_buffer,
    unsigned long long *u_buffer) {

    // (ytz): oddly enough the order of atom_i and atom_j
    // seem to not matter. I think this is because distance calculations
    // are bitwise identical in both dij(i, j) and dij(j, i) . However we
    // do need the calculation done for exclusions to perfectly mirror
    // that of the nonbonded kernel itself. Remember that floating points
    // commute but are not associative.

    const int e_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(e_idx >= E) {
        return;
    }

    int atom_i_idx = exclusion_idxs[e_idx*2 + 0];
    int lambda_offset_i = lambda_offset_idxs[atom_i_idx];
    int lambda_plane_i = lambda_plane_idxs[atom_i_idx];

    RealType ci_x = coords[atom_i_idx*3+0];
    RealType ci_y = coords[atom_i_idx*3+1];
    RealType ci_z = coords[atom_i_idx*3+2];
    unsigned long long gi_x = 0;
    unsigned long long gi_y = 0;
    unsigned long long gi_z = 0;
    unsigned long long du_dl_i = 0;

    int charge_param_idx_i = atom_i_idx*3 + 0;
    int lj_param_idx_sig_i = atom_i_idx*3 + 1;
    int lj_param_idx_eps_i = atom_i_idx*3 + 2;

    RealType qi = params[charge_param_idx_i];
    RealType sig_i = params[lj_param_idx_sig_i];
    RealType eps_i = params[lj_param_idx_eps_i];

    unsigned long long g_qi = 0;
    unsigned long long g_sigi = 0;
    unsigned long long g_epsi = 0;

    int atom_j_idx = exclusion_idxs[e_idx*2 + 1];
    int lambda_offset_j = lambda_offset_idxs[atom_j_idx];
    int lambda_plane_j = lambda_plane_idxs[atom_j_idx];

    RealType cj_x = coords[atom_j_idx*3+0];
    RealType cj_y = coords[atom_j_idx*3+1];
    RealType cj_z = coords[atom_j_idx*3+2];
    unsigned long long gj_x = 0;
    unsigned long long gj_y = 0;
    unsigned long long gj_z = 0;
    unsigned long long du_dl_j = 0;

    int charge_param_idx_j = atom_j_idx*3+0;
    int lj_param_idx_sig_j = atom_j_idx*3 + 1;
    int lj_param_idx_eps_j = atom_j_idx*3 + 2;

    RealType qj = params[charge_param_idx_j];
    RealType sig_j = params[lj_param_idx_sig_j];
    RealType eps_j = params[lj_param_idx_eps_j];

    unsigned long long g_qj = 0;
    unsigned long long g_sigj = 0;
    unsigned long long g_epsj = 0;

    RealType real_lambda = static_cast<RealType>(lambda);
    RealType real_beta = static_cast<RealType>(beta);
    RealType cutoff_squared = cutoff*cutoff;

    RealType charge_scale = scales[e_idx*2 + 0];
    RealType lj_scale = scales[e_idx*2 + 1];

    RealType box_x = box[0*3+0];
    RealType box_y = box[1*3+1];
    RealType box_z = box[2*3+2];

    RealType inv_box_x = 1/box_x;
    RealType inv_box_y = 1/box_y;
    RealType inv_box_z = 1/box_z;

    RealType delta_x = ci_x - cj_x;
    RealType delta_y = ci_y - cj_y;
    RealType delta_z = ci_z - cj_z;

    delta_x -= box_x*nearbyint(delta_x*inv_box_x);
    delta_y -= box_y*nearbyint(delta_y*inv_box_y);
    delta_z -= box_z*nearbyint(delta_z*inv_box_z);

    RealType delta_w = (lambda_plane_i - lambda_plane_j)*cutoff + (lambda_offset_i - lambda_offset_j)*real_lambda*cutoff;

    RealType d2ij = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z + delta_w*delta_w;

    unsigned long long energy = 0;

    // see note: this must be strictly less than
    if(d2ij < cutoff_squared) {

        RealType inv_dij = rsqrt(d2ij);

        RealType dij = d2ij*inv_dij;
        RealType inv_d2ij = inv_dij*inv_dij;

        RealType beta_dij = real_beta*dij;
        RealType ebd = erfc(beta_dij);

        RealType qij = qi*qj;
        RealType es_prefactor = charge_scale*qij*inv_dij*real_es_factor(real_beta, dij, inv_d2ij);

        // lennard jones
        RealType inv_d4ij = inv_d2ij*inv_d2ij;
        RealType inv_d6ij = inv_d4ij*inv_d2ij;

        RealType delta_prefactor = es_prefactor;

        RealType u = charge_scale*qij*inv_dij*ebd;
        // lennard jones force
        if(eps_i != 0 && eps_j != 0) {
            RealType eps_ij = eps_i * eps_j;
            RealType sig_ij = sig_i + sig_j;
            RealType sig5 = sig_ij*sig_ij*sig_ij*sig_ij*sig_ij;

            RealType sig_inv_dij = sig_ij*inv_dij;
            RealType sig2_inv_d2ij = sig_inv_dij*sig_inv_dij;
            RealType sig6_inv_d6ij = sig2_inv_d2ij*sig2_inv_d2ij*sig2_inv_d2ij;
            RealType sig6_inv_d8ij = sig6_inv_d6ij*inv_d2ij;

            RealType lj_prefactor = lj_scale*eps_ij*sig6_inv_d8ij*(sig6_inv_d6ij*48 - 24);
            delta_prefactor -= lj_prefactor;
            u += lj_scale*4*eps_ij*(sig6_inv_d6ij-1)*sig6_inv_d6ij;

            RealType eps_grad = lj_scale*4*(sig6_inv_d6ij-1)*sig6_inv_d6ij;
            RealType sig_grad = lj_scale*24*eps_ij*sig5*inv_d6ij*(2*sig6_inv_d6ij-1);
            g_sigi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(-sig_grad);
            g_sigj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(-sig_grad);

            g_epsi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(-eps_grad*eps_j);
            g_epsj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(-eps_grad*eps_i);

        }

        gi_x += FLOAT_TO_FIXED(-delta_prefactor*delta_x);
        gi_y += FLOAT_TO_FIXED(-delta_prefactor*delta_y);
        gi_z += FLOAT_TO_FIXED(-delta_prefactor*delta_z);

        gj_x += FLOAT_TO_FIXED(delta_prefactor*delta_x);
        gj_y += FLOAT_TO_FIXED(delta_prefactor*delta_y);
        gj_z += FLOAT_TO_FIXED(delta_prefactor*delta_z);

        du_dl_i += FLOAT_TO_FIXED(-delta_prefactor*delta_w*lambda_offset_i*cutoff);
        du_dl_j += FLOAT_TO_FIXED(delta_prefactor*delta_w*lambda_offset_j*cutoff);

        // energy is size extensive so this may not be a good idea
        energy += FLOAT_TO_FIXED(-u);

        g_qi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(-charge_scale*qj*inv_dij*ebd);
        g_qj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(-charge_scale*qi*inv_dij*ebd);

        if(du_dx) {

            // these are unsorted
            auto old_i_x = atomicAdd(du_dx + atom_i_idx*3 + 0, gi_x);
            auto old_i_y = atomicAdd(du_dx + atom_i_idx*3 + 1, gi_y);
            auto old_i_z = atomicAdd(du_dx + atom_i_idx*3 + 2, gi_z);

            auto old_j_x = atomicAdd(du_dx + atom_j_idx*3 + 0, gj_x);
            auto old_j_y = atomicAdd(du_dx + atom_j_idx*3 + 1, gj_y);
            auto old_j_z = atomicAdd(du_dx + atom_j_idx*3 + 2, gj_z);

        }

        if(du_dp) {
            atomicAdd(du_dp + charge_param_idx_i, g_qi);
            atomicAdd(du_dp + charge_param_idx_j, g_qj);

            atomicAdd(du_dp + lj_param_idx_sig_i, g_sigi);
            atomicAdd(du_dp + lj_param_idx_eps_i, g_epsi);

            atomicAdd(du_dp + lj_param_idx_sig_j, g_sigj);
            atomicAdd(du_dp + lj_param_idx_eps_j, g_epsj);
        }

        if(du_dl_buffer) {
            atomicAdd(du_dl_buffer + atom_i_idx, du_dl_i + du_dl_j);
        }

        if(u_buffer) {
            atomicAdd(u_buffer + atom_i_idx, energy);
        }

    }

}
