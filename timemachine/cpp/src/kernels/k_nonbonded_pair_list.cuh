#pragma once

// NOTE: The order of associative operations performed by this kernel
// should be identical to that of k_nonbonded. This is to ensure that
// we get exact cancellation when subtracting exclusions computed
// using this kernel.

#include "../fixed_point.hpp"
#include "k_nonbonded_common.cuh"

template <bool Negated>
void __device__ __forceinline__ accumulate(unsigned long long *__restrict acc, unsigned long long val) {
    atomicAdd(acc, Negated ? -val : val);
}

template <typename RealType, bool Negated, bool COMPUTE_U>
void __global__ k_nonbonded_pair_list(
    const int M, // number of pairs
    const double *__restrict__ coords,
    const double *__restrict__ params,
    const double *__restrict__ box,
    const int *__restrict__ pair_idxs, // [M, 2] pair-list of atoms
    const double *__restrict__ scales, // [M, 2]
    const double beta,
    const double cutoff,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ u_buffer) {

    // (ytz): oddly enough the order of atom_i and atom_j
    // seem to not matter. I think this is because distance calculations
    // are bitwise identical in both dij(i, j) and dij(j, i) . However we
    // do need the calculation done for exclusions to perfectly mirror
    // that of the nonbonded kernel itself. Remember that floating points
    // commute but are not associative.

    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= M) {
        return;
    }

    int atom_i_idx = pair_idxs[pair_idx * 2 + 0];
    int atom_j_idx = pair_idxs[pair_idx * 2 + 1];

    RealType ci_x = coords[atom_i_idx * 3 + 0];
    RealType ci_y = coords[atom_i_idx * 3 + 1];
    RealType ci_z = coords[atom_i_idx * 3 + 2];

    RealType cj_x = coords[atom_j_idx * 3 + 0];
    RealType cj_y = coords[atom_j_idx * 3 + 1];
    RealType cj_z = coords[atom_j_idx * 3 + 2];

    int params_i_idx = atom_i_idx * PARAMS_PER_ATOM;
    int charge_param_idx_i = params_i_idx + PARAM_OFFSET_CHARGE;
    int lj_param_idx_sig_i = params_i_idx + PARAM_OFFSET_SIG;
    int lj_param_idx_eps_i = params_i_idx + PARAM_OFFSET_EPS;
    int w_param_idx_i = params_i_idx + PARAM_OFFSET_W;

    RealType qi = params[charge_param_idx_i];
    RealType sig_i = params[lj_param_idx_sig_i];
    RealType eps_i = params[lj_param_idx_eps_i];
    RealType w_i = params[w_param_idx_i];

    int params_j_idx = atom_j_idx * PARAMS_PER_ATOM;
    int charge_param_idx_j = params_j_idx + PARAM_OFFSET_CHARGE;
    int lj_param_idx_sig_j = params_j_idx + PARAM_OFFSET_SIG;
    int lj_param_idx_eps_j = params_j_idx + PARAM_OFFSET_EPS;
    int w_param_idx_j = params_j_idx + PARAM_OFFSET_W;

    RealType qj = params[charge_param_idx_j];
    RealType sig_j = params[lj_param_idx_sig_j];
    RealType eps_j = params[lj_param_idx_eps_j];
    RealType w_j = params[w_param_idx_j];

    RealType real_beta = static_cast<RealType>(beta);

    RealType real_cutoff = static_cast<RealType>(cutoff);
    RealType cutoff_squared = real_cutoff * real_cutoff;

    RealType box_x = box[0 * 3 + 0];
    RealType box_y = box[1 * 3 + 1];
    RealType box_z = box[2 * 3 + 2];

    RealType box_inv_x = 1 / box_x;
    RealType box_inv_y = 1 / box_y;
    RealType box_inv_z = 1 / box_z;

    RealType delta_x = ci_x - cj_x;
    RealType delta_y = ci_y - cj_y;
    RealType delta_z = ci_z - cj_z;

    delta_x -= box_x * nearbyint(delta_x * box_inv_x);
    delta_y -= box_y * nearbyint(delta_y * box_inv_y);
    delta_z -= box_z * nearbyint(delta_z * box_inv_z);

    RealType delta_w = w_i - w_j;
    RealType d2ij = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z + delta_w * delta_w;

    // Note: this must be strictly less than to evaluate
    if (d2ij >= cutoff_squared) {
        return;
    }

    RealType charge_scale = scales[pair_idx * 2 + 0];
    RealType lj_scale = scales[pair_idx * 2 + 1];

    RealType u;
    RealType ebd;
    RealType es_prefactor;
    RealType dij;
    RealType inv_dij;
    RealType inv_d2ij;
    compute_electrostatics<RealType, COMPUTE_U>(
        charge_scale, qi, qj, d2ij, real_beta, dij, inv_dij, inv_d2ij, ebd, es_prefactor, u);

    RealType delta_prefactor = es_prefactor;
    // lennard jones force
    RealType sig_grad;
    RealType eps_grad;
    if (eps_i != 0 && eps_j != 0) {
        compute_lj<RealType, COMPUTE_U>(
            lj_scale, eps_i, eps_j, sig_i, sig_j, inv_dij, inv_d2ij, u, delta_prefactor, sig_grad, eps_grad);
    }

    if (du_dx) {
        accumulate<Negated>(du_dx + atom_i_idx * 3 + 0, FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_x));
        accumulate<Negated>(du_dx + atom_i_idx * 3 + 1, FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_y));
        accumulate<Negated>(du_dx + atom_i_idx * 3 + 2, FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_z));

        accumulate<Negated>(du_dx + atom_j_idx * 3 + 0, FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_x));
        accumulate<Negated>(du_dx + atom_j_idx * 3 + 1, FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_y));
        accumulate<Negated>(du_dx + atom_j_idx * 3 + 2, FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_z));
    }

    if (du_dp) {
        accumulate<Negated>(
            du_dp + charge_param_idx_i,
            FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(charge_scale * qj * inv_dij * ebd));
        accumulate<Negated>(
            du_dp + lj_param_idx_sig_i, FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad));
        accumulate<Negated>(
            du_dp + lj_param_idx_eps_i, FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(eps_grad * eps_j));
        accumulate<Negated>(
            du_dp + w_param_idx_i, FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(delta_prefactor * delta_w));

        accumulate<Negated>(
            du_dp + charge_param_idx_j,
            FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(charge_scale * qi * inv_dij * ebd));
        accumulate<Negated>(
            du_dp + lj_param_idx_sig_j, FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad));
        accumulate<Negated>(
            du_dp + lj_param_idx_eps_j, FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(eps_grad * eps_i));
        accumulate<Negated>(
            du_dp + w_param_idx_j, FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(-delta_prefactor * delta_w));
    }

    if (u_buffer) {
        // energy is size extensive so this may not be a good idea
        accumulate<Negated>(u_buffer + atom_i_idx, FLOAT_TO_FIXED_NONBONDED(u));
    }
}
