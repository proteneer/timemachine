#pragma once

// NOTE: The order of associative operations performed by this kernel
// should be identical to that of k_nonbonded. This is to ensure that
// we get exact cancellation when subtracting exclusions computed
// using this kernel.

#include "../fixed_point.hpp"
#include "nonbonded_common.cuh"

template <bool Negated>
void __device__ __forceinline__ accumulate(unsigned long long *__restrict acc, unsigned long long val) {
    atomicAdd(acc, Negated ? -val : val);
}

template <typename RealType, bool Negated>
void __global__ k_nonbonded_pair_list(
    const int M, // number of pairs
    const double *__restrict__ coords,
    const double *__restrict__ params,
    const double *__restrict__ box,
    const int *__restrict__ pair_idxs, // [M, 2] pair-list of atoms
    const double *__restrict__ scales, // [M]
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

    RealType ci_x = coords[atom_i_idx * 3 + 0];
    RealType ci_y = coords[atom_i_idx * 3 + 1];
    RealType ci_z = coords[atom_i_idx * 3 + 2];

    RealType ci_w = params[atom_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_W];

    unsigned long long gi_x = 0;
    unsigned long long gi_y = 0;
    unsigned long long gi_z = 0;

    int charge_param_idx_i = atom_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_CHARGE;
    int lj_param_idx_sig_i = atom_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_SIG;
    int lj_param_idx_eps_i = atom_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_EPS;

    RealType qi = params[charge_param_idx_i];
    RealType sig_i = params[lj_param_idx_sig_i];
    RealType eps_i = params[lj_param_idx_eps_i];

    unsigned long long g_qi = 0;
    unsigned long long g_sigi = 0;
    unsigned long long g_epsi = 0;

    int atom_j_idx = pair_idxs[pair_idx * 2 + 1];

    RealType cj_x = coords[atom_j_idx * 3 + 0];
    RealType cj_y = coords[atom_j_idx * 3 + 1];
    RealType cj_z = coords[atom_j_idx * 3 + 2];

    RealType cj_w = params[atom_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_W];

    unsigned long long gj_x = 0;
    unsigned long long gj_y = 0;
    unsigned long long gj_z = 0;

    int charge_param_idx_j = atom_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_CHARGE;
    int lj_param_idx_sig_j = atom_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_SIG;
    int lj_param_idx_eps_j = atom_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_EPS;

    RealType qj = params[charge_param_idx_j];
    RealType sig_j = params[lj_param_idx_sig_j];
    RealType eps_j = params[lj_param_idx_eps_j];

    unsigned long long g_qj = 0;
    unsigned long long g_sigj = 0;
    unsigned long long g_epsj = 0;

    RealType real_beta = static_cast<RealType>(beta);

    RealType real_cutoff = static_cast<RealType>(cutoff);
    RealType cutoff_squared = real_cutoff * real_cutoff;

    RealType charge_scale = scales[pair_idx * 2 + 0];
    RealType lj_scale = scales[pair_idx * 2 + 1];

    RealType box_x = box[0 * 3 + 0];
    RealType box_y = box[1 * 3 + 1];
    RealType box_z = box[2 * 3 + 2];

    RealType inv_box_x = 1 / box_x;
    RealType inv_box_y = 1 / box_y;
    RealType inv_box_z = 1 / box_z;

    RealType delta_x = ci_x - cj_x;
    RealType delta_y = ci_y - cj_y;
    RealType delta_z = ci_z - cj_z;

    delta_x -= box_x * nearbyint(delta_x * inv_box_x);
    delta_y -= box_y * nearbyint(delta_y * inv_box_y);
    delta_z -= box_z * nearbyint(delta_z * inv_box_z);

    RealType delta_w = ci_w - cj_w;
    RealType d2ij = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z + delta_w * delta_w;

    unsigned long long energy = 0;

    // see note: this must be strictly less than
    if (d2ij < cutoff_squared) {

        RealType u;
        RealType ebd;
        RealType es_prefactor;
        RealType dij;
        RealType inv_dij;
        RealType inv_d2ij;
        compute_electrostatics<RealType, true>(
            charge_scale, qi, qj, d2ij, beta, dij, inv_dij, inv_d2ij, ebd, es_prefactor, u);

        RealType delta_prefactor = es_prefactor;
        // lennard jones force
        if (eps_i != 0 && eps_j != 0) {
            RealType sig_grad;
            RealType eps_grad;
            compute_lj<RealType, true>(
                lj_scale, eps_i, eps_j, sig_i, sig_j, inv_dij, inv_d2ij, u, delta_prefactor, sig_grad, eps_grad);

            g_sigi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad);
            g_sigj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad);
            g_epsi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(eps_grad * eps_j);
            g_epsj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(eps_grad * eps_i);
        }

        gi_x += FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_x);
        gi_y += FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_y);
        gi_z += FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_z);

        gj_x += FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_x);
        gj_y += FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_y);
        gj_z += FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_z);

        // energy is size extensive so this may not be a good idea
        energy += FLOAT_TO_FIXED_NONBONDED(u);

        g_qi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(charge_scale * qj * inv_dij * ebd);
        g_qj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(charge_scale * qi * inv_dij * ebd);

        if (du_dx) {
            accumulate<Negated>(du_dx + atom_i_idx * 3 + 0, gi_x);
            accumulate<Negated>(du_dx + atom_i_idx * 3 + 1, gi_y);
            accumulate<Negated>(du_dx + atom_i_idx * 3 + 2, gi_z);

            accumulate<Negated>(du_dx + atom_j_idx * 3 + 0, gj_x);
            accumulate<Negated>(du_dx + atom_j_idx * 3 + 1, gj_y);
            accumulate<Negated>(du_dx + atom_j_idx * 3 + 2, gj_z);
        }

        if (du_dp) {
            accumulate<Negated>(du_dp + charge_param_idx_i, g_qi);
            accumulate<Negated>(du_dp + charge_param_idx_j, g_qj);

            accumulate<Negated>(du_dp + lj_param_idx_sig_i, g_sigi);
            accumulate<Negated>(du_dp + lj_param_idx_eps_i, g_epsi);

            accumulate<Negated>(du_dp + lj_param_idx_sig_j, g_sigj);
            accumulate<Negated>(du_dp + lj_param_idx_eps_j, g_epsj);
        }

        if (u_buffer) {
            accumulate<Negated>(u_buffer + atom_i_idx, energy);
        }
    }
}
