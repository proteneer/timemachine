#pragma once

// NOTE: The order of associative operations performed by this kernel
// should be identical to that of k_nonbonded. This is to ensure that
// we get exact cancellation when subtracting exclusions computed
// using this kernel.

#include "../fixed_point.hpp"
#include "nonbonded_common.cuh"

template <typename RealType, bool Negated>
void __global__ k_nonbonded_pairs(
    const int M, // number of pairs
    const double *__restrict__ coords,
    const double *__restrict__ params,
    const double *__restrict__ box,
    const double *__restrict__ dp_dl,
    const double *__restrict__ coords_w, // 4D coords
    const double *__restrict__ dw_dl,    // 4D derivatives
    const int *__restrict__ pair_idxs,   // [M, 2] pair-list of atoms
    const double *__restrict__ scales,   // [M]
    const double beta,
    const double cutoff,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ du_dl_buffer,
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

    const int sign = Negated ? -1 : 1;

    int atom_i_idx = pair_idxs[pair_idx * 2 + 0];

    RealType ci_x = coords[atom_i_idx * 3 + 0];
    RealType ci_y = coords[atom_i_idx * 3 + 1];
    RealType ci_z = coords[atom_i_idx * 3 + 2];
    RealType ci_w = coords_w[atom_i_idx];

    RealType dq_dl_i = dp_dl[atom_i_idx * 3 + 0];
    RealType dsig_dl_i = dp_dl[atom_i_idx * 3 + 1];
    RealType deps_dl_i = dp_dl[atom_i_idx * 3 + 2];
    RealType dw_dl_i = dw_dl[atom_i_idx];

    unsigned long long gi_x = 0;
    unsigned long long gi_y = 0;
    unsigned long long gi_z = 0;

    int charge_param_idx_i = atom_i_idx * 3 + 0;
    int lj_param_idx_sig_i = atom_i_idx * 3 + 1;
    int lj_param_idx_eps_i = atom_i_idx * 3 + 2;

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
    RealType cj_w = coords_w[atom_j_idx];

    RealType dq_dl_j = dp_dl[atom_j_idx * 3 + 0];
    RealType dsig_dl_j = dp_dl[atom_j_idx * 3 + 1];
    RealType deps_dl_j = dp_dl[atom_j_idx * 3 + 2];
    RealType dw_dl_j = dw_dl[atom_j_idx];

    unsigned long long gj_x = 0;
    unsigned long long gj_y = 0;
    unsigned long long gj_z = 0;

    int charge_param_idx_j = atom_j_idx * 3 + 0;
    int lj_param_idx_sig_j = atom_j_idx * 3 + 1;
    int lj_param_idx_eps_j = atom_j_idx * 3 + 2;

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

    // `is_vanilla = true` means no parameter interpolation and 4D
    // coordinates are all zero. In this case we can use a more
    // efficient kernel.
    int is_vanilla =
        (ci_w == 0 && dq_dl_i == 0 && dsig_dl_i == 0 && deps_dl_i == 0 && cj_w == 0 && dq_dl_j == 0 && dsig_dl_j == 0 &&
         deps_dl_j == 0);

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
        RealType real_du_dl = 0;
        // lennard jones force
        if (eps_i != 0 && eps_j != 0) {
            RealType sig_grad;
            RealType eps_grad;
            compute_lj<RealType, true>(
                lj_scale, eps_i, eps_j, sig_i, sig_j, inv_dij, inv_d2ij, u, delta_prefactor, sig_grad, eps_grad);

            g_sigi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(-sig_grad);
            g_sigj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(-sig_grad);
            g_epsi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(-eps_grad * eps_j);
            g_epsj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(-eps_grad * eps_i);

            real_du_dl += sign * sig_grad * (dsig_dl_i + dsig_dl_j);
            RealType term = eps_grad * fix_nvidia_fmad(eps_j, deps_dl_i, eps_i, deps_dl_j);
            real_du_dl += sign * term;
        }

        gi_x += sign * FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_x);
        gi_y += sign * FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_y);
        gi_z += sign * FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_z);

        gj_x += sign * FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_x);
        gj_y += sign * FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_y);
        gj_z += sign * FLOAT_TO_FIXED_NONBONDED(-delta_prefactor * delta_z);

        // energy is size extensive so this may not be a good idea
        energy += sign * FLOAT_TO_FIXED_NONBONDED(u);

        g_qi += sign * FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(charge_scale * qj * inv_dij * ebd);
        g_qj += sign * FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(charge_scale * qi * inv_dij * ebd);

        real_du_dl += sign * delta_w * delta_prefactor * (dw_dl_i - dw_dl_j);
        real_du_dl += sign * charge_scale * inv_dij * ebd * fix_nvidia_fmad(qj, dq_dl_i, qi, dq_dl_j);

        if (du_dx) {
            atomicAdd(du_dx + atom_i_idx * 3 + 0, gi_x);
            atomicAdd(du_dx + atom_i_idx * 3 + 1, gi_y);
            atomicAdd(du_dx + atom_i_idx * 3 + 2, gi_z);

            atomicAdd(du_dx + atom_j_idx * 3 + 0, gj_x);
            atomicAdd(du_dx + atom_j_idx * 3 + 1, gj_y);
            atomicAdd(du_dx + atom_j_idx * 3 + 2, gj_z);
        }

        if (du_dp) {
            atomicAdd(du_dp + charge_param_idx_i, g_qi);
            atomicAdd(du_dp + charge_param_idx_j, g_qj);

            atomicAdd(du_dp + lj_param_idx_sig_i, g_sigi);
            atomicAdd(du_dp + lj_param_idx_eps_i, g_epsi);

            atomicAdd(du_dp + lj_param_idx_sig_j, g_sigj);
            atomicAdd(du_dp + lj_param_idx_eps_j, g_epsj);
        }

        if (du_dl_buffer && !is_vanilla) {
            atomicAdd(du_dl_buffer + atom_i_idx, FLOAT_TO_FIXED_NONBONDED(real_du_dl));
        }

        if (u_buffer) {
            atomicAdd(u_buffer + atom_i_idx, energy);
        }
    }
}
