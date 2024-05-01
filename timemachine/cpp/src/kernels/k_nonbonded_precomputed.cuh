#pragma once

#include "k_nonbonded_common.cuh"

namespace timemachine {

// Shape of parameter array is identical to other nonbonded variants
// except that rows map to pairs instead of individual atoms
static const int PARAMS_PER_PAIR = PARAMS_PER_ATOM;

template <typename RealType>
void __global__ k_nonbonded_precomputed(
    const int M,                       // number of pairs
    const double *__restrict__ coords, // [N, 3] coordinates
    const double *__restrict__ params, // [M, 4] q_ij, s_ij, e_ij, w_offset_ij
    const double *__restrict__ box,    // box vectors
    const int *__restrict__ pair_idxs, // [M, 2] pair-list of atoms
    const RealType beta,
    const RealType cutoff_squared,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    __int128 *__restrict__ u_buffer) {

    const RealType box_x = box[0 * 3 + 0];
    const RealType box_y = box[1 * 3 + 1];
    const RealType box_z = box[2 * 3 + 2];

    const RealType inv_box_x = 1 / box_x;
    const RealType inv_box_y = 1 / box_y;
    const RealType inv_box_z = 1 / box_z;

    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (pair_idx < M) {

        int params_ij_idx = pair_idx * PARAMS_PER_PAIR;
        RealType q_ij = params[params_ij_idx + PARAM_OFFSET_CHARGE];
        RealType sig_ij = params[params_ij_idx + PARAM_OFFSET_SIG];
        RealType eps_ij = params[params_ij_idx + PARAM_OFFSET_EPS];
        RealType delta_w = params[params_ij_idx + PARAM_OFFSET_W];

        unsigned long long g_q_ij = 0;
        unsigned long long g_sig_ij = 0;
        unsigned long long g_eps_ij = 0;
        unsigned long long g_dw_ij = 0;

        int atom_i_idx = pair_idxs[pair_idx * 2 + 0];
        int atom_j_idx = pair_idxs[pair_idx * 2 + 1];

        RealType ci_x = coords[atom_i_idx * 3 + 0];
        RealType ci_y = coords[atom_i_idx * 3 + 1];
        RealType ci_z = coords[atom_i_idx * 3 + 2];

        unsigned long long gi_x = 0;
        unsigned long long gi_y = 0;
        unsigned long long gi_z = 0;

        RealType cj_x = coords[atom_j_idx * 3 + 0];
        RealType cj_y = coords[atom_j_idx * 3 + 1];
        RealType cj_z = coords[atom_j_idx * 3 + 2];

        unsigned long long gj_x = 0;
        unsigned long long gj_y = 0;
        unsigned long long gj_z = 0;

        RealType delta_x = ci_x - cj_x;
        RealType delta_y = ci_y - cj_y;
        RealType delta_z = ci_z - cj_z;

        delta_x -= box_x * nearbyint(delta_x * inv_box_x);
        delta_y -= box_y * nearbyint(delta_y * inv_box_y);
        delta_z -= box_z * nearbyint(delta_z * inv_box_z);

        __int128 energy = 0;

        RealType d2_ij = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z + delta_w * delta_w;

        if (d2_ij < cutoff_squared) {

            RealType d_ij = sqrt(d2_ij);

            RealType inv_dij = 1 / d_ij;

            if (q_ij != 0) {

                RealType damping_factor;
                RealType es_factor = real_es_factor(beta, d_ij, inv_dij, inv_dij * inv_dij, damping_factor);

                if (u_buffer) {
                    // energies
                    RealType coulomb = q_ij * inv_dij;
                    RealType nrg = damping_factor * coulomb;
                    energy += FLOAT_TO_FIXED_ENERGY<RealType>(nrg);
                }

                if (du_dx || du_dp) {
                    RealType du_dr = q_ij * es_factor;

                    RealType force_prefactor = du_dr * inv_dij;
                    if (du_dx) {
                        // forces
                        gi_x += FLOAT_TO_FIXED_NONBONDED(delta_x * force_prefactor);
                        gi_y += FLOAT_TO_FIXED_NONBONDED(delta_y * force_prefactor);
                        gi_z += FLOAT_TO_FIXED_NONBONDED(delta_z * force_prefactor);

                        gj_x += FLOAT_TO_FIXED_NONBONDED(-delta_x * force_prefactor);
                        gj_y += FLOAT_TO_FIXED_NONBONDED(-delta_y * force_prefactor);
                        gj_z += FLOAT_TO_FIXED_NONBONDED(-delta_z * force_prefactor);
                    }

                    if (du_dp) {
                        // du/dp
                        g_q_ij += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(damping_factor * inv_dij);
                        g_dw_ij += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(delta_w * force_prefactor);
                    }
                }
            }

            if (eps_ij != 0 && sig_ij != 0) {
                RealType d4_ij = d2_ij * d2_ij;
                RealType d6_ij = d4_ij * d2_ij;

                RealType sig2_ij = sig_ij * sig_ij;
                RealType sig4_ij = sig2_ij * sig2_ij;
                RealType sig6_ij = sig4_ij * sig2_ij;
                RealType du_de;
                if (u_buffer || du_dp) {
                    RealType sig2_inv_d2ij = (sig_ij * inv_dij) * (sig_ij * inv_dij);
                    RealType sig4_inv_d4ij = sig2_inv_d2ij * sig2_inv_d2ij;
                    RealType sig6_inv_d6ij = sig4_inv_d4ij * sig2_inv_d2ij;
                    du_de = static_cast<RealType>(4.0) * (sig6_inv_d6ij - 1) * sig6_inv_d6ij;
                    if (u_buffer) {
                        // energies
                        RealType nrg = eps_ij * du_de;
                        energy += FLOAT_TO_FIXED_ENERGY<RealType>(nrg);
                    }
                }

                if (du_dx || du_dp) {
                    RealType d12_ij = d6_ij * d6_ij;
                    RealType du_dr = eps_ij * static_cast<RealType>(24.0) * sig6_ij *
                                     (d6_ij - static_cast<RealType>(2.0) * sig6_ij) / (d12_ij * d_ij);

                    RealType force_prefactor = du_dr * inv_dij;
                    if (du_dx) {
                        gi_x += FLOAT_TO_FIXED_NONBONDED(delta_x * force_prefactor);
                        gi_y += FLOAT_TO_FIXED_NONBONDED(delta_y * force_prefactor);
                        gi_z += FLOAT_TO_FIXED_NONBONDED(delta_z * force_prefactor);

                        gj_x += FLOAT_TO_FIXED_NONBONDED(-delta_x * force_prefactor);
                        gj_y += FLOAT_TO_FIXED_NONBONDED(-delta_y * force_prefactor);
                        gj_z += FLOAT_TO_FIXED_NONBONDED(-delta_z * force_prefactor);

                        atomicAdd(du_dx + atom_i_idx * 3 + 0, gi_x);
                        atomicAdd(du_dx + atom_i_idx * 3 + 1, gi_y);
                        atomicAdd(du_dx + atom_i_idx * 3 + 2, gi_z);

                        atomicAdd(du_dx + atom_j_idx * 3 + 0, gj_x);
                        atomicAdd(du_dx + atom_j_idx * 3 + 1, gj_y);
                        atomicAdd(du_dx + atom_j_idx * 3 + 2, gj_z);
                    }

                    if (du_dp) {
                        RealType du_ds = static_cast<RealType>(-24.0) * eps_ij * (sig4_ij * sig_ij) *
                                         (d6_ij - static_cast<RealType>(2.0) * sig6_ij) / d12_ij;

                        g_eps_ij += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(du_de);
                        g_sig_ij += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(du_ds);
                        g_dw_ij += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(delta_w * force_prefactor);

                        atomicAdd(du_dp + params_ij_idx + PARAM_OFFSET_CHARGE, g_q_ij);
                        atomicAdd(du_dp + params_ij_idx + PARAM_OFFSET_SIG, g_sig_ij);
                        atomicAdd(du_dp + params_ij_idx + PARAM_OFFSET_EPS, g_eps_ij);
                        atomicAdd(du_dp + params_ij_idx + PARAM_OFFSET_W, g_dw_ij);
                    }
                }
            }
        }
        // Always set the energy to avoid having to running memset on energy buffer
        if (u_buffer) {
            u_buffer[pair_idx] = energy;
        }
        pair_idx += gridDim.x * blockDim.x;
    }
}

} // namespace timemachine
