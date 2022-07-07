#pragma once

#include "nonbonded_common.cuh"

template <typename RealType>
void __global__ k_nonbonded_precomputed(
    const int M,                          // number of pairs
    const double *__restrict__ coords,    // [N, 3] coordinates
    const double *__restrict__ params,    // [M, 3] q_ij, s_ij, e_ij
    const double *__restrict__ box,       // box vectors
    const double *__restrict__ w_offsets, // [M] for vdw and electrostatics
    const int *__restrict__ pair_idxs,    // [M, 2] pair-list of atoms
    const double beta,
    const double cutoff,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ u_buffer) {

    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= M) {
        return;
    }

    RealType q_ij = params[pair_idx * 3 + 0];
    RealType sig_ij = params[pair_idx * 3 + 1];
    RealType eps_ij = params[pair_idx * 3 + 2];

    unsigned long long g_q_ij = 0;
    unsigned long long g_sig_ij = 0;
    unsigned long long g_eps_ij = 0;

    int atom_i_idx = pair_idxs[pair_idx * 2 + 0];

    RealType ci_x = coords[atom_i_idx * 3 + 0];
    RealType ci_y = coords[atom_i_idx * 3 + 1];
    RealType ci_z = coords[atom_i_idx * 3 + 2];

    unsigned long long gi_x = 0;
    unsigned long long gi_y = 0;
    unsigned long long gi_z = 0;

    int atom_j_idx = pair_idxs[pair_idx * 2 + 1];

    RealType cj_x = coords[atom_j_idx * 3 + 0];
    RealType cj_y = coords[atom_j_idx * 3 + 1];
    RealType cj_z = coords[atom_j_idx * 3 + 2];

    unsigned long long gj_x = 0;
    unsigned long long gj_y = 0;
    unsigned long long gj_z = 0;

    RealType real_beta = static_cast<RealType>(beta);
    // RealType real_cutoff = static_cast<RealType>(cutoff);
    // RealType cutoff_squared = real_cutoff * real_cutoff;

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

    unsigned long long energy = 0;

    RealType delta_w = w_offsets[pair_idx];

    RealType d_ij = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z + delta_w * delta_w);

    if (d_ij >= cutoff) {
        return;
    }

    if (q_ij != 0) {

        // energies

        RealType nrg = q_ij * erfc(beta * d_ij) / d_ij;
        energy += FLOAT_TO_FIXED_NONBONDED(nrg);

        RealType du_dr = q_ij * ((2 * beta * d_ij * exp(-beta * beta * d_ij * d_ij)) / sqrt(PI) + erfc(beta * d_ij));
        du_dr /= -d_ij * d_ij;

        // forces
        gi_x += FLOAT_TO_FIXED_NONBONDED(du_dr * delta_x / d_ij);
        gi_y += FLOAT_TO_FIXED_NONBONDED(du_dr * delta_y / d_ij);
        gi_z += FLOAT_TO_FIXED_NONBONDED(du_dr * delta_z / d_ij);

        gj_x += FLOAT_TO_FIXED_NONBONDED(-du_dr * delta_x / d_ij);
        gj_y += FLOAT_TO_FIXED_NONBONDED(-du_dr * delta_y / d_ij);
        gj_z += FLOAT_TO_FIXED_NONBONDED(-du_dr * delta_z / d_ij);

        // du/dp
        g_q_ij += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(erfc(beta * d_ij) / d_ij);
    }

    if (eps_ij != 0 && sig_ij != 0) {

        // energies
        RealType nrg = 4 * eps_ij * (pow(sig_ij / d_ij, 12) - pow(sig_ij / d_ij, 6));
        energy += FLOAT_TO_FIXED_NONBONDED(nrg);

        RealType du_dr = eps_ij * 24 * pow(sig_ij, 6) * (pow(d_ij, 6) - 2 * pow(sig_ij, 6)) / pow(d_ij, 13);

        gi_x += FLOAT_TO_FIXED_NONBONDED(du_dr * delta_x / d_ij);
        gi_y += FLOAT_TO_FIXED_NONBONDED(du_dr * delta_y / d_ij);
        gi_z += FLOAT_TO_FIXED_NONBONDED(du_dr * delta_z / d_ij);

        gj_x += FLOAT_TO_FIXED_NONBONDED(-du_dr * delta_x / d_ij);
        gj_y += FLOAT_TO_FIXED_NONBONDED(-du_dr * delta_y / d_ij);
        gj_z += FLOAT_TO_FIXED_NONBONDED(-du_dr * delta_z / d_ij);

        RealType du_de = 4 * (pow(sig_ij / d_ij, 12) - pow(sig_ij / d_ij, 6));
        RealType du_ds = -24 * eps_ij * pow(sig_ij, 5) * (pow(d_ij, 6) - 2 * pow(sig_ij, 6)) / pow(d_ij, 12);

        g_eps_ij += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(du_de);
        g_sig_ij += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(du_ds);
    }

    if (du_dp) {
        atomicAdd(du_dp + pair_idx * 3 + 0, g_q_ij);
        atomicAdd(du_dp + pair_idx * 3 + 1, g_sig_ij);
        atomicAdd(du_dp + pair_idx * 3 + 2, g_eps_ij);
    }

    if (du_dx) {
        atomicAdd(du_dx + atom_i_idx * 3 + 0, gi_x);
        atomicAdd(du_dx + atom_i_idx * 3 + 1, gi_y);
        atomicAdd(du_dx + atom_i_idx * 3 + 2, gi_z);

        atomicAdd(du_dx + atom_j_idx * 3 + 0, gj_x);
        atomicAdd(du_dx + atom_j_idx * 3 + 1, gj_y);
        atomicAdd(du_dx + atom_j_idx * 3 + 2, gj_z);
    }

    if (u_buffer) {
        atomicAdd(u_buffer + atom_i_idx, energy);
    }
}
