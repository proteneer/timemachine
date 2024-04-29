#include "../fixed_point.hpp"
#include "k_fixed_point.cuh"

namespace timemachine {

// branchless implementation of piecewise function
template <typename RealType>
RealType __device__ __forceinline__ compute_flat_bottom_energy(RealType k, RealType r, RealType rmin, RealType rmax) {
    RealType r_gt_rmax = static_cast<RealType>(r > rmax);
    RealType r_lt_rmin = static_cast<RealType>(r < rmin);
    RealType d_rmin = r - rmin;
    RealType d_rmin_2 = d_rmin * d_rmin;
    RealType d_rmin_4 = d_rmin_2 * d_rmin_2;

    RealType d_rmax = r - rmax;
    RealType d_rmax_2 = d_rmax * d_rmax;
    RealType d_rmax_4 = d_rmax_2 * d_rmax_2;

    return (k / static_cast<RealType>(4.0)) * ((r_lt_rmin * d_rmin_4) + (r_gt_rmax * d_rmax_4));
}

template <typename RealType>
void __global__ k_log_probability_selection(
    const int N,                             // Num atoms
    const double kBT,                        // BOLTZ * temperature
    const float radius,                      // Radius, corresponds to r_max for flat bottom
    const float k,                           // Constant restraint value
    const unsigned int reference_idx,        // Idx that the probability is specific to
    const double *__restrict__ coords,       // [N, 3]
    const double *__restrict__ box,          // [3, 3]
    const float *__restrict__ probabilities, // [N] probabilities of selection
    unsigned int *__restrict__ selected      // [N] idx array, N if idx is not selected, else idx of coordinate
) {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const float radius_sq = radius * radius;
    const RealType bx = box[0 * 3 + 0];
    const RealType by = box[1 * 3 + 1];
    const RealType bz = box[2 * 3 + 2];

    const RealType inv_bx = 1 / bx;
    const RealType inv_by = 1 / by;
    const RealType inv_bz = 1 / bz;

    RealType atom_atom_dx = coords[idx * 3 + 0] - coords[reference_idx * 3 + 0];
    RealType atom_atom_dy = coords[idx * 3 + 1] - coords[reference_idx * 3 + 1];
    RealType atom_atom_dz = coords[idx * 3 + 2] - coords[reference_idx * 3 + 2];

    atom_atom_dx -= bx * nearbyint(atom_atom_dx * inv_bx);
    atom_atom_dy -= by * nearbyint(atom_atom_dy * inv_by);
    atom_atom_dz -= bz * nearbyint(atom_atom_dz * inv_bz);

    const RealType distance_sq =
        atom_atom_dx * atom_atom_dx + atom_atom_dy * atom_atom_dy + atom_atom_dz * atom_atom_dz;

    RealType prob = 1.0;
    if (distance_sq >= radius_sq) {
        RealType energy = compute_flat_bottom_energy<RealType>(
            static_cast<RealType>(k),
            sqrt(distance_sq),
            static_cast<RealType>(0.0), // Any value works just fine here
            static_cast<RealType>(radius));

        prob = exp(-energy / kBT);
    }
    // Exclude the reference idx, should be considered frozen
    selected[idx] = (idx != reference_idx && prob >= probabilities[idx]) ? idx : N;
}

template <typename RealType>
void __global__ k_flat_bottom_bond(
    const int B, // number of bonds
    const double *__restrict__ coords,
    const double *__restrict__ box,
    const double *__restrict__ params, // [B, 3]
    const int *__restrict__ bond_idxs, // [B, 2]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    __int128 *__restrict__ u) {

    // which bond
    const auto b_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (b_idx >= B) {
        return;
    }

    // which atoms
    const int num_atoms = 2;
    int atoms_idx = b_idx * num_atoms;
    int src_idx = bond_idxs[atoms_idx + 0];
    int dst_idx = bond_idxs[atoms_idx + 1];

    // look up params
    const int num_params = 3;
    int params_idx = b_idx * num_params;
    int k_idx = params_idx + 0;
    int rmin_idx = params_idx + 1;
    int rmax_idx = params_idx + 2;

    RealType k = params[k_idx];
    RealType rmin = params[rmin_idx];
    RealType rmax = params[rmax_idx];

    // compute common subexpressions involving distance, displacements
    RealType dx[3];
    RealType r2 = 0;
#pragma unroll
    for (int d = 0; d < 3; d++) {
        double delta = coords[src_idx * 3 + d] - coords[dst_idx * 3 + d];
        delta -= box[d * 3 + d] * nearbyint(delta / box[d * 3 + d]);
        dx[d] = delta;
        r2 += delta * delta;
    }
    RealType r = sqrt(r2);

    // branches -> masks
    RealType r_gt_rmax = static_cast<RealType>(r > rmax);
    RealType r_lt_rmin = static_cast<RealType>(r < rmin);
    if (u) {
        RealType u_real = compute_flat_bottom_energy<RealType>(k, r, rmin, rmax);

        // Always set the energy buffer value to ensure buffer is initialized
        u[b_idx] = FLOAT_TO_FIXED_ENERGY<RealType>(u_real);
    }
    if (du_dp || du_dx) {
        RealType d_r_min = r - rmin;
        RealType d_r_max = r - rmax;
        RealType d_rmin_3 = d_r_min * d_r_min * d_r_min;
        RealType d_rmax_3 = d_r_max * d_r_max * d_r_max;
        if (du_dp) {
            // compute parameter derivatives
            RealType du_dk_real =
                (r_gt_rmax * ((d_rmax_3 * d_r_max) / static_cast<RealType>(4)) +
                 (r_lt_rmin * ((d_rmin_3 * d_r_min) / static_cast<RealType>(4))));
            RealType du_drmin_real = r_lt_rmin * (-k * d_rmin_3);
            RealType du_drmax_real = r_gt_rmax * (-k * d_rmax_3);

            // cast float -> fixed
            unsigned long long du_dk = FLOAT_TO_FIXED_BONDED<RealType>(du_dk_real);
            unsigned long long du_drmin = FLOAT_TO_FIXED_BONDED<RealType>(du_drmin_real);
            unsigned long long du_drmax = FLOAT_TO_FIXED_BONDED<RealType>(du_drmax_real);

            // increment du_dp array
            atomicAdd(du_dp + k_idx, du_dk);
            atomicAdd(du_dp + rmin_idx, du_drmin);
            atomicAdd(du_dp + rmax_idx, du_drmax);
        }

        if (du_dx) {
            RealType du_dr = k * ((r_gt_rmax * d_rmax_3) + (r_lt_rmin * d_rmin_3));

            RealType inv_r = 1 / r;
#pragma unroll
            for (int d = 0; d < 3; d++) {
                // compute du/dcoords
                RealType du_dsrc_real = du_dr * dx[d] * inv_r;
                RealType du_ddst_real = -du_dsrc_real;

                // cast float -> fixed
                unsigned long long du_dsrc = FLOAT_TO_FIXED_BONDED<RealType>(du_dsrc_real);
                unsigned long long du_ddst = FLOAT_TO_FIXED_BONDED<RealType>(du_ddst_real);

                // increment du_dx array
                atomicAdd(du_dx + src_idx * 3 + d, du_dsrc);
                atomicAdd(du_dx + dst_idx * 3 + d, du_ddst);
            }
        }
    }
}

} // namespace timemachine
