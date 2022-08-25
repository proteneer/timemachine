#include "../fixed_point.hpp"
#include "k_fixed_point.cuh"

// branchless implementation of piecewise function
template <typename RealType>
RealType __device__ __forceinline__ compute_flat_bottom_energy(
    RealType k, RealType r_lt_rmin, RealType r_gt_rmax, RealType r, RealType rmin, RealType rmax) {
    return (k / 4) * (r_lt_rmin * (pow(r - rmin, 4)) + r_gt_rmax * (pow(r - rmax, 4)));
}

template <typename RealType>
void __global__ k_log_probability_selection(
    const int N,                             // Num atoms
    const double kBt,                        // BOLTZ * temperature
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

    const RealType distance =
        sqrt(atom_atom_dx * atom_atom_dx + atom_atom_dy * atom_atom_dy + atom_atom_dz * atom_atom_dz);

    RealType prob = 1.0;
    if (distance >= radius) {
        RealType energy = compute_flat_bottom_energy<RealType>(
            static_cast<RealType>(k),
            static_cast<RealType>(0.0), // set to 0 to avoid using r_min length
            static_cast<RealType>(1.0), // set to 1.0 to use r_max length
            distance,
            static_cast<RealType>(0.0), // Any value works just fine here
            static_cast<RealType>(radius));

        prob = exp(-energy / kBt);
    }
    // Exclude the reference idx, should be considered frozen
    selected[idx] = (idx != reference_idx && prob >= probabilities[idx]) ? idx : N;
}

template <typename RealType>
void __global__ k_flat_bottom_bond(
    const int B, // number of bonds
    const double *__restrict__ coords,
    const double *__restrict__ box,
    const double *__restrict__ params, // [P, 3]
    const int *__restrict__ bond_idxs, // [B, 2]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ u) {

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
        RealType u_real = compute_flat_bottom_energy<RealType>(k, r_lt_rmin, r_gt_rmax, r, rmin, rmax);

        // cast float -> fixed
        auto sum_u = FLOAT_TO_FIXED_BONDED<RealType>(u_real);

        // atomic add to u array
        atomicAdd(u + src_idx, sum_u);
    }

    if (du_dp) {
        // compute parameter derivatives
        RealType du_dk_real = (r_gt_rmax * (pow(r - rmax, 4) / 4)) + (r_lt_rmin * (pow(r - rmin, 4) / 4));
        RealType du_drmin_real = r_lt_rmin * (-k * pow(r - rmin, 3));
        RealType du_drmax_real = r_gt_rmax * (-k * pow(r - rmax, 3));

        // cast float -> fixed
        auto du_dk = FLOAT_TO_FIXED_BONDED<RealType>(du_dk_real);
        auto du_drmin = FLOAT_TO_FIXED_BONDED<RealType>(du_drmin_real);
        auto du_drmax = FLOAT_TO_FIXED_BONDED<RealType>(du_drmax_real);

        // increment du_dp array
        atomicAdd(du_dp + k_idx, du_dk);
        atomicAdd(du_dp + rmin_idx, du_drmin);
        atomicAdd(du_dp + rmax_idx, du_drmax);
    }

    if (du_dx) {
        RealType du_dr = k * ((r_gt_rmax * pow(r - rmax, 3)) + (r_lt_rmin * pow(r - rmin, 3)));

        for (int d = 0; d < 3; d++) {
            // compute du/dcoords
            RealType du_dsrc_real = du_dr * dx[d] / r;
            RealType du_ddst_real = -du_dsrc_real;

            // cast float -> fixed
            auto du_dsrc = FLOAT_TO_FIXED_BONDED<RealType>(du_dsrc_real);
            auto du_ddst = FLOAT_TO_FIXED_BONDED<RealType>(du_ddst_real);

            // increment du_dx array
            atomicAdd(du_dx + src_idx * 3 + d, du_dsrc);
            atomicAdd(du_dx + dst_idx * 3 + d, du_ddst);
        }
    }
}
