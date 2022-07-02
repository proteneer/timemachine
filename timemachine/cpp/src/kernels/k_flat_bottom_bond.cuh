#include "../fixed_point.hpp"
#include "k_fixed_point.cuh"

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

    RealType box_diag[3];
    box_diag[0] = box[0 * 3 + 0];
    box_diag[1] = box[1 * 3 + 1];
    box_diag[2] = box[2 * 3 + 2];
    RealType inv_box[3];
    inv_box[0] = 1 / box_diag[0];
    inv_box[1] = 1 / box_diag[1];
    inv_box[2] = 1 / box_diag[2];

    RealType k = params[k_idx];
    RealType rmin = params[rmin_idx];
    RealType rmax = params[rmax_idx];

    // compute common subexpressions involving distance, displacements
    RealType dx[3];
    RealType r2 = 0;
    for (int d = 0; d < 3; d++) {
        RealType delta = coords[src_idx * 3 + d] - coords[dst_idx * 3 + d];
        delta -= box_diag[d] * nearbyint(delta * inv_box[d]);
        dx[d] = delta;
        r2 += delta * delta;
    }
    RealType r = sqrt(r2);

    // branches -> masks
    RealType r_gt_rmax = static_cast<RealType>(r > rmax);
    RealType r_lt_rmin = static_cast<RealType>(r < rmin);

    if (u) {
        // branchless implementation of piecewise function
        RealType u_real = (k / 4) * (r_lt_rmin * (pow(r - rmin, 4)) + r_gt_rmax * (pow(r - rmax, 4)));

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
