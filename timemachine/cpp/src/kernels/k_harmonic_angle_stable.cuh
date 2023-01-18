#include "../fixed_point.hpp"
#include "k_fixed_point.cuh"

template <typename RealType>
void __global__ k_harmonic_angle_stable(
    const int A,                        // number of angles
    const double *__restrict__ coords,  // [N, 3]
    const double *__restrict__ params,  // [P, 3]
    const int *__restrict__ angle_idxs, // [A, 3]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ u) {

    const auto a_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (a_idx >= A) {
        return;
    }

    int i_idx = angle_idxs[a_idx * 3 + 0];
    int j_idx = angle_idxs[a_idx * 3 + 1];
    int k_idx = angle_idxs[a_idx * 3 + 2];

    RealType rij[3];
    RealType rkj[3];

    for (int d = 0; d < 3; d++) {
        rij[d] = coords[j_idx * 3 + d] - coords[i_idx * 3 + d];
        rkj[d] = coords[j_idx * 3 + d] - coords[k_idx * 3 + d];
    }

    const int ka_idx = a_idx * 3 + 0;
    const int a0_idx = a_idx * 3 + 1;
    const int eps_idx = a_idx * 3 + 2;

    RealType ka = params[ka_idx];
    RealType a0 = params[a0_idx];
    RealType eps = params[eps_idx];

    RealType eps2 = eps * eps;

    RealType rij_dot_rkj = eps2;
    RealType rij_dot_rij = eps2;
    RealType rkj_dot_rkj = eps2;

    for (int d = 0; d < 3; d++) {
        rij_dot_rkj += rij[d] * rkj[d];
        rij_dot_rij += rij[d] * rij[d];
        rkj_dot_rkj += rkj[d] * rkj[d];
    }

    RealType norm = sqrt(rij_dot_rij * rkj_dot_rkj);
    RealType delta = rij_dot_rkj / norm - cos(a0);

    RealType cij = rij_dot_rkj / rij_dot_rij;
    RealType ckj = rij_dot_rkj / rkj_dot_rkj;

    if (du_dx) {
        RealType c = ka * delta / norm;

        for (int d = 0; d < 3; d++) {
            RealType g_i = cij * rij[d] - rkj[d];
            atomicAdd(du_dx + i_idx * 3 + d, FLOAT_TO_FIXED_BONDED<RealType>(c * g_i));

            RealType g_j = (1 - cij) * rij[d] + (1 - ckj) * rkj[d];
            atomicAdd(du_dx + j_idx * 3 + d, FLOAT_TO_FIXED_BONDED<RealType>(c * g_j));

            RealType g_k = ckj * rkj[d] - rij[d];
            atomicAdd(du_dx + k_idx * 3 + d, FLOAT_TO_FIXED_BONDED<RealType>(c * g_k));
        }
    }

    if (du_dp) {
        RealType dka_grad = delta * delta / 2;
        atomicAdd(du_dp + ka_idx, FLOAT_TO_FIXED_BONDED(dka_grad));
        RealType da0_grad = ka * delta * sin(a0);
        atomicAdd(du_dp + a0_idx, FLOAT_TO_FIXED_BONDED(da0_grad));
        RealType deps_grad = ka * delta * eps * (2.0 - ckj - cij) / norm;
        atomicAdd(du_dp + eps_idx, FLOAT_TO_FIXED_BONDED(deps_grad));
    }

    if (u) {
        atomicAdd(u + i_idx, FLOAT_TO_FIXED_BONDED(ka / 2 * delta * delta));
    }
}
