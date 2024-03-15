#pragma once

#include "../fixed_point.hpp"
#include "k_fixed_point.cuh"

#include "chiral_utils.cuh" // Vector, cross_product, etc.

namespace timemachine {

template <typename RealType, int D>
void __global__ k_harmonic_angle(
    const int A,                        // number of bonds
    const double *__restrict__ coords,  // [N, 3]
    const double *__restrict__ params,  // [P, 2]
    const int *__restrict__ angle_idxs, // [A, 3]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    __int128 *__restrict__ u) {

    const auto a_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (a_idx >= A) {
        return;
    }

    int i_idx = angle_idxs[a_idx * D + 0];
    int j_idx = angle_idxs[a_idx * D + 1];
    int k_idx = angle_idxs[a_idx * D + 2];

    RealType rji[D];  // vector from j to i
    RealType rjk[D];  // vector from j to k
    RealType nij = 0; // initialize your summed variables!
    RealType njk = 0; // initialize your summed variables!
    RealType top = 0;
    // first pass, compute the norms
    for (int d = 0; d < D; d++) {
        RealType vji = coords[i_idx * D + d] - coords[j_idx * D + d];
        RealType vjk = coords[k_idx * D + d] - coords[j_idx * D + d];
        rji[d] = vji;
        rjk[d] = vjk;
        nij += vji * vji;
        njk += vjk * vjk;
        top += vji * vjk;
    }

    nij = sqrt(nij);
    njk = sqrt(njk);

    RealType hi = 0;
    RealType lo = 0;

    // second pass, compute the hi/lo values
    for (int d = 0; d < D; d++) {
        RealType vji = coords[i_idx * D + d] - coords[j_idx * D + d];
        RealType vjk = coords[k_idx * D + d] - coords[j_idx * D + d];
        RealType a = njk * vji - nij * vjk;
        RealType b = njk * vji + nij * vjk;
        hi += a * a;
        lo += b * b;
    }

    RealType y = sqrt(hi);
    RealType x = sqrt(lo);
    RealType angle = 2 * atan2(y, x);

    int ka_idx = a_idx * 2 + 0;
    int a0_idx = a_idx * 2 + 1;

    RealType ka = params[ka_idx];
    RealType a0 = params[a0_idx];
    RealType delta = angle - a0;

    auto a = Vector<RealType>({rji[0], rji[1], rji[2]});
    auto b = Vector<RealType>({rjk[0], rjk[1], rjk[2]});

    RealType a_norm = a.norm();
    RealType b_norm = b.norm();

    auto vtp_i = cross_product(a, cross_product(a, b));
    auto vtp_k = cross_product(b, cross_product(b, a));

    RealType prefactor = ka * delta;

    // compute the derivative using the vector triple product
    if (du_dx) {

        auto coeff_i = prefactor * (1 / a_norm);
        auto coeff_k = prefactor * (1 / b_norm);

        // formal singularity, unremovable
        Vector<RealType> grad_i = (vtp_i.norm() == 0) ? Vector<RealType>({0, 0, 0}) : vtp_i.unit();
        Vector<RealType> grad_k = (vtp_k.norm() == 0) ? Vector<RealType>({0, 0, 0}) : vtp_k.unit();

        RealType dx_i = coeff_i * grad_i.x;
        RealType dy_i = coeff_i * grad_i.y;
        RealType dz_i = coeff_i * grad_i.z;

        atomicAdd(du_dx + i_idx * D + 0, FLOAT_TO_FIXED_BONDED<RealType>(dx_i));
        atomicAdd(du_dx + i_idx * D + 1, FLOAT_TO_FIXED_BONDED<RealType>(dy_i));
        atomicAdd(du_dx + i_idx * D + 2, FLOAT_TO_FIXED_BONDED<RealType>(dz_i));

        RealType dx_k = coeff_k * grad_k.x;
        RealType dy_k = coeff_k * grad_k.y;
        RealType dz_k = coeff_k * grad_k.z;

        atomicAdd(du_dx + k_idx * D + 0, FLOAT_TO_FIXED_BONDED<RealType>(dx_k));
        atomicAdd(du_dx + k_idx * D + 1, FLOAT_TO_FIXED_BONDED<RealType>(dy_k));
        atomicAdd(du_dx + k_idx * D + 2, FLOAT_TO_FIXED_BONDED<RealType>(dz_k));

        atomicAdd(du_dx + j_idx * D + 0, FLOAT_TO_FIXED_BONDED<RealType>(-dx_i - dx_k));
        atomicAdd(du_dx + j_idx * D + 1, FLOAT_TO_FIXED_BONDED<RealType>(-dy_i - dy_k));
        atomicAdd(du_dx + j_idx * D + 2, FLOAT_TO_FIXED_BONDED<RealType>(-dz_i - dz_k));
    }

    if (du_dp) {
        RealType dka_grad = delta * delta / 2;
        atomicAdd(du_dp + ka_idx, FLOAT_TO_FIXED_BONDED(dka_grad));
        RealType da0_grad = -delta * ka;
        atomicAdd(du_dp + a0_idx, FLOAT_TO_FIXED_BONDED(da0_grad));
    }

    if (u) {
        u[a_idx] = FLOAT_TO_FIXED_ENERGY<RealType>((ka / 2) * delta * delta);
    }
}

} // namespace timemachine
