#pragma once

#include "k_fixed_point.cuh"

typedef void (*k_nonbonded_fn)(
    const int N,
    const int NR,
    const double *__restrict__ coords,
    const double *__restrict__ params, // [N]
    const double *__restrict__ box,
    const double *__restrict__ dl_dp,
    const double *__restrict__ coords_w, // 4D coords
    const double *__restrict__ dw_dl,    // 4D derivatives
    const double beta,
    const double cutoff,
    const unsigned int *__restrict__ row_idxs,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    unsigned long long *__restrict__ du_dl_buffer,
    unsigned long long *__restrict__ u_buffer);

#define PI 3.141592653589793115997963468544185161
#define TWO_OVER_SQRT_PI 1.128379167095512595889238330988549829708

double __device__ __forceinline__ real_es_factor(double real_beta, double dij, double inv_d2ij, double &erfc_beta_dij) {
    double beta_dij = real_beta * dij;
    double exp_beta_dij_2 = exp(-beta_dij * beta_dij);
    erfc_beta_dij = erfc(beta_dij);
    return -inv_d2ij * (static_cast<double>(TWO_OVER_SQRT_PI) * beta_dij * exp_beta_dij_2 + erfc_beta_dij);
}

float __device__ __forceinline__ real_es_factor(float real_beta, float dij, float inv_d2ij, float &erfc_beta_dij) {
    float beta_dij = real_beta * dij;
    // max ulp error is: 2 + floor(abs(1.16 * x))
    float exp_beta_dij_2 = __expf(-beta_dij * beta_dij);
    // 5th order gaussian polynomial approximation, we need the exp(-x^2) anyways for the chain rule
    // so we use last variant in https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    float t = 1.0f / (1.0f + 0.3275911f * beta_dij);
    erfc_beta_dij = (0.254829592f + (-0.284496736f + (1.421413741f + (-1.453152027f + 1.061405429f * t) * t) * t) * t) *
                    t * exp_beta_dij_2;
    return -inv_d2ij * (static_cast<float>(TWO_OVER_SQRT_PI) * beta_dij * exp_beta_dij_2 + erfc_beta_dij);
}

void __global__ k_arange(int N, unsigned int *arr);

// Compute the terms associated with electrostatics.
// This is pulled out into a function to ensure that the same bit values
// are computed to ensure that that the fixed point values are exactly the same regardless
// of where the values are computed.
template <typename RealType, bool COMPUTE_U>
void __device__ __forceinline__ compute_electrostatics(
    const RealType charge_scale,
    const RealType qi,
    const RealType qj,
    const RealType d2ij,
    const RealType beta,
    RealType &dij,
    RealType &inv_dij,
    RealType &inv_d2ij,
    RealType &ebd,
    RealType &es_prefactor,
    RealType &u) {
    inv_dij = rsqrt(d2ij);

    dij = d2ij * inv_dij;
    inv_d2ij = inv_dij * inv_dij;

    RealType qij = qi * qj;
    es_prefactor = charge_scale * qij * inv_dij * real_es_factor(beta, dij, inv_d2ij, ebd);

    if (COMPUTE_U) {
        u = charge_scale * qij * inv_dij * ebd;
    }
}

// Handles the computation related to the LJ terms.
// This is pulled out into a function to ensure that the same bit values
// are computed to ensure that that the fixed point values are exactly the same regardless
// of where the values are computed.
template <typename RealType, bool COMPUTE_U>
void __device__ __forceinline__ compute_lj(
    RealType lj_scale,
    RealType eps_i,
    RealType eps_j,
    RealType sig_i,
    RealType sig_j,
    RealType inv_dij,
    RealType inv_d2ij,
    RealType &u,
    RealType &delta_prefactor,
    RealType &sig_grad,
    RealType &eps_grad) {
    RealType eps_ij = eps_i * eps_j;
    RealType sig_ij = sig_i + sig_j;

    RealType sig_inv_dij = sig_ij * inv_dij;
    RealType sig2_inv_d2ij = sig_inv_dij * sig_inv_dij;
    RealType sig4_inv_d4ij = sig2_inv_d2ij * sig2_inv_d2ij;
    RealType sig6_inv_d6ij = sig4_inv_d4ij * sig2_inv_d2ij;
    RealType sig6_inv_d8ij = sig6_inv_d6ij * inv_d2ij;
    RealType sig5_inv_d6ij = sig_ij * sig4_inv_d4ij * inv_d2ij;

    RealType lj_prefactor = lj_scale * eps_ij * sig6_inv_d8ij * (sig6_inv_d6ij * 48 - 24);
    if (COMPUTE_U) {
        u += lj_scale * 4 * eps_ij * (sig6_inv_d6ij - 1) * sig6_inv_d6ij;
    }

    delta_prefactor -= lj_prefactor;

    sig_grad = lj_scale * 24 * eps_ij * sig5_inv_d6ij * (2 * sig6_inv_d6ij - 1);
    eps_grad = lj_scale * 4 * (sig6_inv_d6ij - 1) * sig6_inv_d6ij;
}

void __global__
k_add_ull_to_ull(const int N, const unsigned long long *__restrict__ src, unsigned long long *__restrict__ dest);
