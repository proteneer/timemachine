#pragma once

#include "../nonbonded_common.hpp"
#include "k_fixed_point.cuh"

namespace timemachine {

// Empirically selected number of kernel blocks, focusing on the cuda 8.* architectures
static const int NONBONDED_KERNEL_BLOCKS = 2048;
static const int NONBONDED_KERNEL_THREADS_PER_BLOCK = 256;

#define PI 3.141592653589793115997963468544185161
#define TWO_OVER_SQRT_PI 1.128379167095512595889238330988549829708

template <typename RealType> RealType __device__ __forceinline__ switch_fn(RealType dij) {
    constexpr RealType cutoff = 1.2;
    constexpr RealType inv_cutoff = 1 / cutoff;

    RealType pi = static_cast<RealType>(PI);
    //RealType dij_k = dij / cutoff; // TODO: multiply by inv cutoff
    RealType dij_k = dij * inv_cutoff;

    // exponentiation
    RealType dij_k2 = dij_k * dij_k;
    RealType dij_k4 = dij_k2 * dij_k2;
    RealType dij_k8 = dij_k4 * dij_k4;

    RealType cos_arg = cos(0.5 * pi * dij_k8);

    // exponentiation
    RealType cos_arg3 = cos_arg * cos_arg * cos_arg;
    return cos_arg3;
}

// same as above, but with cos -> __cosf
float __device__ __forceinline__ switch_fn(float dij) {
    constexpr float cutoff = 1.2;
    constexpr float inv_cutoff = 1 / cutoff;

    float pi = static_cast<float>(PI);
    //RealType dij_k = dij / cutoff; // TODO: multiply by inv cutoff
    float dij_k = dij * inv_cutoff;

    // exponentiation
    float dij_k2 = dij_k * dij_k;
    float dij_k4 = dij_k2 * dij_k2;
    float dij_k8 = dij_k4 * dij_k4;

    float cos_arg = __cosf(0.5 * (pi * dij_k8));

    // exponentiation
    float cos_arg3 = cos_arg * cos_arg * cos_arg;
    return cos_arg3;
}

template <typename RealType> RealType __device__ __forceinline__ d_switch_fn_dr(RealType dij) {
    constexpr RealType cutoff = 1.2;
    RealType pi = static_cast<RealType>(PI);

    // exponentiation
    RealType inv_cutoff = 1.0 / cutoff;
    RealType k2 = inv_cutoff * inv_cutoff;
    RealType k4 = k2 * k2;
    RealType k8 = k4 * k4;

    RealType dij2 = dij * dij;
    RealType dij4 = dij2 * dij2;
    RealType dij7 = dij4 * dij2 * dij;
    RealType dij8 = dij4 * dij4;

    RealType dij_k8 = dij8 * k8;

    RealType arg = 0.5 * pi * dij_k8;

    RealType cos_arg = cos(arg);
    RealType cos_arg2 = cos_arg * cos_arg;

    return -12 * pi * dij7 * sin(arg) * cos_arg2 / k8;
}

// same as above, but with (sin(a), cos(a)) -> __sincosf(a)
float __device__ __forceinline__ d_switch_fn_dr(float dij) {

    constexpr float cutoff = 1.2;
    float pi = static_cast<float>(PI);

    // exponentiation
    float inv_cutoff = 1.0 / cutoff;
    float k2 = inv_cutoff * inv_cutoff;
    float k4 = k2 * k2;
    float k8 = k4 * k4;

    float dij2 = dij * dij;
    float dij4 = dij2 * dij2;
    float dij7 = dij4 * dij2 * dij;
    float dij8 = dij4 * dij4;

    float dij_k8 = dij8 * k8;

    float arg = 0.5 * pi * dij_k8;

    float sin_arg;
    float cos_arg;
    __sincosf(arg, &sin_arg, &cos_arg);
    float cos_arg2 = cos_arg * cos_arg;

    return -12 * pi * dij7 * sin_arg * cos_arg2 / k8;
}

float __device__ __forceinline__ fast_erfc(float x) {
    // TODO: consider using fasterfc implementations listed in this thread:
    // https://forums.developer.nvidia.com/t/calling-all-juffas-whats-up-with-erfcf-nowadays/262973/4

    float exp_beta_x2 = __expf(-x * x);
    // (ytz) 5th order gaussian polynomial approximation, we need the exp(-x^2) anyways for the chain rule
    // so we use last variant in https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    float t = 1.0f / (1.0f + 0.3275911f * x);
    return (0.254829592f + (-0.284496736f + (1.421413741f + (-1.453152027f + 1.061405429f * t) * t) * t) * t) * t *
           exp_beta_x2;
}

template <typename RealType> RealType __device__ __forceinline__ d_erfc_beta_r_dr(RealType beta, RealType dij) {
    // -2 beta exp(-(beta dij)^2) / sqrt(pi)
    RealType beta_dij = beta * dij;
    RealType exp_beta_dij_2 = exp(-beta_dij * beta_dij);
    return -static_cast<RealType>(TWO_OVER_SQRT_PI) * beta * exp_beta_dij_2;
}

// same as above, but with exp -> __expf
float __device__ __forceinline__ d_erfc_beta_r_dr(float beta, float dij) {
    // (ytz): max ulp error is: 2 + floor(abs(1.16 * x))
    float beta_dij = beta * dij;
    float exp_beta_dij_2 = __expf(-beta_dij * beta_dij);
    return -static_cast<float>(TWO_OVER_SQRT_PI) * beta * exp_beta_dij_2;
}

template <typename RealType>
RealType __device__ __forceinline__
real_es_factor(RealType real_beta, RealType dij, RealType inv_dij, RealType inv_d2ij, RealType &damping_factor) {
    RealType beta_dij = real_beta * dij;
    RealType erfc_beta_dij = erfc(beta_dij);

    damping_factor = erfc_beta_dij * switch_fn(dij);
    RealType damping_factor_prime =
        (erfc_beta_dij * d_switch_fn_dr(dij)) + (d_erfc_beta_r_dr(real_beta, dij) * switch_fn(dij));
    RealType d_es_dr = damping_factor_prime * inv_dij - damping_factor * inv_d2ij;
    return d_es_dr;
}

float __device__ __forceinline__
real_es_factor(float real_beta, float dij, float inv_dij, float inv_d2ij, float &damping_factor) {
    float beta_dij = real_beta * dij;
    float erfc_beta_dij = fast_erfc(beta_dij);

    damping_factor = erfc_beta_dij * switch_fn(dij);
    float damping_factor_prime =
        (erfc_beta_dij * d_switch_fn_dr(dij)) + (d_erfc_beta_r_dr(real_beta, dij) * switch_fn(dij));
    float d_es_dr = damping_factor_prime * inv_dij - damping_factor * inv_d2ij;
    return d_es_dr;
}

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
    RealType &damping_factor,
    RealType &es_prefactor,
    RealType &u) {
    inv_dij = rsqrt(d2ij);

    dij = d2ij * inv_dij;
    inv_d2ij = inv_dij * inv_dij;

    RealType qij = qi * qj;
    es_prefactor = charge_scale * qij * inv_dij * real_es_factor(beta, dij, inv_dij, inv_d2ij, damping_factor);

    if (COMPUTE_U) {
        u = charge_scale * qij * inv_dij * damping_factor;
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

} // namespace timemachine
