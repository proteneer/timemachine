#pragma once

#include "../nonbonded_common.hpp"
#include "k_fixed_point.cuh"

namespace timemachine {

// Empirically selected number of kernel blocks, focusing on the cuda 8.* architectures
static const int NONBONDED_KERNEL_BLOCKS = 2048;
static const int NONBONDED_KERNEL_THREADS_PER_BLOCK = 256;

#define PI 3.141592653589793115997963468544185161
#define TWO_OVER_SQRT_PI 1.128379167095512595889238330988549829708

// f64 switch fxn
double __device__ __forceinline__ switch_fn(double dij) {
    // constants
    constexpr double cutoff = 1.2;
    if (dij >= cutoff) {
        return 0.0;
    }
    constexpr double inv_cutoff = 1 / cutoff;
    constexpr double pi = static_cast<double>(PI);

    // (dij/cutoff)^8
    double dij_k = dij * inv_cutoff;
    double dij_k2 = dij_k * dij_k;
    double dij_k4 = dij_k2 * dij_k2;
    double dij_k8 = dij_k4 * dij_k4;

    // cos(0.5*pi*(dij/cutoff)^8)^3
    double cos_arg = cos(0.5 * pi * dij_k8);
    double cos_arg3 = cos_arg * cos_arg * cos_arg;
    return cos_arg3;
}

double __device__ __forceinline__ d_switch_fn_dr(double dij) {
    // constants
    constexpr double cutoff = 1.2;
    constexpr double pi = static_cast<double>(PI);

    if (dij >= cutoff) {
        return 0.0;
    }

    // cutoff^-8
    constexpr double inv_cutoff = 1.0 / cutoff;
    constexpr double k2 = inv_cutoff * inv_cutoff;
    constexpr double k4 = k2 * k2;
    constexpr double k8 = k4 * k4;

    // dij^7 and dij^8
    double dij2 = dij * dij;
    double dij4 = dij2 * dij2;
    double dij7 = dij4 * dij2 * dij;
    double dij8 = dij4 * dij4;

    // arg = 0.5 * pi * (dij / cutoff)^8
    double dij_k8 = dij8 * k8;
    double arg = 0.5 * pi * dij_k8;

    double sin_arg;
    double cos_arg;
    sincos(arg, &sin_arg, &cos_arg);

    double cos_arg2 = cos_arg * cos_arg;
    constexpr double minus_12_pi_k8 = -12 * pi * k8;

    return dij7 * sin_arg * cos_arg2 * minus_12_pi_k8;
}

double __device__ __forceinline__ d_erfc_beta_r_dr(double beta, double dij) {
    // -2 beta exp(-(beta dij)^2) / sqrt(pi)
    double beta_dij = beta * dij;
    double exp_beta_dij_2 = exp(-beta_dij * beta_dij);
    return -static_cast<double>(TWO_OVER_SQRT_PI) * beta * exp_beta_dij_2;
}

double __device__ __forceinline__
real_es_factor(double real_beta, double dij, double inv_dij, double inv_d2ij, double &damping_factor) {
    double beta_dij = real_beta * dij;
    double erfc_beta_dij = erfc(beta_dij);
    double sr = switch_fn(dij);

    // write erfc(beta * dij) * switch_fn(dij) into damping_factor
    damping_factor = erfc_beta_dij * sr;

    // chain rule
    double dsdr = d_switch_fn_dr(dij);
    double debd = d_erfc_beta_r_dr(real_beta, dij);
    double damping_factor_prime = (erfc_beta_dij * dsdr) + (debd * sr);
    double d_es_dr = damping_factor_prime * inv_dij - damping_factor * inv_d2ij;
    return d_es_dr;
}

// f32 code path merges (1) switch_fn and its deriv into switch_fn_and_deriv
// and (2) a fast erfc approximation and its deriv into fast_erfc_and_deriv
float __device__ __forceinline__ switch_fn_and_deriv(float dij, float &dsdr) {

    // constants
    constexpr float cutoff = 1.2;
    constexpr float pi = static_cast<float>(PI);
    constexpr float pi_over_2 = 0.5f * pi;
    constexpr float inv_cutoff = 1 / cutoff;

    if (dij >= cutoff) {
        dsdr = 0.0f;
        return 0.0f;
    }

    // cutoff^-8
    constexpr float k2 = inv_cutoff * inv_cutoff;
    constexpr float k4 = k2 * k2;
    constexpr float k8 = k4 * k4;

    // dij^7, dij^8
    float dij2 = dij * dij;
    float dij4 = dij2 * dij2;
    float dij7 = dij4 * dij2 * dij;
    float dij8 = dij4 * dij4;

    // arg = (dij/cutoff)^8 * pi/2
    float dij_k8 = dij8 * k8;
    float arg = pi_over_2 * dij_k8;

    // sin(arg), cos(arg)
    float sin_arg;
    float cos_arg;
    __sincosf(arg, &sin_arg, &cos_arg);

    // exponentiation
    float cos_arg2 = cos_arg * cos_arg;
    float cos_arg3 = cos_arg2 * cos_arg;

    // write d switch_fn d r
    constexpr float minus_12_pi_k8 = -12 * pi * k8;
    dsdr = minus_12_pi_k8 * dij7 * sin_arg * cos_arg2;

    // return switch_fn(dij)
    float sr = cos_arg3;
    return sr;
}

float __device__ __forceinline__ fast_erfc_and_deriv(float x, float &dedx) {
    // TODO: consider using fasterfc implementations listed in this thread:
    // https://forums.developer.nvidia.com/t/calling-all-juffas-whats-up-with-erfcf-nowadays/262973/4

    float exp_x2 = __expf(-x * x);
    // (ytz) 5th order gaussian polynomial approximation, we need the exp(-x^2) anyways for the chain rule
    // so we use last variant in https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    float t = __frcp_rn(1.0f + 0.3275911f * x);
    float erfc_x =
        (0.254829592f + (-0.284496736f + (1.421413741f + (-1.453152027f + 1.061405429f * t) * t) * t) * t) * t * exp_x2;
    constexpr float minus_two_over_sqrt_pi = -static_cast<float>(TWO_OVER_SQRT_PI);
    dedx = minus_two_over_sqrt_pi * exp_x2;
    return erfc_x;
}

float __device__ __forceinline__
real_es_factor(float real_beta, float dij, float inv_dij, float inv_d2ij, float &damping_factor) {
    float beta_dij = real_beta * dij;

    // f(dij) = erfc(beta * dij)
    // ebd = f(dij), debd = f'(dij)
    float debd;
    float ebd = fast_erfc_and_deriv(beta_dij, debd);
    debd = real_beta * debd;

    // g(dij) = switch_fn(dij)
    // sr = g(dij), dsdr = g'(dij)
    float dsdr;
    float sr = switch_fn_and_deriv(dij, dsdr);

    damping_factor = ebd * sr;
    float damping_factor_prime = (ebd * dsdr) + (debd * sr);
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
