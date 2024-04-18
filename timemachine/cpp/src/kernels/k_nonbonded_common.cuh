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
    RealType cutoff = 1.2;
    RealType pi = static_cast<RealType>(PI);
    RealType dij_k = dij / cutoff; // TODO: multiply by inv cutoff

    // exponentiation
    RealType dij_k2 = dij_k * dij_k;
    RealType dij_k4 = dij_k2 * dij_k2;
    RealType dij_k8 = dij_k4 * dij_k4;

    RealType cos_arg = cos(0.5 * (pi * dij_k8));

    // exponentiation
    RealType cos_arg3 = cos_arg * cos_arg * cos_arg;
    return cos_arg3;
}

template <typename RealType> RealType __device__ __forceinline__ d_switch_fn_dr(RealType dij) {
    RealType cutoff = 1.2;
    RealType pi = static_cast<RealType>(PI);

    // exponentiation
    RealType k2 = cutoff * cutoff;
    RealType k4 = k2 * k2;
    RealType k8 = k4 * k4;

    RealType dij2 = dij * dij;
    RealType dij4 = dij2 * dij2;
    RealType dij7 = dij4 * dij2 * dij;
    RealType dij8 = dij4 * dij4;

    RealType arg = pi * dij8 / (2 * k8);
    RealType cos_arg = cos(arg);
    RealType cos_arg2 = cos_arg * cos_arg;

    return -12 * pi * dij7 * sin(arg) * cos_arg2 / k8;
}

template <typename RealType> RealType __device__ __forceinline__ d_erfc_beta_r_dr(RealType beta, RealType dij) {
    RealType beta_dij = beta * dij;
    RealType exp_beta_dij_2 = exp(-beta_dij * beta_dij);
    return -static_cast<RealType>(TWO_OVER_SQRT_PI) * beta * exp_beta_dij_2;
}

template <typename RealType>
RealType __device__ __forceinline__
real_es_factor(RealType real_beta, RealType dij, RealType inv_dij, RealType inv_d2ij, RealType &damping_factor) {
    RealType beta_dij = real_beta * dij;
    damping_factor = erfc(beta_dij) * switch_fn(dij);
    RealType damping_factor_prime =
        (erfc(beta_dij) * d_switch_fn_dr(dij)) + (d_erfc_beta_r_dr(real_beta, dij) * switch_fn(dij));
    RealType d_es_dr = damping_factor_prime * inv_dij - damping_factor * inv_d2ij;
    return d_es_dr;
}

// // opaque approximations applied here, only valid when real_beta == 2.0
// float __device__ __forceinline__
// real_es_factor(float real_beta, float dij, float inv_dij, float inv_d2ij, float &damping_factor) {
//     float t = dij;
//     // assert real_beta = 2.0?

//     // clang-format off

//     // polynomial approximation to erfc(beta_dij) * switch_fn(dij) for dij between 0 and 1.2
//     //damping_factor = (1.000000003f + (-2.256767524f + (-0.038179357f + (3.387718845f + (-1.060439901f + (-3.360820701f + (3.158115609f + -0.825868456f * t) * t) * t) * t) * t) * t) * t);
//     damping_factor = (1.000000000f + (-2.256753880f + (0.018515233f + (2.657773313f + (2.377065883f + (-11.423294474f + (13.618150926f + (-8.464803786f + (2.941368615f + -0.464257418f * t) * t) * t) * t) * t) * t) * t) * t) * t);

//     // polynomial approximation to deriv of damping_factor, for dij between 0 and 1.2
//     //float damping_factor_prime = (-2.256758371f + (-0.000008247f + (8.992386707f + (1.572164882f + (-29.997988338f + (34.101203109f + (-14.371321036f + 1.912459462f * t) * t) * t) * t) * t) * t) * t);
//     float damping_factor_prime = (-2.256758331f + (0.000011562f + (9.075084876f + (-1.117918509f + (-8.749246274f + (-38.496124435f + (113.496561608f + (-119.796066138f + (59.561671589f + -11.763822412f * t) * t) * t) * t) * t) * t) * t) * t) * t);

//     // clang-format on

//     float d_es_dr = damping_factor_prime * inv_dij - damping_factor * inv_d2ij;

//     return d_es_dr;
// }

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
    // TODO: should I really be multiplying real_es_factor by inv_dij?
    //    or was a factor of inv_dij pulled out as an optimization, now being double-counted?
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
