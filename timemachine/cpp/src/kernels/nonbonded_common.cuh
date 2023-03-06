#pragma once

#include "k_fixed_point.cuh"

// each atom parameterized by a 4-tuple: charge, lj sigma, lj epsilon, 4D coordinate w
enum { PARAM_OFFSET_CHARGE = 0, PARAM_OFFSET_SIG, PARAM_OFFSET_EPS, PARAM_OFFSET_W, PARAMS_PER_ATOM };

typedef void (*k_nonbonded_fn)(
    const int N,
    const int NR,
    const double *__restrict__ coords,
    const double *__restrict__ params, // [N]
    const double *__restrict__ box,
    const double beta,
    const double cutoff,
    const unsigned int *__restrict__ row_idxs,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
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

/*
WIP: alternative to erfc reaction field

References
----------
[Kubincová, Riniker, Hünenberger, 2020] Reaction-field electrostatics in molecular dynamics simulations:
    development of a conservative scheme compatible with an atomic cutoff
    https://pubs.rsc.org/en/content/articlelanding/2020/CP/D0CP03835K

Implementations:
* As OpenMM custom forces in Reeds
    https://github.com/rinikerlab/reeds/blob/2584f7d049f622df6a3acd1cc4216b14e404f01e/reeds/openmm/reeds_openmm.py#L229-L390
* As a proposed potential in Jax MD
    https://github.com/jax-md/jax-md/pull/199/files#diff-a0a92fb308e4969b7d9019f447d3a3871db8dd09e07db12170605ade70e77312R565-R585
*/

// u_pair(q_ij, d_ij) = q_ij * rxn_field_u_factor(d_ij)
template <typename RealType> RealType __device__ __forceinline__ rxn_field_u_factor(RealType dij) {

    auto d2ij = dij * dij;
    auto d4ij = d2ij * d2ij;
    auto d6ij = d4ij * d2ij;

    // TODO: expose parameters (eps_rf=78.5, cutoff=1.2, mrf=4, nrf=6)
    // (derived from https://github.com/proteneer/timemachine/commit/cdaae0a3f4f855bd6740d51964d1987b32065034#diff-baf9ec3eb8fc3515ca70f986de186962ac433f97cbc6ba3dfc2601592ec97dd9R20-R33
    RealType krf = 0.28385782934833576;
    RealType arfm = 0.15451938291139242;
    RealType arfn = -0.07065358157814013;
    RealType crf = 1.3515295358649788;

    return (1 / dij) + (krf * d2ij) + (arfm * d4ij) + (arfn * d6ij) - crf;
}

// du_dr_pair(q_ij, d_ij) = q_ij * rxn_field_du_dr_factor(d_ij)
template <typename RealType>
RealType __device__ __forceinline__ rxn_field_du_dr_factor(RealType dij, RealType inv_d2ij) {

    auto d2ij = dij * dij;
    auto d3ij = d2ij * dij;
    auto d5ij = d3ij * d2ij;

    // TODO: expose parameters (eps_rf=78.5, cutoff=1.2, mrf=4, nrf=6)
    // (derived from https://github.com/proteneer/timemachine/commit/cdaae0a3f4f855bd6740d51964d1987b32065034#diff-baf9ec3eb8fc3515ca70f986de186962ac433f97cbc6ba3dfc2601592ec97dd9R20-R33
    RealType krf = 0.28385782934833576;
    RealType arfm = 0.15451938291139242;
    RealType arfn = -0.07065358157814013;

    return -inv_d2ij + (2 * krf * dij) + (4 * arfm * d3ij) + (6 * arfn * d5ij);
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
    RealType &ebd,
    RealType &es_prefactor,
    RealType &u) {
    inv_dij = rsqrt(d2ij);

    dij = d2ij * inv_dij;
    inv_d2ij = inv_dij * inv_dij;

    RealType qij = qi * qj;

    // TODO: template on rxn field enum?
    // es_prefactor = charge_scale * qij * inv_dij * real_es_factor(beta, dij, inv_d2ij, ebd);
    es_prefactor = charge_scale * qij * rxn_field_du_dr_factor<RealType>(dij, inv_d2ij);

    if (COMPUTE_U) {
        // u = charge_scale * qij * inv_dij * ebd;
        u = charge_scale * qij * rxn_field_u_factor<RealType>(dij);
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
