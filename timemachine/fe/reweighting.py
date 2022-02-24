__all__ = [
    "one_sided_exp",
    "construct_endpoint_reweighting_estimator",
    "construct_mixture_reweighting_estimator",
    "interpret_as_mixture_potential",
    "reweight_from_mixture",
]

from jax import numpy as np
from jax import vmap
from jax.scipy.special import logsumexp


def log_mean(log_values):
    # log(mean(values))
    # = log(sum(values) / len(values))
    return logsumexp(log_values) - np.log(len(log_values))


def estimate_log_z_ratio(log_importance_weights):
    # log(mean(importance_weights))
    return log_mean(log_importance_weights)


def one_sided_exp(delta_us):
    # delta_us = -log_importance_weights
    # delta_f  = -log_z_ratio
    return -estimate_log_z_ratio(-delta_us)


def interpret_as_mixture_potential(u_kn, f_k, N_k):
    """https://arxiv.org/abs/1704.00891"""
    n_states, n_samples = u_kn.shape
    assert f_k.shape == (n_states,)
    assert np.sum(N_k) == n_samples

    return -logsumexp(f_k - u_kn.T, b=N_k, axis=1)


def reweight_from_mixture(u_kn, f_k, N_k):
    """https://arxiv.org/abs/1704.00891"""
    mixture_u_n = interpret_as_mixture_potential(u_kn, f_k, N_k)
    delta_u_kn = u_kn - mixture_u_n[np.newaxis, :]
    estimated_f_k = vmap(one_sided_exp)(delta_u_kn)
    return estimated_f_k


def construct_endpoint_reweighting_estimator(samples_0, samples_1, vec_u_0_fxn, vec_u_1_fxn, ref_params, ref_delta_f):
    """assuming
    * endpoint samples (samples_0, samples_1)
    * precise estimate of free energy difference at initial params
        ref_delta_f ~= f(ref_params, 1) - f(ref_params, 0)

    construct an estimator of f(params, 1) - f(params, 0)
    """
    ref_u_0 = vec_u_0_fxn(samples_0, params=ref_params)
    ref_u_1 = vec_u_1_fxn(samples_1, params=ref_params)

    def endpoint_correction_0(params):
        """estimate f(ref, 0) -> f(params, 0) by reweighting"""
        delta_us = vec_u_0_fxn(samples_0, params) - ref_u_0
        return one_sided_exp(delta_us)

    def endpoint_correction_1(params):
        """estimate f(ref, 1) -> f(params, 1) by reweighting"""
        delta_us = vec_u_1_fxn(samples_1, params) - ref_u_1
        return one_sided_exp(delta_us)

    def estimate_delta_f(params):
        """estimate f(params, 1) - f(params, 0)

        using this thermodynamic cycle:

        f(params, 0)  --->  f(params, 1)

             ^                   ^
             |                   |
             |                   |
             |                   |

        f(ref, 0)     --->  f(ref, 1)


        where
        * "f(ref, 0) -> f(ref, 1)" is assumed precomputed (using any precise free energy method)
        * "f(ref, 0) -> f(params, 0)" is estimated by reweighting
        * "f(ref, 1) -> f(params, 1)" is estimated by reweighting
        """
        return ref_delta_f - endpoint_correction_0(params) + endpoint_correction_1(params)

    return estimate_delta_f


def construct_mixture_reweighting_estimator(samples, log_weights, vec_u_0_fxn, vec_u_1_fxn):
    """assuming
    * samples from a distribution p_ref(x)
      that has good overlap with BOTH p_0(params)(x) and p_1(params)(x),

    construct an estimator for the free energy difference
    f_1(params) - f_0(params)
    """
    assert len(samples) == len(log_weights)

    def f_0(params):
        """estimate f(params, 0) - f(ref) by reweighting"""
        log_numerator = -vec_u_0_fxn(samples, params)
        log_importance_weights = log_numerator - log_weights
        return one_sided_exp(-log_importance_weights)

    def f_1(params):
        """estimate f(params, 1) - f(ref) by reweighting"""
        log_numerator = -vec_u_1_fxn(samples, params)
        log_importance_weights = log_numerator - log_weights
        return one_sided_exp(-log_importance_weights)

    def estimate_delta_f(params):
        r"""estimate f(params, 1) - f(params, 1)

        using this thermodynamic cycle:

        f(params, 0)  --->  f(params, 1)

                /\         /\
                 \         /
                  \       /
                   \     /

                   f(ref)
        where
        * "f(params, 0) - f(ref)" is estimated by reweighting
        * "f(params, 1) - f(ref)" is estimated by reweighting"""

        return f_1(params) - f_0(params)

    return estimate_delta_f
