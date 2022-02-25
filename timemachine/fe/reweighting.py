__all__ = [
    "one_sided_exp",
    "construct_endpoint_reweighting_estimator",
    "construct_mixture_reweighting_estimator",
    "interpret_as_mixture_potential",
]

from jax import numpy as np
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
    """Interpret samples from multiple states as if they originate from a *single* state given by this potential.

    Notes
    -----
    * This is not the only way to interpret samples from multiple distributions
        as if they come from a single mixture distribution.
        In ref [2], alternatives are systematically enumerated.
        Assuming f_k are exact, the current approach corresponds to N3 in ref [2].

        (This is the most expensive of the options in ref [2],
        since it requires evaluating all k energy functions on every sample,
        but it is also the lowest variance.)

    References
    ----------
    [1] [Shirts, 2017] https://arxiv.org/abs/1704.00891
        https://arxiv.org/abs/1704.00891
    [2] [Elvira+, 2019] Generalized multiple importance sampling
        https://arxiv.org/abs/1511.03095
    """
    n_states, n_samples = u_kn.shape
    N_k = np.array(N_k)
    assert f_k.shape == (n_states,)
    assert np.sum(N_k) == n_samples

    mixture_u_n = -logsumexp(f_k - u_kn.T, b=N_k, axis=1)

    assert mixture_u_n.shape == (n_samples,)

    return mixture_u_n


def construct_endpoint_reweighting_estimator(
    samples_0, samples_1, batched_u_0_fxn, batched_u_1_fxn, ref_params, ref_delta_f
):
    """assuming
    * endpoint samples (samples_0, samples_1)
    * precise estimate of free energy difference at initial params
        ref_delta_f ~= f(ref_params, 1) - f(ref_params, 0)

    construct an estimator of f(params, 1) - f(params, 0)
    """
    ref_u_0 = batched_u_0_fxn(samples_0, ref_params)
    ref_u_1 = batched_u_1_fxn(samples_1, ref_params)

    def endpoint_correction_0(params):
        """estimate f(ref, 0) -> f(params, 0) by reweighting"""
        delta_us = batched_u_0_fxn(samples_0, params) - ref_u_0
        return one_sided_exp(delta_us)

    def endpoint_correction_1(params):
        """estimate f(ref, 1) -> f(params, 1) by reweighting"""
        delta_us = batched_u_1_fxn(samples_1, params) - ref_u_1
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


def construct_mixture_reweighting_estimator(samples, log_weights, batched_u_0_fxn, batched_u_1_fxn):
    """assuming
    * samples from a distribution p_ref(x)
      that has good overlap with BOTH p_0(params)(x) and p_1(params)(x),

    construct an estimator for the free energy difference
    f_1(params) - f_0(params)
    """
    assert len(samples) == len(log_weights)

    def f_0(params):
        """estimate f(params, 0) - f(ref) by reweighting"""
        log_numerator = -batched_u_0_fxn(samples, params)
        log_importance_weights = log_numerator - log_weights
        return one_sided_exp(-log_importance_weights)

    def f_1(params):
        """estimate f(params, 1) - f(ref) by reweighting"""
        log_numerator = -batched_u_1_fxn(samples, params)
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
