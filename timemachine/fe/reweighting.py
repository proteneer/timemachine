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
    * This is nicely illustrated in ref [1] as a way to interpret MBAR [2]
    * Depends on the accuracy of the input free energy estimates (f_k - f_k[0]) for the source states.
    * This is not the only way to interpret samples from multiple distributions
        as if they come from a single mixture distribution.
        In ref [3], alternatives are systematically enumerated.
        Assuming f_k are exact, the current approach corresponds to N3 in ref [3].
        (N3 the most computationally expensive of the options in ref [3],
        since it requires evaluating all k energy functions on every sample,
        but it is also the lowest variance.)

    References
    ----------
    [1] [Shirts, 2017] Reweighting from the mixture distribution as a better way to describe
        the Multistate Bennett Acceptance Ratio
        https://arxiv.org/abs/1704.00891
    [2] [Shirts, Chodera, 2008] Statistically optimal analysis of samples from multiple equilibrium states.
        J. Chem. Phys. 129:124105, 2008.
        http://dx.doi.org/10.1063/1.2978177
    [3] [Elvira+, 2019] Generalized multiple importance sampling
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

    Notes
    -----
    * This is essentially `computePerturbedFreeEnergies` in pymbar [1], but written in a slightly more generic way.
        (Allows the samples to come from sources other than an MBAR mixture, produces a function that can be
        differentiated w.r.t. params if batched_u_0_fxn, batched_u_0_fxn are differentiable w.r.t. params.)
    * Reweighting from a single reference state is used in works like ref [2] in the context of force field fitting
    * Forming a single reference state as a mixture of several states (i.e. a constant denominator "log_weights")
        and differentiating the numerator ("-u(samples, params)") wr.t. params
        is used in works like ref [3] to differentiate free energy estimates w.r.t. params.

    References
    ----------
    [1] pymbar implementation of computePerturbedFreeEnergies
        https://github.com/choderalab/pymbar/blob/3c4262c490261110a7595eec37df3e2b8caeab37/pymbar/mbar.py#L1163-L1237
    [2] Messerly RA, Razavi SM, and Shirts MR. Configuration-Sampling-Based Surrogate Models for Rapid
        parameterization of Non-Bonded Interactions.
        J. Chem. Theory Comput. 2018, 14, 6, 3144–3162 https://doi.org/10.1021/acs.jctc.8b00223
    [3] Wieder et al. PyTorch implementation of differentiable reweighting in neutromeratio
        https://github.com/choderalab/neutromeratio/blob/2abf29f03e5175a988503b5d6ceeee8ce5bfd4ad/neutromeratio/parameter_gradients.py#L246-L267
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
