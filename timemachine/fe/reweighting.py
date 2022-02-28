__all__ = [
    "one_sided_exp",
    "construct_endpoint_reweighting_estimator",
    "construct_mixture_reweighting_estimator",
    "interpret_as_mixture_potential",
]

from typing import Callable, Collection

from jax import numpy as np
from jax.scipy.special import logsumexp

Samples = Collection
Params = Collection
Array = np.ndarray
Energies = Array

BatchedReducedPotentialFxn = Callable[[Samples, Params], Energies]


def log_mean(log_values: Array) -> float:
    """stable log(mean(values))

    log(mean(values))
    = log(sum(values / len(values)))
    = logsumexp(log(values) - log(len(values))
    """
    return logsumexp(log_values - np.log(len(log_values)))


def estimate_log_z_ratio(log_importance_weights: Array) -> float:
    """stable log(mean(importance_weights))"""
    return log_mean(log_importance_weights)


def one_sided_exp(delta_us: Array) -> float:
    """exponential averaging

    References
    ----------
    [1] pymbar implementation
        https://github.com/choderalab/pymbar/blob/15f932a271343e611ed4be2d468c77b1d11cf01f/pymbar/exp.py#L54
    """
    # delta_us = -log_importance_weights
    # delta_f  = -log_z_ratio
    return -estimate_log_z_ratio(-delta_us)


def interpret_as_mixture_potential(u_kn: Array, f_k: Array, N_k: Array) -> Array:
    r"""Interpret samples from multiple states k as if they originate from a single state
    defined as a weighted mixture:

    $p_{mix}(x) \propto \sum_k w_k q_k(x) / Z_k$

    where
    * $q_k(x) = exp(-u_k(x))$ is the Boltzmann weight function for state k
    * $f_k = - log Z_k$ is the assumed normalization for state k
    * $w_k \propto N_k$ is the mixture weight of state k

    (up to a single constant)

    Parameters
    ----------
    u_kn : [K, N] float array
        reduced potentials of all N samples evaluated in all K states
        u_kn[k, n] = u_k(x_n)
    f_k : [K,] float array
        reduced free energies of all K states
        (up to an additive constant)
    N_k : [K,] int array
        number of samples from each individual state
        (sum(N_k) must equal N)

    Returns
    -------
    mixture_u_n : [N,] float array
        mixture_u_n[n] = u_mix(x_n)

        where u_mix(x) = -logsumexp([-u_k(x) + f_k[k] for k in range(K)])

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

    q_kn = -u_kn
    log_Z_k = -f_k[:, np.newaxis]
    normalized_q_kn = q_kn - log_Z_k
    w_k = N_k / np.sum(N_k)

    mixture_q_n = logsumexp(normalized_q_kn, b=w_k, axis=0)
    mixture_u_n = -mixture_q_n

    assert mixture_u_n.shape == (n_samples,)

    return mixture_u_n


def construct_endpoint_reweighting_estimator(
    samples_0: Samples,
    samples_1: Samples,
    batched_u_0_fxn: BatchedReducedPotentialFxn,
    batched_u_1_fxn: BatchedReducedPotentialFxn,
    ref_params: Params,
    ref_delta_f: float,
) -> Callable[[Params], float]:
    """assuming
    * endpoint samples (samples_0, samples_1)
    * precise estimate of free energy difference at initial params
        ref_delta_f ~= f(ref_params, 1) - f(ref_params, 0)

    construct an estimator of f(params, 1) - f(params, 0)

    Parameters
    ----------
    samples_0: [N_0,] collection
        samples from endstate 0
    samples_1: [N_1,] collection
        samples from endstate 1
    batched_u_0_fxn
        function that computes batch of endstate 0 energies at specified params
        [u_0(x, params) for x in samples_0]
    batched_u_1_fxn
        function that computes batch of endstate 1 energies at specified params
        [u_1(x, params) for x in samples_1]
    ref_params
        assume samples_0 ~ exp(-u_0(., ref_params)) and
               samples_1 ~ exp(-u_1(., ref_params))
    ref_delta_f
        free energy difference between endstates 0, 1 at ref_params
        ref_delta_f ~= f(ref_params, 1) - f(ref_params, 0)

    Returns
    -------
    estimate_delta_f
        computes an estimate f(params, 1) - f(params, 0) for arbitrary params

        notes:
        * estimate_delta_f(ref_params) == ref_delta_f
        * estimate_delta_f(params) can become unreliable when
          params is very different from ref_params
    """
    ref_u_0 = batched_u_0_fxn(samples_0, ref_params)
    ref_u_1 = batched_u_1_fxn(samples_1, ref_params)

    def endpoint_correction_0(params) -> float:
        """estimate f(ref, 0) -> f(params, 0) by reweighting"""
        delta_us = batched_u_0_fxn(samples_0, params) - ref_u_0
        return one_sided_exp(delta_us)

    def endpoint_correction_1(params) -> float:
        """estimate f(ref, 1) -> f(params, 1) by reweighting"""
        delta_us = batched_u_1_fxn(samples_1, params) - ref_u_1
        return one_sided_exp(delta_us)

    def estimate_delta_f(params: Params) -> float:
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


def construct_mixture_reweighting_estimator(
    samples: Samples,
    log_weights: Array,
    batched_u_0_fxn: BatchedReducedPotentialFxn,
    batched_u_1_fxn: BatchedReducedPotentialFxn,
) -> Callable[[Params], float]:
    r"""assuming
    * samples from a distribution p_ref(x) \propto(exp(-u_ref(x))
      that has good overlap with BOTH p_0(params)(x) and p_1(params)(x),
    * evaluation (or estimates) of log_weight(x) = -u_ref(x)

    construct an estimator for the free energy difference
    f_1(params) - f_0(params)

    Parameters
    ----------
    samples: [N,] collection
        x ~ p_ref(x) \propto exp(-u_ref(x))
    log_weights: [N,] array
        log_weights[n] = -u_ref(samples[n])
    batched_u_0_fxn
        computes batch of endstate 0 energies at specified params
        [u_0(x, params) for x in samples]
    batched_u_1_fxn
        computes batch of endstate 1 energies at specified params
        [u_1(x, params) for x in samples]

    Returns
    -------
    estimate_delta_f
        computes an estimate f(params, 1) - f(params, 0) for arbitrary params

        notes:
        * estimate_delta_f(ref_params) == ref_delta_f
        * estimate_delta_f(params) can become unreliable when
          params is very different from ref_params

    Notes
    -----
    * This is essentially `computePerturbedFreeEnergies` in pymbar [1], but written in a slightly more generic way.
        (Allows the samples to come from sources other than an MBAR mixture, produces a function that can be
        differentiated w.r.t. params if batched_u_0_fxn, batched_u_0_fxn are differentiable w.r.t. params.)
    * Reweighting from a single reference state is used in works like ref [2] in the context of force field fitting
    * Forming a single reference state as a mixture of several states (i.e. a constant denominator "log_weights")
        and differentiating the numerator ("-u(samples, params)") wr.t. params
        is used in works like ref [3] to differentiate free energy estimates w.r.t. params.
    * Non-requirement: u_ref does not have to be of the same functional form as u_0, u_1

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

    def f_1(params) -> float:
        """estimate f(params, 1) - f(ref) by reweighting"""
        log_numerator = -batched_u_1_fxn(samples, params)
        log_importance_weights = log_numerator - log_weights
        return one_sided_exp(-log_importance_weights)

    def estimate_delta_f(params) -> float:
        r"""estimate f(params, 1) - f(params, 0)

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
