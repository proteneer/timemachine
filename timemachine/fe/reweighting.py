__all__ = [
    "one_sided_exp",
    "construct_endpoint_reweighting_estimator",
    "construct_mixture_reweighting_estimator",
    "interpret_as_mixture_potential",
    "construct_rw_uncertainty_estimate",
]

from typing import Any, Callable, Collection

import numpy as np
from jax import numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp

Samples = Collection
Params = Collection
Array = Any  # see https://github.com/google/jax/issues/943
Energies = Array

BatchedReducedPotentialFxn = Callable[[Samples, Params], Energies]


def one_sided_exp(delta_us: Array) -> float:
    """exponential averaging
    see https://github.com/choderalab/pymbar/blob/15f932a271343e611ed4be2d468c77b1d11cf01f/pymbar/exp.py#L54"""

    return -logsumexp(-delta_us - jnp.log(len(delta_us)))


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
        reduced potentials of all N samples evaluated in all K states (u_kn[k, n] = u_k(x_n))
    f_k : [K,] float array
        reduced free energies of all K states (up to an additive constant)
    N_k : [K,] int array
        number of samples from each individual state (sum(N_k) must equal N)

    Returns
    -------
    mixture_u_n : [N,] float array
        mixture_u_n[n] = u_mix(x_n), where

        u_mix(x) = -log(q_mix(x))
        q_mix(x) = \sum_k w_k (q_k(x) / Z_k)

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
    return -logsumexp(f_k - u_kn.T, b=N_k, axis=1)


def construct_rw_uncertainty_estimate(n_works, n_replicates=1000, seed=1234):
    """Freeze bootstrap indices to form a deterministic, differentiable estimate of reweighting uncertainty"""

    # Generate and freeze a large number of random bootstrap indices.
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(low=0, high=n_works, size=(n_replicates, n_works))

    def rw_uncertainty(delta_us):
        """deterministic, differentiable function of delta_us"""
        estimates = vmap(one_sided_exp)(delta_us[bootstrap_indices])
        return jnp.std(estimates)

    return rw_uncertainty


def construct_endpoint_reweighting_estimator(
    samples_0: Samples,
    samples_1: Samples,
    batched_u_0_fxn: BatchedReducedPotentialFxn,
    batched_u_1_fxn: BatchedReducedPotentialFxn,
    ref_params: Params,
    ref_delta_f: float,
    ref_delta_f_std: float = 0,
    n_bootstrap_replicates: int = 1000,
    seed: int = 1234,
) -> Callable[[Params], Array]:
    """assuming
    * endpoint samples (samples_0, samples_1)
    * precise estimate of free energy difference at initial params
        ref_delta_f ~= f(ref_params, 1) - f(ref_params, 0)

    construct an estimator of delta_f(params) = f(params, 1) - f(params, 0)

    (uses bootstrapping to generate many estimates for delta_f(params))

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
    ref_delta_f_std
        assumed noise level in estimate of ref_delta_f
    n_bootstrap_replicates : int
    seed : int
        used to generate frozen bootstrap indices

    Returns
    -------
    estimate_delta_f
        computes n_bootstrap_replicates estimates of f(params, 1) - f(params, 0) for arbitrary params

        notes:
        * estimate_delta_f(ref_params) ~ N(ref_delta_f, ref_delta_f_std)
        * estimate_delta_f(params) can become unreliable when
          params is very different from ref_params
    """
    ref_u_0 = batched_u_0_fxn(samples_0, ref_params)
    ref_u_1 = batched_u_1_fxn(samples_1, ref_params)
    n_0 = len(ref_u_0)
    n_1 = len(ref_u_1)

    # Generate and freeze a large number of random bootstrap indices.
    rng = np.random.default_rng(seed)
    bootstrap_indices_0 = rng.integers(low=0, high=n_0, size=(n_bootstrap_replicates, n_0))
    bootstrap_indices_1 = rng.integers(low=0, high=n_1, size=(n_bootstrap_replicates, n_1))
    ref_delta_f_smoothed_bootstrap = ref_delta_f + rng.normal(loc=0, scale=ref_delta_f_std, size=n_bootstrap_replicates)

    def estimate_delta_f_bootstrap(params):
        delta_us_0 = batched_u_0_fxn(samples_0, params) - ref_u_0
        delta_us_1 = batched_u_1_fxn(samples_1, params) - ref_u_1

        estimates_0 = vmap(one_sided_exp)(delta_us_0[bootstrap_indices_0])
        estimates_1 = vmap(one_sided_exp)(delta_us_1[bootstrap_indices_1])

        # TODO[decision] : return array of estimates, or just a (loc, scale) summary?
        delta_f_estimates = ref_delta_f_smoothed_bootstrap - estimates_0 + estimates_1  # (n_bootstrap_replicates)
        return delta_f_estimates

    return estimate_delta_f_bootstrap


def construct_mixture_reweighting_estimator(
    samples_n: Samples,
    u_ref_n: Array,
    batched_u_0_fxn: BatchedReducedPotentialFxn,
    batched_u_1_fxn: BatchedReducedPotentialFxn,
    n_bootstrap_replicates: int = 1000,
    seed: int = 1234,
) -> Callable[[Params], float]:
    r"""assuming
    * samples x_n from a distribution p_ref(x) \propto(exp(-u_ref(x))
      that has good overlap with BOTH p_0(params)(x) and p_1(params)(x),
    * evaluation (or estimates) of u_ref_n[n] = u_ref(x_n)

    construct an estimator for the free energy difference
    f_1(params) - f_0(params)

    Parameters
    ----------
    samples_n: [N,] collection
        samples[n] ~ p_ref
        p_ref(x) \propto exp(-u_ref(x))
    u_ref_n: [N,] array
        u_ref_n[n] = u_ref(samples[n])
    batched_u_0_fxn
        computes batch of endstate 0 energies at specified params
        [u_0(x, params) for x in samples_n]
    batched_u_1_fxn
        computes batch of endstate 1 energies at specified params
        [u_1(x, params) for x in samples_n]

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
    * Forming a single reference state as a mixture of several states (i.e. a constant denominator "u_ref_n")
        and differentiating the numerator ("-u(samples_n, params)") w.r.t. params
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
    assert len(samples_n) == len(u_ref_n)

    n = len(samples_n)

    # Generate and freeze a large number of random bootstrap indices.
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(low=0, high=n, size=(n_bootstrap_replicates, n))
    # TODO[thinking] : should the bootstrap indices be shared for the ref->0 ref->1 edges?

    def estimate_delta_f_bootstrap(params) -> float:
        u_0_n = batched_u_0_fxn(samples_n, params)
        delta_us_0 = u_0_n - u_ref_n
        f_0_estimates = vmap(one_sided_exp)(delta_us_0[bootstrap_indices])

        u_1_n = batched_u_1_fxn(samples_n, params)
        delta_us_1 = u_1_n - u_ref_n
        f_1_estimates = vmap(one_sided_exp)(delta_us_1[bootstrap_indices])

        # TODO[decision] : return array of estimates, or just a (loc, scale) summary?
        delta_f_estimates = f_1_estimates - f_0_estimates
        return delta_f_estimates

    return estimate_delta_f_bootstrap
