# adapted from https://github.com/proteneer/timemachine/blob/7d6099b0f5b4a2d0b26c3edc7a91c18f7a526c00/md/experimental/smc.py

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from jax import numpy as jnp
from jax.scipy.special import logsumexp as jlogsumexp
from numpy.typing import NDArray
from scipy.optimize import root_scalar
from scipy.special import logsumexp

# type annotations
Sample = Any
Samples = List[Sample]  # e.g. List[CoordsVelBox]

Lambda = float
LogWeight = float
Array = NDArray
IndexArray = Array
LogWeights = Array
First = bool

BatchPropagator = Callable[[Samples, Lambda], Samples]
BatchLogProb = Callable[[Samples, Lambda], LogWeights]
BatchLogProbASMC = Callable[[Samples, Lambda, First], LogWeights]

Resampler = Callable[[LogWeights], Tuple[IndexArray, LogWeights]]
ResultDict = Dict[str, Any]


def sequential_monte_carlo(
    samples: Samples,
    lambdas: Array,
    propagate: BatchPropagator,
    log_prob: BatchLogProb,
    resample: Resampler,
) -> ResultDict:
    """barebones implementation of Sequential Monte Carlo (SMC)

    Parameters
    ----------
    samples: [N,] list
    lambdas: [K,] array
    propagate: function
        [move(x, lam) for x in xs]
        for example, move(x, lam) might mean "run 100 steps of all-atom MD targeting exp(-u(., lam)), initialized at x"
    log_prob: function
        [exp(-u(x, lam)) for x in xs]
    resample: function
        (optionally) perform resampling given an array of log weights

    Returns
    -------
    trajs_dict
        "sample_traj"
            [K-1, N] list of snapshots
        "incremental_log_weights_traj"
            [K-1, N] array of incremental log weights
        "ancestry_traj"
            [K-1, N] array of ancestor idxs
        "log_weights_traj"
            [K, N] array of accumulated log weights

    References
    ----------
    * Arnaud Doucet's annotated bibliography of SMC
        https://www.stats.ox.ac.uk/~doucet/smc_resources.html
    * [Dai, Heng, Jacob, Whiteley, 2020] An invitation to sequential Monte Carlo samplers
        https://arxiv.org/abs/2007.11936

    See Also
    --------
    * get_endstate_samples_from_smc_result
    """
    n = len(samples)
    log_weights = np.zeros(n)

    # store
    sample_traj = [samples]
    ancestry_traj = [np.arange(n)]
    log_weights_traj = [np.array(log_weights)]
    incremental_log_weights_traj = []  # note: redundant but convenient

    def accumulate_results(samples, indices, log_weights, incremental_log_weights):
        sample_traj.append(samples)
        ancestry_traj.append(indices)
        log_weights_traj.append(np.array(log_weights))
        incremental_log_weights_traj.append(np.array(incremental_log_weights))

    # Note: loop bounds [1:-1] intentional:
    #   these steps are the only ones that contribute to the SMC result,
    #   and some care is needed to extract endstate samples at t=0, t=T-1
    #   See also
    #   * discussion at https://github.com/proteneer/timemachine/pull/718#discussion_r854276326
    #   * helper function get_endstate_samples_from_smc_result
    for lam_initial, lam_target in zip(lambdas[:-2], lambdas[1:-1]):
        # update log weights
        incremental_log_weights = log_prob(sample_traj[-1], lam_target) - log_prob(sample_traj[-1], lam_initial)
        log_weights += incremental_log_weights

        # resample
        indices, log_weights = resample(log_weights)
        resampled = [sample_traj[-1][i] for i in indices]

        # propagate
        samples = propagate(resampled, lam_target)

        # log
        accumulate_results(samples, indices, log_weights, incremental_log_weights)

    # final result: a collection of samples, with associated log weights
    incremental_log_weights = log_prob(samples, lambdas[-1]) - log_prob(samples, lambdas[-2])
    incremental_log_weights_traj.append(incremental_log_weights)
    log_weights_traj.append(np.array(log_weights + incremental_log_weights))

    # cast everything (except samples list) to arrays
    trajs_dict = dict(
        traj=sample_traj,
        log_weights_traj=np.array(log_weights_traj),
        ancestry_traj=np.array(ancestry_traj),
        incremental_log_weights_traj=np.array(incremental_log_weights_traj),
    )
    return trajs_dict


class ASMCMaxIterError(Exception):
    """
    Exception when ASMC exceeds the maximum number of iters.
    """

    pass


def adaptive_sequential_monte_carlo(
    samples: Samples,
    propagate: BatchPropagator,
    log_prob: BatchLogProbASMC,
    resample: Resampler,
    cess_target: float,
    epsilon=1e-2,
    store_intermediate_traj=True,
    max_iterations=100,
) -> ResultDict:
    """Implementation of Adaptive Sequential Monte Carlo (SMC).
       This will adaptively interpolate between lambda=0 and lambda=1,
       starting at lambda=0.

    Parameters
    ----------
    samples: [N,] list
    propagate: function
        [move(x, lam) for x in xs]
        for example, move(x, lam) might mean "run 100 steps of all-atom MD targeting exp(-u(., lam)), initialized at x"
    log_prob: function
        [exp(-u(x, lam, first: bool)) for x in xs]
        first is set to True for the first iteration of each binary search.
        This may be used to improve performance by caching prefactors.
    resample: function
        (optionally) perform resampling given an array of log weights
    cess_target: float
        Target CESS (see `conditional_effective_sample_size`). Intermediate lambdas
        will be sampled keeping the CESS between successive windows at approximately
        this value. This value should be in the range (1, N).
    epsilon:
        Used to determine the precision of the adaptive binary search.
    store_intermediate_traj:
        Set to True (default) to store intermediate trajectories.
    max_iterations:
        Set to the maximum number of iterations. If exceeded, will throw an
        `ASMCMaxIterError` exception.
    Returns
    -------
    trajs_dict
        "traj"
            [K-1, N] list of snapshots only if `store_intermediate_traj` = True.
            [1, N] list of snapshots if `store_intermediate_traj` = False.
        "incremental_log_weights_traj"
            [K-1, N] array of incremental log weights
        "ancestry_traj"
            [K-1, N] array of ancestor idxs
        "log_weights_traj"
            [K, N] array of accumulated log weights
        "lambdas_traj"
            [K] array of adaptive lambdas

    References
    ----------
    * [Zhou, Johansen, Aston, 2016]
        Towards Automatic Model Comparison: An Adaptive Sequential Monte Carlo Approach
        https://arxiv.org/pdf/1303.3123 (Algorithm #4)
    """
    n = len(samples)

    # check cess_target
    assert cess_target > 1, f"cess_target is too small: {cess_target} <= 1"
    assert cess_target < n, f"cess_target is too large: {cess_target} >= {n}"

    log_weights = np.zeros(n)
    norm_log_weights = log_weights - logsumexp(log_weights)

    # store
    sample_traj = [samples]
    ancestry_traj = [np.arange(n)]
    log_weights_traj = [np.array(log_weights)]
    incremental_log_weights_traj = []  # note: redundant but convenient
    lambdas_traj = [0.0]

    def accumulate_results(samples, indices, log_weights, incremental_log_weights, lam_target):
        if store_intermediate_traj:
            sample_traj.append(samples)
        else:
            # only store one intermediate set of samples to reduce memory usage
            sample_traj[0] = samples
        ancestry_traj.append(indices)
        log_weights_traj.append(np.array(log_weights))
        incremental_log_weights_traj.append(np.array(incremental_log_weights))
        lambdas_traj.append(lam_target)

    lam_initial = 0.0
    lam_target = 1.0  # adapted

    # Main ASMC loop
    for _ in range(max_iterations):
        # binary search for lambda that gives cess ~= cess_target
        cur_log_prob = log_prob(sample_traj[-1], lam_initial, True)

        # Used to pass incremental_log_weights out of the closure
        incremental_log_weights_closure = [None]

        def f_opt(lam: float) -> float:
            incremental_log_weights_closure[0] = log_prob(sample_traj[-1], lam, False) - cur_log_prob
            cess = conditional_effective_sample_size(norm_log_weights, incremental_log_weights_closure[0])
            return cess - cess_target

        try:
            lam_target = root_scalar(f_opt, bracket=(lam_initial, lam_target), method="bisect", xtol=epsilon).root
        except ValueError:
            # no root, just run at the final lambda
            pass

        incremental_log_weights = incremental_log_weights_closure[0]
        assert incremental_log_weights is not None

        # Stop when lam_target == 1.0
        #   See
        #   * discussion at https://github.com/proteneer/timemachine/pull/718#discussion_r854276326
        #   * helper function get_endstate_samples_from_smc_result
        if lam_target == 1.0:
            break

        # resample
        indices, log_weights = resample(log_weights + incremental_log_weights)
        norm_log_weights = log_weights - logsumexp(log_weights)
        resampled = [sample_traj[-1][i] for i in indices]

        # propagate
        samples = propagate(resampled, lam_target)

        # log
        accumulate_results(samples, indices, log_weights, incremental_log_weights, lam_target)

        # update target
        lam_initial = lam_target
        lam_target = 1.0
    else:
        raise ASMCMaxIterError(f"ASMC exceeded maximum number of iterations {max_iterations}.")

    # final result: a collection of samples, with associated log weights
    incremental_log_weights_traj.append(incremental_log_weights)
    log_weights_traj.append(np.array(log_weights + incremental_log_weights))
    lambdas_traj.append(lam_target)

    # cast everything (except samples list) to arrays
    trajs_dict = dict(
        traj=sample_traj,
        log_weights_traj=np.array(log_weights_traj),
        ancestry_traj=np.array(ancestry_traj),
        incremental_log_weights_traj=np.array(incremental_log_weights_traj),
        lambdas_traj=np.array(lambdas_traj),
    )

    return trajs_dict


def identity_resample(log_weights):
    """No interaction"""
    indices = np.arange(len(log_weights))
    return indices, log_weights


def multinomial_resample(log_weights):
    """sample proportional to exp(log_weights), with replacement"""
    normed_weights = np.exp(log_weights - logsumexp(log_weights))
    assert np.isclose(np.sum(normed_weights), 1.0)
    n = len(log_weights)
    indices = np.random.choice(np.arange(n), size=n, p=normed_weights)

    # new weights
    avg_log_weights = logsumexp(log_weights - np.log(n)) * np.ones(n)

    return indices, avg_log_weights


def stratified_resample(log_weights):
    """
    Split cummulative sum of weights into N interval
    and pick one particle per subinterval.

    References
    ----------
    * [Douc, Cappé, Moulines, 2005]
        Comparison of Resampling Schemes for Particle Filtering
        https://arxiv.org/abs/cs/0507025

    * Code based on:
        https://github.com/rlabbe/filterpy/blob/6cc052bea5a806d3531b59c820530c42b67a910c/filterpy/monte_carlo/resampling.py
        [MIT license]
    """
    weights = np.exp(log_weights - logsumexp(log_weights))
    assert np.isclose(np.sum(weights), 1.0)
    n = len(log_weights)

    # generate n subintervals that are on average 1/n apart
    subintervals = (np.random.random(n) + np.arange(n)) / n

    # init
    indices = np.zeros(n, dtype=int)
    cumulative_sum = np.cumsum(weights)

    i, j = 0, 0
    while i < n:
        if subintervals[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1

    # new weights
    avg_log_weights = logsumexp(log_weights - np.log(n)) * np.ones(n)

    return indices, avg_log_weights


def effective_sample_size(log_weights):
    r"""Effective sample size, taking values in interval [1, len(log_weights)]

    Notes
    -----
    * This uses the conventional definition ESS(w) = 1 / \sum_i w_i^2, which has some important known limitations!
    * See [Elvira, Martino, Robert, 2018] "Rethinking the effective sample size" https://arxiv.org/abs/1809.04129
        and references therein for some insightful discussion of limitations and possible improvements
    """
    norm_weights = jnp.exp(log_weights - jlogsumexp(log_weights))
    return 1 / jnp.sum(norm_weights**2)


def conditional_effective_sample_size(norm_log_weights, incremental_log_weights):
    r"""
    Conditional Effective sample size, taking values in interval [1, len(log_weights)]
    This is a different approximation for the effective sample size, which
    empirically shows better results when used with Adaptive SMC if resampling is not
    done every step. If resampling is done every step, this is the same as `effective_sample_size`.
    See the paper for more details.

    Parameters
    ----------
    norm_log_weights: [N]

    Notes
    -----
    * This uses the definition from [Zhou, Johansen, Aston, 2016]
      "Towards Automatic Model Comparison: An Adaptive Sequential Monte Carlo Approach"
      https://arxiv.org/pdf/1303.3123 (eq 3.16)
    * This is equal to the ESS if resampling is performed each iteration (i.e. not using conditional resampling)
    """
    n = len(norm_log_weights)
    summed_weights = norm_log_weights + incremental_log_weights
    num = 2 * jlogsumexp(summed_weights)
    denom = jlogsumexp(summed_weights + incremental_log_weights)
    return n * jnp.exp(num - denom)


def conditional_multinomial_resample(log_weights, thresh=0.5):
    """if fractional_effective_sample_size(log_weights) < thresh, then multinomial_resample"""
    n = len(log_weights)
    fractional_ess = effective_sample_size(log_weights) / n
    if fractional_ess < thresh:
        return multinomial_resample(log_weights)
    else:
        return identity_resample(log_weights)


def refine_samples(samples: Samples, log_weights: LogWeights, propagate: BatchPropagator, lam: float) -> Samples:
    """resample according to log_weights, then propagate at lam for a little bit"""

    # weighted samples -> equally weighted samples
    # TODO: replace multinomial resampling with something less wasteful, like stratified or systematic resampling
    resample = multinomial_resample  # but not: identity_resample or conditional_multinomial_resample
    resampled_inds, log_weights = resample(log_weights)
    assert np.isclose(np.std(log_weights), 0), "Need equally weighted samples"

    # diversify
    updated_samples = propagate([samples[i] for i in resampled_inds], lam)
    return updated_samples


def get_endstate_samples_from_smc_result(
    smc_result: ResultDict, propagate: BatchPropagator, lambdas: Array
) -> Tuple[Samples, Samples]:
    """unweighted approximate samples from lambdas[0] and lambdas[-1]

    TODO: include lambdas array in smc_result dict? Some other way to match up {lambdas[0], lambdas[-1]} to {0.0, 1.0}?
    """
    initial_samples = refine_samples(smc_result["traj"][0], smc_result["log_weights_traj"][0], propagate, lambdas[0])
    final_samples = refine_samples(smc_result["traj"][-1], smc_result["log_weights_traj"][-1], propagate, lambdas[-1])

    return initial_samples, final_samples
