"""Offline protocol optimization

Setting
-------
Given simulation results from an initial alchemical protocol, we would like to construct a new protocol
whose intermediates are more "evenly spaced" according to something like "thermodynamic distance".

Approach
--------
* Define a "thermodynamic distance" between two values of lambda in terms of work standard deviation:
    "thermodynamic_distance(prev_lam, next_lam) = max(forward_work_stddev, reverse_work_stddev)"
* Compute a deterministic approximation of this distance by reweighting initial samples,
    and make this reweighting approximation extremely cheap by linearly interpolating previously computed energies.
* Build a protocol "left to right" by repeatedly using bisection to place the next lambda window at a fixed
    "thermodynamic distance" away from the previous window.

References
----------
1. The approach is directly inspired by "thermodynamic trailblazing," developed in Andrea Rizzi's thesis
    https://search.proquest.com/openview/0f0bda7dc135aad7216b6acecb815d3c/1.pdf?pq-origsite=gscholar&cbl=18750&diss=y
    and implemented in Yank
    https://github.com/choderalab/yank/blob/59fc6313b3b7d82966afc539604c36f4db9b952c/Yank/pipeline.py#L1983-L2648
    * Similarities:
        * Overall problem framing
        * Notion of thermodynamic distance in terms of work standard deviation
        * Sequential search strategy
    * Differences:
        * Offline rather than online
        * "Lighter weight"
            Requires no new simulations, requires no new calls to the potential energy function, and
            doesn't even require to look at the stored samples.

2. This implementation grew out of previous protocol optimization approaches considered in timemachine:
    * https://github.com/proteneer/timemachine/pull/442
        Variant of "trailblazing", greedily constructing a protocol where
        forward work_stddev(i -> i+1) is intended to be ~ constant for all i, and where
        approximate samples from each lambda window are generated on the fly using short simulations
        initialized with samples from the previous lambda window.
    * https://github.com/proteneer/timemachine/pull/437
        Gradient-based optimization of high-dimensional protocols, using a reweighting-based estimate of a
        a T.I.-tailored objective, stddev(du/dlambda).
"""

from typing import Callable, cast

import numpy as np
from jax import Array, jit
from jax import numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from scipy.optimize import bisect

Float = float
DistanceFxn = Callable[[Float, Float], Float]
WorkStddevEstimator = DistanceFxn


def rebalance_initial_protocol(
    lambdas_k: Array,
    f_k: Array,
    u_kn: Array,
    N_k: Array,
    work_stddev_threshold: Float = 1.0,
) -> Array:
    """Given simulation results from an initial protocol,
    return a new protocol satisfying the heuristic
        work_stddev(i -> i+1) <= work_stddev_threshold,
        work_stddev(i+1 -> i) <= work_stddev_threshold
        for all i

    Parameters
    ----------
    lambdas_k : monotonic sequence starting at 0 and ending at 1
    f_k : estimated reduced free energies of initial lambda windows
    u_kn : reduced potential energies of initial samples in all initial lambda windows
    N_k : number of samples from each initial lambda window
    work_stddev_threshold : controls spacing of new lambda windows (smaller threshold -> tighter spacing)

    Notes
    -----
    Applies the following approximations:
    * u(x_n, lam) for new trial values of lam can be well-approximated by linear interpolation of u_kn
    * work_stddev(prev_lam, next_lam) can be well-approximated by reweighting samples from initial protocol
    """
    # aggregate all samples from initial protocol
    reference_log_weights_n = log_weights_from_mixture(u_kn, f_k, N_k)

    # linearly interpolate initial energies
    vec_u_interp = linear_u_kn_interpolant(lambdas_k, u_kn)

    # function to estimate work_stddev(prev_lam, next_lam)
    work_stddev_estimator = construct_work_stddev_estimator(reference_log_weights_n, vec_u_interp)

    # function needed to place the next lambda window given the location of the previous window
    distance_fxn = construct_max_work_stddev_distance(work_stddev_estimator)

    # build a new protocol one state at a time
    optimized_protocol = greedily_optimize_protocol(distance_fxn, target_distance=work_stddev_threshold)
    return optimized_protocol


def log_weights_from_mixture(u_kn: ArrayLike, f_k: ArrayLike, N_k: ArrayLike) -> Array:
    r"""Assuming
    * K reduced potential energy functions u_k
    * N_k samples from each state e^{-u_k} / Z_k
    * reduced free energy estimates f_k ~= - log Z_k
    * u_kn[k, n] = u_k(xs[n)
        (where samples xs = concatenation of all samples, in any order,
        and len(xs) == N = sum(N_k))

    interpret the collection of N = \sum_k N_k samples as coming from a
    mixture of states p(x) = (1 / K) \sum_k e^-u_k / Z_k
    """
    f_k = jnp.asarray(f_k)
    u_kn = jnp.asarray(u_kn)

    log_q_k = f_k - u_kn.T
    N_k = np.array(N_k, dtype=np.float64)  # may be ints, or in a list...
    log_weights = logsumexp(log_q_k, b=N_k, axis=1)
    return log_weights


def linear_u_kn_interpolant(lambdas: Array, u_kn: Array) -> Callable:
    """Given a matrix u_kn[k, n] = u(xs[n], lambdas[k]) produce linear interpolated estimates of u(xs[n], lam)
    at arbitrary new values lam"""

    def u_interp(u_n: ArrayLike, lam: ArrayLike) -> Array:
        return jnp.nan_to_num(jnp.interp(lam, lambdas, u_n), nan=+jnp.inf, posinf=+jnp.inf)

    @jit
    def vec_u_interp(lam: Float) -> Array:
        return vmap(u_interp, (1, None))(u_kn, lam)

    return vec_u_interp


def construct_work_stddev_estimator(reference_log_weights_n: Array, vec_u: Callable) -> WorkStddevEstimator:
    """Construct reweighted estimator for stddev from a collection of reference samples"""

    def work_stddev_estimator(prev_lam: Float, next_lam: Float) -> Float:
        target_logpdf_n = -vec_u(prev_lam)
        delta_us = vec_u(next_lam) - vec_u(prev_lam)

        stddev_estimate = reweighted_stddev(
            f_n=delta_us,
            target_logpdf_n=target_logpdf_n,
            source_logpdf_n=reference_log_weights_n,
        )

        return stddev_estimate

    return work_stddev_estimator


def reweighted_stddev(f_n: Array, target_logpdf_n: Array, source_logpdf_n: Array) -> Float:
    """Compute reweighted estimate of
    stddev(f(x)) under x ~ p_target
    based on samples   x ~ p_source

    where
        p_target(x) = exp(target_logpdf(x)) / Z_target

    using samples from a different source
        x_n ~ p_source
        where
        p_source(x) = exp(source_logpdf(x)) / Z_source

    The inputs are arrays "{fxn_name}_n" containing the result of
    calling each fxn on a fixed array of samples:

    * f_n = [f(x_n) for x_n in samples]
    * target_logpdf_n = [target_logpdf(x_n) for x_n in samples]
    * source_logpdf_n = [source_logpdf(x_n) for x_n in samples]
    """

    log_weights_n = target_logpdf_n - source_logpdf_n
    weights = jnp.exp(log_weights_n - logsumexp(log_weights_n)).flatten()

    f_mean = jnp.sum(weights * f_n)
    squared_deviations = (f_n - f_mean) ** 2

    # sanitize 0 * inf -> 0 (instead of nan)
    weighted_squared_deviations = weights * squared_deviations
    sanitized = jnp.nan_to_num(weighted_squared_deviations, nan=0)
    stddev = jnp.sqrt(jnp.sum(sanitized))

    return cast(Float, stddev)


def construct_max_work_stddev_distance(work_stddev_estimator) -> DistanceFxn:
    """Construct a distance function `distance(prev_lam, trial_lam)` where bisection search
    on `f(trial_lam) = distance(prev_lam, trial_lam) - target_distance`
    can be used to select next_lam"""

    def max_work_stddev_distance(prev_lam, next_lam, max_step=0.25):
        """if (next_lam - prev_lam <= max_step), compute max(forward_stddev, reverse_stddev)"""
        too_far = next_lam - prev_lam > max_step
        if too_far:
            return +jnp.inf

        # compute max of forward, reverse work stddevs
        forward_stddev = work_stddev_estimator(prev_lam, next_lam)
        reverse_stddev = work_stddev_estimator(next_lam, prev_lam)

        return max(forward_stddev, reverse_stddev)

    return max_work_stddev_distance


def greedily_optimize_protocol(distance_fxn: DistanceFxn, target_distance=0.5, max_iterations=1000) -> Array:
    """Optimize a lambda protocol from "left to right"

    Sequentially pick next_lam so that
    distance_fxn(prev_lam, next_lam) ~= target_distance
    """
    protocol = [0.0]

    for t in range(max_iterations):
        prev_lam = protocol[-1]

        # can we directly jump to the end?
        if distance_fxn(prev_lam, 1.0) < target_distance:
            break

        # otherwise, binary search for next
        next_lam = bisect(
            f=lambda trial_lam: distance_fxn(prev_lam, trial_lam) - target_distance,
            a=prev_lam,
            b=1.0,
        )
        protocol.append(next_lam)

        if t == max_iterations - 1:
            print(UserWarning("Exceeded max_iterations!"))

    if protocol[-1] != 1.0:
        protocol.append(1.0)

    return jnp.array(protocol)
