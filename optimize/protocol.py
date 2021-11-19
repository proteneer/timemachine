"""Protocol optimization"""

from jax import jit, vmap, numpy as np
from jax.scipy.special import logsumexp
from scipy.optimize import bisect

from functools import partial
from typing import Tuple, Callable

Float = float
Array = np.array


def linear_u_kn_interpolant(lambdas: Array, u_kn: Array) -> Tuple[Callable, Callable, Callable]:
    """Given a matrix u_kn[k, n] = u(xs[n], lambdas[k]) produce linear interpolated estimates of u(xs[n], lam)
    at arbitrary new values lam"""

    def u_interp(u_n: Array, lam: Float) -> Float:
        return np.nan_to_num(np.interp(lam, lambdas, u_n), nan=+np.inf, posinf=+np.inf)

    @jit
    def vec_u_interp(lam: Float) -> Array:
        return vmap(u_interp, (1, None))(u_kn, lam)

    @jit
    def vec_delta_u(from_lam: Float, to_lam: Float) -> Array:
        """+inf minus +inf -> 0, rather than +inf minus +inf -> nan"""
        raw_delta_u = vec_u_interp(to_lam) - vec_u_interp(from_lam)
        return np.nan_to_num(raw_delta_u)

    return u_interp, vec_u_interp, vec_delta_u


def log_weights_from_mixture(u_kn: Array, f_k: Array, N_k: Array) -> Array:
    """Assuming
    * K energy functions u_k
    * N_k samples from each state e^{-u_k} / Z_k
    * free energy estimates f_k ~= - log Z_k
    * u_kn[k, n] = u_k(xs[n)
        (where samples xs = concatenation of all samples, in any order,
        and len(xs) == N = sum(N_k))

    interpret the collection of N = \sum_k N_k samples as coming from a
    mixture of states p(x) = (1 / K) \sum_k e^-u_k / Z_k
    """
    log_q_k = f_k - u_kn.T
    N_k = np.array(N_k, dtype=np.float64)  # may be ints, or in a list...
    log_weights = logsumexp(log_q_k, b=N_k, axis=1)
    return log_weights


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
    weights = np.exp(log_weights_n - logsumexp(log_weights_n)).flatten()

    f_mean = np.sum(weights * f_n)
    squared_deviations = (f_n - f_mean) ** 2

    # sanitize 0 * inf -> 0 (instead of nan)
    weighted_squared_deviations = weights * squared_deviations
    sanitized = np.nan_to_num(weighted_squared_deviations, nan=0)
    stddev = np.sqrt(np.sum(sanitized))

    return stddev


WorkStddevEstimator = StepAssessor = Callable[[Float, Float], Float]


def construct_work_stddev_estimator(
        reference_log_weights_n: Array,
        vec_u: Callable,
        vec_delta_u: Callable) -> WorkStddevEstimator:
    """Construct reweighted estimator for stddev from a collection of reference samples"""

    def work_stddev_estimator(prev_lam: Float, next_lam: Float) -> Float:
        target_logpdf_n = - vec_u(prev_lam)
        delta_us = vec_delta_u(prev_lam, next_lam)

        stddev_estimate = reweighted_stddev(
            f_n=delta_us,
            target_logpdf_n=target_logpdf_n,
            source_logpdf_n=reference_log_weights_n,
        )

        return stddev_estimate

    return work_stddev_estimator


def construct_heuristic_lambda_pair_assessor(work_stddev_estimator) -> StepAssessor:
    """Construct a function f(prev_lam, trial_next_lam) where bisection search on second argument
    can be used to select next_lam so that p(x|next_lam) is a specified "distance" from p(x|prev_lam)"""

    def assess_lambda_pair(prev_lam, next_lam, desired_stddev=1.0, max_step=0.25):
        """if (next_lam - prev_lam <= max_step), compute (max(forward_stddev, reverse_stddev) - desired_stddev)"""
        too_far = next_lam - prev_lam > max_step
        if too_far:
            return + np.inf

        # compute max of forward, reverse work stddevs
        forward_stddev = work_stddev_estimator(prev_lam, next_lam)
        reverse_stddev = work_stddev_estimator(next_lam, prev_lam)
        higher_stddev = max(forward_stddev, reverse_stddev)

        return higher_stddev - desired_stddev

    return assess_lambda_pair


def greedily_optimize_protocol(assess_lambda_pair: StepAssessor, max_iterations=1000) -> Array:
    """Optimize a lambda protocol from "left to right"

    Sequentially pick next_lam so that
    assess_lambda_pair(prev_lam, next_lam) ~= 0

    assess_lambda_pair(prev_lam, next_lam) might compute
    some measure of distance between p(x | prev_lam) and p(x | next_lam)
    and compare that distance to a desired spacing `distance - target_distance`
        (returning a negative number if too close, a positive number if too far)
    """
    protocol = [0.0]

    for t in range(max_iterations):
        prev_lam = protocol[-1]

        # can we directly jump to the end?
        if assess_lambda_pair(prev_lam, 1.0) < 0:
            break

        # otherwise, binary search for next
        next_lam = bisect(
            f=lambda trial_lam: assess_lambda_pair(prev_lam, trial_lam),
            a=prev_lam,
            b=1.0,
        )
        protocol.append(next_lam)

        if t == max_iterations - 1:
            print(UserWarning('Exceeded max_iterations!'))

    if protocol[-1] != 1.0:
        protocol.append(1.0)

    return np.array(protocol)


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
    * stddev(prev_lam, next_lam) can be well-approximated by reweighting samples from initial protocol
    """
    # aggregate all samples from initial protocol
    reference_log_weights_n = log_weights_from_mixture(u_kn, f_k, N_k)

    # linearly interpolate previous energies
    _, vec_u_interp, vec_delta_u_interp = linear_u_kn_interpolant(lambdas_k, u_kn)

    #
    work_stddev_estimator = construct_work_stddev_estimator(reference_log_weights_n, vec_u_interp, vec_delta_u_interp)
    assess_lambda_pair = partial(
        construct_heuristic_lambda_pair_assessor(work_stddev_estimator),
        desired_stddev=work_stddev_threshold,
    )
    optimized_protocol = greedily_optimize_protocol(assess_lambda_pair)
    return optimized_protocol
