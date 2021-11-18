"""Protocol optimization"""

from jax import jit, vmap, numpy as np
from jax.scipy.special import logsumexp
from scipy.optimize import bisect

from typing import Tuple, Callable

Float = float
Array = np.array


def linear_u_kn_interpolant(lambdas: Array, u_kn: Array) -> Tuple[Callable, Callable, Callable]:
    """given a matrix u_kn[k, n] = u(xs[n], lambdas[k])

    produce linear interpolated estimates of u(xs[n], lam)
    at arbitrary new values lam
    """

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
    """assume
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


StepAssessor = Callable[[Float, Float], Float]


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
