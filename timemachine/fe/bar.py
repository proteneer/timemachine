import logging
from time import time

import jax
import jax.numpy as jnp
import numpy as np
import pymbar
from jax.scipy.special import logsumexp
from scipy.stats import normaltest

logger = logging.getLogger(__name__)


def EXP(w_raw):
    """
    Estimate free energy difference using exponential averaging

    Parameters
    ----------
    w : np.ndarray, float, (N)
        work for N frames

    Returns
    ------
    deltaF : scalar, float
        free energy difference

    """
    w = []
    for ww in w_raw:
        if ww is not None:
            w.append(ww)
    w = jnp.array(w)
    T = jnp.float64(jnp.size(w))
    deltaF = -(logsumexp(-w) - jnp.log(T))
    return deltaF


def BARzero(w, deltaF):
    """
    A function that when zeroed is equivalent to the solution of BAR

    Parameters
    ---------
    w : np.ndarray, float, (2, N)
        forward and reverse work for N frames

    deltaF : scalar, float
        free energy difference

    Returns
    ------
    scalar, float
        a variable that is zeroed when deltaF satisfies BAR

    """
    w_F = w[0]
    w_R = w[1]
    T_F = len(w_F)
    T_R = len(w_R)
    M = jnp.log(T_F / T_R)

    exp_arg_F = M + w_F - deltaF
    max_arg_F = jnp.where(exp_arg_F < 0, 0.0, exp_arg_F)

    log_f_F = -max_arg_F - jnp.log(jnp.exp(-max_arg_F) + jnp.exp(exp_arg_F - max_arg_F))
    log_numer = logsumexp(log_f_F)
    exp_arg_R = -(M - w_R - deltaF)
    max_arg_R = jnp.where(exp_arg_R < 0, 0.0, exp_arg_R)

    log_f_R = -max_arg_R - jnp.log(jnp.exp(-max_arg_R) + jnp.exp(exp_arg_R - max_arg_R))

    log_denom = logsumexp(log_f_R)
    fzero = log_numer - log_denom
    return fzero


def dG_dw(w):
    """
    A function that calculates gradient of free energy difference with respect to work

    Parameters
    ---------
    w : np.ndarray, float, (2, N)
        forward and reverse work for N frames

    Returns
    ------
    np.ndarray, float, (2, N)
        the gradient of free energy difference with respect to work

    """
    dG, _ = pymbar.BAR(w[0], w[1])
    dBAR_dw = jax.grad(BARzero, argnums=(0,))
    dBAR_dA = jax.grad(BARzero, argnums=(1,))
    dG_dw = -dBAR_dw(w, dG)[0] / dBAR_dA(w, dG)[0]
    return dG_dw


def bootstrap_bar(w_F, w_R, n_bootstrap=1000, timeout=10):
    """Subsample w_F, w_R with replacement and re-run BAR many times

    Parameters
    ----------
    w_F : array
        forward works
    w_R : array
        reverse works
    n_bootstrap : int
        # bootstrap samples
    timeout : int
        in seconds

    Returns
    -------
    best_estimate : float
        BAR(w_F, w_R, computeUncertainty=False)
    bootstrap_samples: array
        length <= n_bootstrap
        (length < n_bootstrap if timed out)

    Notes
    -----
    * TODO[deboggle] -- upgrade from pymbar3 to pymbar4 and remove this
    * TODO[performance] -- multiprocessing, if needed?
    """
    full_bar_result = pymbar.BAR(w_F, w_R, compute_uncertainty=False)

    n_F, n_R = len(w_F), len(w_R)

    bootstrap_samples = []

    t0 = time()

    for _ in range(n_bootstrap):
        elapsed_time = time() - t0
        if elapsed_time > timeout:
            break

        inds_F = np.random.randint(0, n_F, n_F)
        inds_R = np.random.randint(0, n_R, n_R)

        bar_result = pymbar.BAR(
            w_F=w_F[inds_F],
            w_R=w_R[inds_R],
            DeltaF=full_bar_result,  # warm start
            compute_uncertainty=False,
            relative_tolerance=1e-6,  # reduce cost
        )

        bootstrap_samples.append(bar_result)

    return full_bar_result, np.array(bootstrap_samples)


def bar_with_bootstrapped_uncertainty(w_F, w_R, n_bootstrap=1000, timeout=10):
    """Drop-in replacement for pymbar.BAR(w_F, w_R) -> (df, ddf)
    where first return is forwarded from pymbar.BAR but second return is computed by bootstrapping"""

    df, bootstrap_dfs = bootstrap_bar(w_F, w_R, n_bootstrap=n_bootstrap, timeout=timeout)

    # warn if bootstrap distribution deviates significantly from normality
    normaltest_result = normaltest(bootstrap_dfs)
    pvalue_threshold = 1e-3  # arbitrary, small
    if normaltest_result.pvalue < pvalue_threshold:
        logger.warning(f"bootstrapped errors non-normal: {normaltest_result}")

    # regardless, summarize as if normal
    ddf = np.std(bootstrap_dfs)
    return df, ddf
