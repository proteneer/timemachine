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
    w_raw : np.ndarray, float, (N)
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

    seed = 2022
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstrap):
        elapsed_time = time() - t0
        if elapsed_time > timeout:
            break

        w_F_sample = rng.choice(w_F, size=(n_F,), replace=True)
        w_R_sample = rng.choice(w_R, size=(n_R,), replace=True)

        bar_result = pymbar.BAR(
            w_F=w_F_sample,
            w_R=w_R_sample,
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


def df_err_from_ukln(u_kln):
    k, l, _ = u_kln.shape
    assert k == l == 2
    w_fwd = u_kln[1, 0, :] - u_kln[0, 0, :]
    w_rev = u_kln[0, 1, :] - u_kln[1, 1, :]
    _, df_err = bar_with_bootstrapped_uncertainty(w_fwd, w_rev)
    return df_err


def pair_overlap_from_ukln(u_kln):
    k, l, n = u_kln.shape
    assert k == l == 2
    u_kn = u_kln.reshape(k, -1)
    assert u_kn.shape == (k, l * n)
    N_k = n * np.ones(l)
    return 2 * pymbar.MBAR(u_kn, N_k).computeOverlap()["matrix"][0, 1]  # type: ignore
