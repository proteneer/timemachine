import logging
from time import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pymbar
from jax.scipy.special import logsumexp
from numpy.typing import NDArray
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
    dG = pymbar.bar(w[0], w[1])["Delta_f"]
    dBAR_dw = jax.grad(BARzero, argnums=(0,))
    dBAR_dA = jax.grad(BARzero, argnums=(1,))
    dG_dw = -dBAR_dw(w, dG)[0] / dBAR_dA(w, dG)[0]
    return dG_dw


def bootstrap_bar(w_F, w_R, n_bootstrap=1000, timeout=10) -> Tuple[float, NDArray]:
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
    full_bar_result = pymbar.bar(w_F, w_R, compute_uncertainty=False)

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

        bar_result = pymbar.bar(
            w_F=w_F_sample,
            w_R=w_R_sample,
            DeltaF=full_bar_result["Delta_f"],  # warm start
            compute_uncertainty=False,
            relative_tolerance=1e-6,  # reduce cost
        )

        bootstrap_samples.append(bar_result["Delta_f"])

    return full_bar_result["Delta_f"], np.array(bootstrap_samples)


def bar_with_bootstrapped_uncertainty(w_F, w_R, n_bootstrap=1000, timeout=10) -> Tuple[float, float]:
    """Returns BAR estimate from pymbar.bar and error estimate computed by bootstrapping."""

    df, bootstrap_dfs = bootstrap_bar(w_F, w_R, n_bootstrap=n_bootstrap, timeout=timeout)

    # warn if bootstrap distribution deviates significantly from normality
    normaltest_result = normaltest(bootstrap_dfs)
    pvalue_threshold = 1e-3  # arbitrary, small
    if normaltest_result.pvalue < pvalue_threshold:
        logger.warning(f"bootstrapped errors non-normal: {normaltest_result}")

    # regardless, summarize as if normal
    ddf = np.std(bootstrap_dfs)
    return df, ddf


def df_from_ukln(u_kln: np.ndarray) -> Tuple[float, float]:
    """Extract forward and reverse works from 2-state u_kln matrix and return BAR dF and dF error computed by pymbar

    Parameters
    ----------
    u_kln : [2, 2, n] array
        pymbar u_kln input format, where k = l = 2

    Returns
    -------
    df_err: float
        BAR dF
    df_err: float
        BAR uncertainty
    """
    k, l, _ = u_kln.shape
    assert k == l == 2
    w_fwd = u_kln[1, 0, :] - u_kln[0, 0, :]
    w_rev = u_kln[0, 1, :] - u_kln[1, 1, :]
    bar = pymbar.bar(w_fwd, w_rev)
    df, df_err = bar["Delta_f"], bar["dDelta_f"]
    return df, df_err


def df_err_from_ukln(u_kln: np.ndarray) -> float:
    """Extract forward and reverse works from 2-state u_kln matrix and return BAR error computed by pymbar

    Parameters
    ----------
    u_kln : [2, 2, n] array
        pymbar u_kln input format, where k = l = 2

    Returns
    -------
    df_err: float
        BAR uncertainty
    """
    _, df_err = df_from_ukln(u_kln)
    return df_err


def df_from_ukln_by_lambda(ukln_by_lambda: NDArray) -> Tuple[float, float]:
    """Extract dF and dF error compute by BAR over a series of lambda windows

    Parameters
    ----------
    u_kln : [n_lambda, 2, 2, n] array
        pymbar u_kln input format, where k = l = 2

    Returns
    -------
    df: float
        pair BAR dF across lambda
    df_err: float
        pair BAR uncertainty across lambda
    """
    win_dfs = []
    win_errs = []
    for lambda_idx in range(ukln_by_lambda.shape[0]):
        window_ukln = ukln_by_lambda[lambda_idx]

        dF, dF_err = df_from_ukln(window_ukln)
        win_dfs.append(dF)
        win_errs.append(dF_err)
    return np.sum(win_dfs), np.linalg.norm(win_errs)  # type: ignore


def pair_overlap_from_ukln(u_kln: np.ndarray) -> float:
    """Compute the off-diagonal entry of 2x2 MBAR overlap matrix,
        and normalize to interval [0,1]

    Parameters
    ----------
    u_kln : [2, 2, n] array
        pymbar u_kln input format, where k = l = 2

    Returns
    -------
    pair_overlap: float
        2 * pymbar.MBAR overlap
        (normalized to interval [0,1] rather than [0,0.5])

    """
    k, l, n = u_kln.shape
    assert k == l == 2
    u_kn = u_kln.reshape(k, -1)
    assert u_kn.shape == (k, l * n)
    N_k = n * np.ones(l)
    return 2 * pymbar.MBAR(u_kn, N_k).compute_overlap()["matrix"][0, 1]


def compute_fwd_and_reverse_df_over_time(
    ukln_by_lambda: NDArray, frames_per_step: int = 100
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Provided a u_kln computes the forward and reverse dF estimates.

    Computes the dF value for increasing numbers of samples in both the forward and reverse direction.

    Parameters
    ----------

    ukln_by_lambda: [n_lambda, 2, 2, N] array
        Array of u_kln broken up by lambda windows

    frames_per_step: int
        Number of frames to include in a sample when computing u_kln over time

    Returns
    -------
        fwd_df: [N // frames_per_step] np.ndarray
            numpy array of dF for each chunk
        fwd_df_err: [N // frames_per_step] np.ndarray
            numpy array of dF errors for each chunk
        rev_df: [N // frames_per_step] np.ndarray
            numpy array of dF for each chunk, from reversed ukln
        rev_df_err: [N // frames_per_step] np.ndarray
            numpy array of dF errors for each chunk, from reversed ukln
    """
    assert len(ukln_by_lambda.shape) == 4
    assert ukln_by_lambda.shape[1] == 2
    forward_predictions_ = []
    reverse_predictions_ = []
    total_frames = ukln_by_lambda.shape[-1]
    assert total_frames >= frames_per_step, "fewer samples than frames_per_step"

    # Reverse the u_kln along last axis to get the reverse
    reversed_ukln_by_lambda = np.flip(ukln_by_lambda, 3)
    for num_frames in range(frames_per_step, total_frames + 1, frames_per_step):
        fwd_ukln = ukln_by_lambda[..., :num_frames]
        reverse_ukln = reversed_ukln_by_lambda[..., :num_frames]

        forward_predictions_.append(df_from_ukln_by_lambda(fwd_ukln))
        reverse_predictions_.append(df_from_ukln_by_lambda(reverse_ukln))

    forward_predictions = np.array(forward_predictions_)
    reverse_predictions = np.array(reverse_predictions_)
    return forward_predictions[:, 0], forward_predictions[:, 1], reverse_predictions[:, 0], reverse_predictions[:, 1]
