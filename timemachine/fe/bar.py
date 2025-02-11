import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pymbar
from jax.scipy.special import logsumexp
from numpy.typing import NDArray
from pymbar.utils import kln_to_kn
from scipy.stats import normaltest

DG_KEY = "Delta_f"
DG_ERR_KEY = "dDelta_f"

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
    bar = pymbar.bar(w[0], w[1])
    dG = bar[DG_KEY]
    dBAR_dw = jax.grad(BARzero, argnums=(0,))
    dBAR_dA = jax.grad(BARzero, argnums=(1,))
    return -dBAR_dw(w, dG)[0] / dBAR_dA(w, dG)[0]


def ukln_to_ukn(u_kln: NDArray) -> tuple[NDArray, NDArray]:
    """Convert 2-state u_kln matrix to u_kn and N_k, i.e. the inputs expected by pymbar.MBAR.

    NOTE: similar to https://pymbar.readthedocs.io/en/master/utils.html#pymbar.utils.kln_to_kn,
    but also return the N_k array for MBAR. This uses the PyMBAR axis convention.

    Parameters
    ----------
    u_kln : array (2, 2, N)
        2-state u_kln matrix, where
        * the first dimension (k) indexes the state from which the configuration was sampled
        * the second dimension (l) indexes the state for which we evaluate the energy
    """
    u_kn = kln_to_kn(u_kln)
    k, l, n = u_kln.shape
    assert k == l == 2
    assert u_kn.shape == (k, l * n)
    N_k = n * np.ones(l)
    return u_kn, N_k


DEFAULT_RELATIVE_TOLERANCE = 1e-6  # pymbar default 1e-7
DEFAULT_MAXIMUM_ITERATIONS = 1_000  # pymbar default 10_000


def df_and_err_from_u_kln(u_kln: NDArray, maximum_iterations: int = DEFAULT_MAXIMUM_ITERATIONS) -> tuple[float, float]:
    """Compute free energy difference and uncertainty given a 2-state u_kln matrix."""
    u_kn, N_k = ukln_to_ukn(u_kln)
    mbar = pymbar.mbar.MBAR(u_kn, N_k, maximum_iterations=maximum_iterations)
    try:
        results = mbar.compute_free_energy_differences()
        df, ddf = results[DG_KEY], results[DG_ERR_KEY]
        return df[0, 1], ddf[0, 1]
    except pymbar.utils.ParameterError:
        # As of pymbar 3.1.0, computation of the covariance matrix can raise an exception on incomplete convergence.
        # In this case, return the unconverged estimate with NaN as uncertainty.
        df = mbar.compute_free_energy_differences(compute_uncertainty=False)[DG_KEY]
        return df[0, 1], np.nan


def df_from_u_kln(
    u_kln: NDArray, initial_f_k: Optional[NDArray] = None, maximum_iterations: int = DEFAULT_MAXIMUM_ITERATIONS
) -> float:
    """Compute free energy difference given a 2-state u_kln matrix."""
    u_kn, N_k = ukln_to_ukn(u_kln)
    mbar = pymbar.mbar.MBAR(u_kn, N_k, initial_f_k=initial_f_k, maximum_iterations=maximum_iterations)
    df = mbar.compute_free_energy_differences(compute_uncertainty=False)[DG_KEY]
    return df[0, 1]


def bootstrap_bar(
    u_kln: NDArray, n_bootstrap: int = 100, maximum_iterations: int = DEFAULT_MAXIMUM_ITERATIONS
) -> tuple[float, float, NDArray]:
    """Given a 2-state u_kln matrix, subsample u_kln with replacement and re-run df_from_u_kln many times

    Parameters
    ----------
    u_kln : array
        2-state u_kln matrix
    n_bootstrap : int
        number of bootstrap samples
    maximum_iterations : int
        maximum number of solver iterations to use for each sample

    Returns
    -------
    best_estimate : float
        BAR(w_F, w_R)

    best_estimate_err : float
        MBAR(w_F, w_R) error estimate, using all samples

    bootstrap_samples : array
        shape (n_bootstrap,)

    Notes
    -----
    * TODO[deboggle] -- upgrade from pymbar3 to pymbar4 and remove this
    * TODO[performance] -- multiprocessing, if needed?
    """

    full_bar_result, full_bar_err = df_and_err_from_u_kln(u_kln, maximum_iterations=maximum_iterations)

    _, _, n = u_kln.shape

    bootstrap_samples = []

    seed = 2022
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstrap):
        u_kln_sample = rng.choice(u_kln, size=(n,), replace=True, axis=2)
        bar_result = df_from_u_kln(
            u_kln_sample,
            initial_f_k=np.array([0.0, full_bar_result]),  # warm start
            maximum_iterations=maximum_iterations,
        )
        bootstrap_samples.append(bar_result)

    return full_bar_result, full_bar_err, np.array(bootstrap_samples)


def bar_with_pessimistic_uncertainty(
    u_kln: NDArray, n_bootstrap=100, maximum_iterations: int = DEFAULT_MAXIMUM_ITERATIONS
) -> tuple[float, float]:
    """Given 2-state u_kln, returns free energy difference and the uncertainty. The uncertainty can be produced either by
    BAR using all samples or the bootstrapped error, whichever is greater.

    Parameters
    ----------
    u_kln : array
        2-state u_kln matrix
    n_bootstrap : int
        number of bootstrap samples
    maximum_iterations : int
        maximum number of solver iterations to use for each sample

    Returns
    -------
    best_estimate : float
        BAR(w_F, w_R)

    uncertainty : float
        `max(error_estimates)` where `error_estimates = [bootstrapped_bar_stddev, two_state_mbar_uncertainty]`
    """

    df, ddf, bootstrap_dfs = bootstrap_bar(u_kln, n_bootstrap=n_bootstrap, maximum_iterations=maximum_iterations)

    # warn if bootstrap distribution deviates significantly from normality
    normaltest_result = normaltest(bootstrap_dfs)
    pvalue_threshold = 1e-3  # arbitrary, small
    if normaltest_result.pvalue < pvalue_threshold:
        logger.warning(f"bootstrapped errors non-normal: {normaltest_result}")

    # Take the max of the BAR error estimate using all samples and the bootstrapped error. Summarize as if normal regardless
    # Use np.maximum to always return the NaN
    if not np.isfinite(ddf):
        logger.warning(f"BAR error estimate is not finite, setting to zero: {ddf}")
        ddf = 0.0
    ddf = np.maximum(ddf, np.std(bootstrap_dfs))
    return df, ddf


def bar(w_F: NDArray, w_R: NDArray, **kwargs) -> tuple[float, float | None]:
    """
    Wrapper around `pymbar.bar` to return the free energy and bar uncertainty.

    Parameters
    ----------
    w_F: np.ndarray, float, (N,)
        forward work for N frames

    w_R: np.ndarray, float, (N,)
        reverse work for N frames

    Returns
    -------
    best_estimate : float
        BAR(w_F, w_R)

    uncertainty : float if compute_uncertainty=True (default) else None
    """
    bar_result = pymbar.bar(w_F, w_R, **kwargs)
    if DG_ERR_KEY in bar_result:
        return bar_result[DG_KEY], bar_result[DG_ERR_KEY]
    else:
        return bar_result[DG_KEY], None


def works_from_ukln(u_kln: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract forward and reverse works from 2-state u_kln matrix"""
    k, l, _ = u_kln.shape
    assert k == l == 2
    w_fwd = u_kln[0, 1, :] - u_kln[0, 0, :]
    w_rev = u_kln[1, 0, :] - u_kln[1, 1, :]
    return w_fwd, w_rev


def df_from_ukln_by_lambda(ukln_by_lambda: NDArray) -> tuple[float, float]:
    """Extract df and df error computed by BAR over a series of lambda windows

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
        df, df_err = df_and_err_from_u_kln(window_ukln)
        win_dfs.append(df)
        win_errs.append(df_err)
    return np.sum(win_dfs), np.linalg.norm(win_errs)  # type: ignore


def pair_overlap_from_ukln(u_kln: NDArray) -> float:
    """Compute the off-diagonal entry of 2x2 MBAR overlap matrix,
        and normalize to interval [0,1]

    Parameters
    ----------
    u_kln : [2, 2, n] array
        pymbar u_kln input format, where k = l = 2

    Returns
    -------
    pair_overlap: float
        2 * pymbar.mbar.MBAR overlap
        (normalized to interval [0,1] rather than [0,0.5])

    """
    u_kn, N_k = ukln_to_ukn(u_kln)
    return pymbar.mbar.MBAR(u_kn, N_k).compute_overlap()["scalar"]  # type: ignore


def compute_fwd_and_reverse_df_over_time(
    ukln_by_lambda: NDArray, frames_per_step: int = 100
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
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
