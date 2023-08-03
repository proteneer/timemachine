from functools import partial
from typing import Tuple

import numpy as np
import pymbar
import pytest
from numpy.typing import NDArray
from pymbar.testsystems import ExponentialTestCase

from timemachine.fe.bar import (
    bar_with_bootstrapped_uncertainty,
    bootstrap_bar,
    compute_fwd_and_reverse_df_over_time,
    df_and_err_from_u_kln,
    df_from_u_kln,
    pair_overlap_from_ukln,
    ukln_to_ukn,
    works_from_ukln,
)

pytestmark = [pytest.mark.nocuda]


def make_gaussian_ukln_example(
    params_a: Tuple[float, float], params_b: Tuple[float, float], seed: int = 0, n_samples: int = 2000
) -> Tuple[NDArray, float]:
    """Generate 2-state u_kln matrix for a pair of normal distributions."""

    def u(mu, sigma, x):
        return (x - mu) ** 2 / (2 * sigma ** 2)

    mu_a, sigma_a = params_a
    mu_b, sigma_b = params_b

    u_a = partial(u, mu_a, sigma_a)
    u_b = partial(u, mu_b, sigma_b)

    rng = np.random.default_rng(seed)

    x_a = rng.normal(mu_a, sigma_a, (n_samples,))
    x_b = rng.normal(mu_b, sigma_b, (n_samples,))

    u_kln = np.array([[u_a(x_a), u_a(x_b)], [u_b(x_a), u_b(x_b)]])

    dlogZ = np.log(sigma_a) - np.log(sigma_b)

    return u_kln, dlogZ


def make_partial_overlap_uniform_ukln_example(dlogZ: float, n_samples: int = 100) -> NDArray:
    """Generate 2-state u_kln matrix for uniform distributions with partial overlap"""

    def u_a(x):
        """Unif[0.0, 1.0], with log(Z) = 0"""
        in_bounds = (x > 0) * (x < 1)
        return np.where(in_bounds, 0, +np.inf)

    def u_b(x):
        """Unif[0.5, 1.5], with log(Z) = dlogZ"""
        x_ = x - 0.5
        return u_a(x_) + dlogZ

    rng = np.random.default_rng(2023)

    x_a = rng.uniform(0, 1, (n_samples,))
    x_b = rng.uniform(0.5, 1.5, (n_samples,))

    assert np.isfinite(u_a(x_a)).all()
    assert np.isfinite(u_b(x_b)).all()

    u_kln = np.array([[u_a(x_a), u_a(x_b)], [u_b(x_a), u_b(x_b)]])

    return u_kln


@pytest.mark.parametrize("sigma", [0.1, 1.0, 10.0])
def test_bootstrap_bar(sigma):
    np.random.seed(0)
    n_bootstrap = 100

    # default rbfe instance size, varying difficulty
    u_kln, dlogZ = make_gaussian_ukln_example((0.0, 1.0), (1.0, sigma))

    # estimate 3 times
    df_ref, df_err_ref = df_and_err_from_u_kln(u_kln)
    df_0, bootstrap_samples = bootstrap_bar(u_kln, n_bootstrap=n_bootstrap)
    df_1, bootstrap_sigma = bar_with_bootstrapped_uncertainty(u_kln)

    # assert estimates identical, uncertainties comparable
    print(f"bootstrap uncertainty = {bootstrap_sigma}, pymbar.MBAR uncertainty = {df_err_ref}")
    assert df_0 == df_ref
    assert df_1 == df_ref
    assert len(bootstrap_samples) == n_bootstrap
    np.testing.assert_approx_equal(bootstrap_sigma, df_err_ref, significant=1)

    # assert bootstrap estimate is consistent with exact result
    assert df_1 == pytest.approx(dlogZ, abs=2.0 * bootstrap_sigma)


@pytest.mark.parametrize("sigma", [0.1, 1.0, 10.0])
def test_df_from_u_kln_consistent_with_df_and_err_from_u_kln(sigma):
    u_kln, _ = make_gaussian_ukln_example((0.0, 1.0), (1.0, sigma))
    df_ref, _ = df_and_err_from_u_kln(u_kln)
    df = df_from_u_kln(u_kln)
    assert df == df_ref


@pytest.mark.parametrize("sigma", [0.1, 1.0, 10.0])
def test_df_and_err_from_u_kln_approximates_exact_result(sigma):
    u_kln, dlogZ = make_gaussian_ukln_example((0.0, 1.0), (1.0, sigma))
    df, df_err = df_and_err_from_u_kln(u_kln)
    assert df == pytest.approx(dlogZ, abs=2.0 * df_err)


@pytest.mark.parametrize("sigma", [0.3, 1.0, 10.0])
def test_df_and_err_from_u_kln_consistent_with_pymbar_bar(sigma):
    """Compare the estimator used for 2-state delta fs (currently MBAR) with pymbar.BAR as reference."""
    u_kln, _ = make_gaussian_ukln_example((0.0, 1.0), (1.0, sigma))
    w_F, w_R = works_from_ukln(u_kln)

    df_ref, df_err_ref = pymbar.BAR(w_F, w_R)
    df, df_err = df_and_err_from_u_kln(u_kln)

    assert df == pytest.approx(df_ref, rel=0.05, abs=0.01)
    np.testing.assert_approx_equal(df_err, df_err_ref, significant=1)


def test_df_and_err_from_u_kln_partial_overlap():
    dlogZ = 5.0
    u_kln = make_partial_overlap_uniform_ukln_example(dlogZ)

    w_F, w_R = works_from_ukln(u_kln)

    # this example has some infinite work values
    assert np.any(np.isinf(w_F))
    assert np.any(np.isinf(w_R))

    # but no NaNs
    assert not np.any(np.isnan(w_F))
    assert not np.any(np.isnan(w_R))

    # pymbar.BAR warns and returns zero for df and uncertainty with default method
    assert pymbar.BAR(w_F, w_R) == (0.0, 0.0)

    # pymbar.BAR returns NaNs with self-consistent iteration method
    df_sci, df_err_sci = pymbar.BAR(w_F, w_R, method="self-consistent-iteration")
    assert np.isnan(df_sci)
    assert np.isnan(df_err_sci)

    df, df_err = df_and_err_from_u_kln(u_kln)
    assert df == pytest.approx(dlogZ, abs=2.0 * df_err)
    assert np.isfinite(df_err) and df_err > 0.0


def test_df_from_u_kln_does_not_raise_on_incomplete_convergence():
    u_kln = make_partial_overlap_uniform_ukln_example(5.0)

    # pymbar raises an exception on incomplete convergence when computing covariances
    u_kn, N_k = ukln_to_ukn(u_kln)
    mbar = pymbar.MBAR(u_kn, N_k, maximum_iterations=1)
    with pytest.raises(pymbar.utils.ParameterError):
        _ = mbar.getFreeEnergyDifferences()

    # no exception if we don't compute uncertainty
    _ = mbar.getFreeEnergyDifferences(compute_uncertainty=False)

    # df_from_u_kln, df_and_err_from_u_kln wrappers do not raise exceptions
    df = df_from_u_kln(u_kln, maximum_iterations=1)
    assert np.isfinite(df)

    df, ddf = df_and_err_from_u_kln(u_kln, maximum_iterations=1)
    assert np.isfinite(df)
    assert np.isnan(ddf)  # returns NaN for uncertainty on incomplete convergence


def test_pair_overlap_from_ukln():
    # identical distributions
    u_kln, _ = make_gaussian_ukln_example((0, 1), (0, 1))
    assert pair_overlap_from_ukln(u_kln) == pytest.approx(1.0)

    # non-overlapping
    u_kln, _ = make_gaussian_ukln_example((0, 0.01), (1, 0.01))
    assert pair_overlap_from_ukln(u_kln) < 1e-10

    # overlapping
    u_kln, _ = make_gaussian_ukln_example((0, 0.1), (0.5, 0.2))
    assert pair_overlap_from_ukln(u_kln) > 0.1


@pytest.mark.parametrize("frames_per_step", [1, 5, 10])
def test_compute_fwd_and_reverse_df_over_time(frames_per_step):
    seed = 2023
    pair_u_klns = 47

    rng = np.random.default_rng(seed)

    _, u_kln, _ = ExponentialTestCase(rates=[1, 2]).sample(N_k=(5, 10), mode="u_kln", seed=seed)
    assert u_kln.shape == (2, 2, 10)
    u_kln_by_lambda = np.stack([u_kln] * pair_u_klns)

    noise = rng.random(size=(u_kln_by_lambda.shape))
    u_kln_by_lambda += noise

    with pytest.raises(AssertionError, match="fewer samples than frames_per_step"):
        compute_fwd_and_reverse_df_over_time(u_kln_by_lambda, frames_per_step=u_kln_by_lambda.shape[-1] + 1)

    fwd, fwd_err, rev, rev_err = compute_fwd_and_reverse_df_over_time(u_kln_by_lambda, frames_per_step=frames_per_step)

    chunks = u_kln.shape[-1] // frames_per_step
    assert len(fwd) == chunks
    assert len(fwd_err) == chunks

    assert len(rev) == chunks
    assert len(rev_err) == chunks

    # The values at the end should be nearly identical since they contain all the samples
    assert np.allclose(fwd[-1], rev[-1])
    assert np.allclose(fwd_err[-1], rev_err[-1])
