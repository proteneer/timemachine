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
    mbar_from_u_kln,
    pair_overlap_from_ukln,
)


def gaussian_ukln_example(
    params_1: Tuple[float, float], params_2: Tuple[float, float], seed: int = 0, n_samples: int = 2000
) -> NDArray:
    """Generate 2-state u_kln matrix for the specified pair of normal distributions."""

    def u(mu, sigma, x):
        return (x - mu) ** 2 / (2 * sigma ** 2)

    mu_1, sigma_1 = params_1
    mu_2, sigma_2 = params_2

    u_1 = partial(u, mu_1, sigma_1)
    u_2 = partial(u, mu_2, sigma_2)

    rng = np.random.default_rng(seed)

    x_1 = rng.normal(mu_1, sigma_1, (n_samples,))
    x_2 = rng.normal(mu_2, sigma_2, (n_samples,))

    return np.array([[u_1(x_1), u_1(x_2)], [u_2(x_1), u_2(x_2)]])


@pytest.mark.nogpu
@pytest.mark.parametrize("sigma", [0.1, 1.0, 10.0])
def test_bootstrap_bar(sigma):
    np.random.seed(0)
    n_bootstrap = 100

    # default rbfe instance size, varying difficulty
    u_kln = gaussian_ukln_example((0.0, 1.0), (1.0, sigma))

    # estimate 3 times
    dfs_ref, ddfs_ref = mbar_from_u_kln(u_kln).getFreeEnergyDifferences()
    df_ref = dfs_ref[1, 0]
    ddf_ref = ddfs_ref[1, 0]
    df_0, bootstrap_samples = bootstrap_bar(u_kln, n_bootstrap=n_bootstrap)
    df_1, bootstrap_sigma = bar_with_bootstrapped_uncertainty(u_kln)

    # assert estimates identical, uncertainties comparable
    print(f"bootstrap uncertainty = {bootstrap_sigma}, pymbar.MBAR uncertainty = {ddf_ref}")
    assert df_0 == df_ref
    assert df_1 == df_ref
    assert len(bootstrap_samples) == n_bootstrap, "timed out on default problem size!"
    np.testing.assert_approx_equal(bootstrap_sigma, ddf_ref, significant=1)


@pytest.mark.parametrize("sigma", [0.3, 1.0, 10.0])
def test_compare_with_pymbar_bar(sigma):
    """Compare the estimator used for 2-state delta fs (currently MBAR) with pymbar.BAR as reference."""
    u_kln = gaussian_ukln_example((0.0, 1.0), (1.0, sigma))

    w_F = u_kln[1, 0] - u_kln[0, 0]
    w_R = u_kln[0, 1] - u_kln[1, 1]

    df_ref, _ = pymbar.BAR(w_F, w_R)
    dfs, _ = mbar_from_u_kln(u_kln).getFreeEnergyDifferences()

    assert dfs[1, 0] == pytest.approx(df_ref, rel=0.05, abs=0.01)


@pytest.mark.nogpu
def test_pair_overlap_from_ukln():
    # identical distributions
    assert pair_overlap_from_ukln(gaussian_ukln_example((0, 1), (0, 1))) == pytest.approx(1.0)

    # non-overlapping
    assert pair_overlap_from_ukln(gaussian_ukln_example((0, 0.01), (1, 0.01))) < 1e-10

    # overlapping
    assert pair_overlap_from_ukln(gaussian_ukln_example((0, 0.1), (0.5, 0.2))) > 0.1


@pytest.mark.nogpu
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
