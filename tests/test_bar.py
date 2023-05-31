import numpy as np
import pymbar
import pytest
from pymbar.testsystems import ExponentialTestCase, gaussian_work_example

from timemachine.fe.bar import (
    bar_with_bootstrapped_uncertainty,
    bootstrap_bar,
    compute_fwd_and_reverse_df_over_time,
    pair_overlap_from_ukln,
)


@pytest.mark.nogpu
def test_bootstrap_bar():
    np.random.seed(0)
    n_bootstrap = 1000

    for sigma_F in [0.1, 1, 10]:
        # default rbfe instance size, varying difficulty
        w_F, w_R = gaussian_work_example(2000, 2000, sigma_F=sigma_F, seed=0)

        # estimate 3 times
        df_ref, ddf_ref = pymbar.BAR(w_F, w_R)
        df_0, bootstrap_samples = bootstrap_bar(w_F, w_R, n_bootstrap=n_bootstrap)
        df_1, bootstrap_sigma = bar_with_bootstrapped_uncertainty(w_F, w_R)

        # assert estimates identical, uncertainties comparable
        print(f"stddev(w_F) = {sigma_F}, bootstrap uncertainty = {bootstrap_sigma}, pymbar.BAR uncertainty = {ddf_ref}")
        assert df_0 == df_ref
        assert df_1 == df_ref
        assert len(bootstrap_samples) == n_bootstrap, "timed out on default problem size!"
        np.testing.assert_approx_equal(bootstrap_sigma, ddf_ref, significant=1)


@pytest.mark.nogpu
def test_pair_overlap_from_ukln():
    def gaussian_overlap(p1, p2):
        def make_gaussian(params):
            mu, sigma = params

            def u(x):
                return (x - mu) ** 2 / (2 * sigma ** 2)

            rng = np.random.default_rng(2022)
            x = rng.normal(mu, sigma, 100)

            return u, x

        u1, x1 = make_gaussian(p1)
        u2, x2 = make_gaussian(p2)

        u_kln = np.array([[u1(x1), u1(x2)], [u2(x1), u2(x2)]])

        return pair_overlap_from_ukln(u_kln)

    # identical distributions
    np.testing.assert_allclose(gaussian_overlap((0, 1), (0, 1)), 1.0)

    # non-overlapping
    assert gaussian_overlap((0, 0.01), (1, 0.01)) < 1e-10

    # overlapping
    assert gaussian_overlap((0, 0.1), (0.5, 0.2)) > 0.1


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
