from timemachine.optimize.protocol import (
    rebalance_initial_protocol,
    log_weights_from_mixture,
    linear_u_kn_interpolant,
    construct_work_stddev_estimator,
)

from pymbar import MBAR
from pymbar.testsystems import HarmonicOscillatorsTestCase

import numpy as np
from scipy.special import logsumexp

np.random.seed(2021)


def test_rebalance_initial_protocol():
    """Integration test: assert that protocol optimization improves run-to-run variance in free energy estimates"""
    initial_protocol = np.linspace(0, 1, 64)
    mbar = simulate_protocol(initial_protocol, seed=2021)
    new_protocol = rebalance_initial_protocol(
        initial_protocol,
        mbar.f_k,
        mbar.u_kn,
        mbar.N_k,
        work_stddev_threshold=0.25,
    )

    # for a fair comparison, re-run initial protocol with same number of lambda windows as new_protocol
    new_K = len(new_protocol)
    old_protocol = np.linspace(0, 1, new_K)

    # run several replicates to measure run-to-run variability
    n_replicates = 10
    old_mbars = []
    new_mbars = []
    for i in range(n_replicates):
        old_mbars.append(simulate_protocol(old_protocol, seed=i))
        new_mbars.append(simulate_protocol(new_protocol, seed=i))

    old_stddev = np.std([mbar.f_k[-1] for mbar in old_mbars])
    new_stddev = np.std([mbar.f_k[-1] for mbar in new_mbars])

    print(f"empirical free energy estimate stddev across {n_replicates} replicates, with # of lambda windows = {new_K}")
    print(f"\told stddev: {old_stddev:.3f}")
    print(f"\tnew stddev: {new_stddev:.3f}")

    assert new_stddev < old_stddev


def test_log_weights_from_mixture():
    """Assert self-consistency between
    (1) delta_f from mbar.f_k[-1] - mbar.f_k[0] and
    (2) delta_f from comparing endpoints to mixture"""
    mbar = simulate_protocol(np.linspace(0, 1, 32), seed=2021)
    source_delta_f = mbar.f_k[-1] - mbar.f_k[0]

    # reconstruct by comparing endpoints to mixture
    log_weights_n = log_weights_from_mixture(mbar.u_kn, mbar.f_k, mbar.N_k)
    logpdf_0_n = -mbar.u_kn[0]
    logpdf_1_n = -mbar.u_kn[-1]

    N = np.sum(mbar.N_k)
    f_0 = -(logsumexp(logpdf_0_n - log_weights_n) - np.log(N))
    f_1 = -(logsumexp(logpdf_1_n - log_weights_n) - np.log(N))
    recons_delta_f = f_1 - f_0

    np.testing.assert_almost_equal(source_delta_f, recons_delta_f)


def test_linear_u_kn_interpolant():
    """Assert self-consistency with input"""
    lambdas = np.linspace(0, 1, 64)
    mbar = simulate_protocol(lambdas, seed=2021)
    vec_u_interp = linear_u_kn_interpolant(lambdas, mbar.u_kn)

    for _ in range(10):
        k = np.random.randint(len(lambdas))
        np.testing.assert_allclose(mbar.u_kn[k], vec_u_interp(lambdas[k]))


def test_work_stddev_estimator():
    """Assert nonegative, assert bigger estimates for more distant pairs"""
    lambdas = np.linspace(0, 1, 64)
    mbar = simulate_protocol(lambdas)
    reference_log_weights_n = log_weights_from_mixture(mbar.u_kn, mbar.f_k, mbar.N_k)
    vec_u_interp = linear_u_kn_interpolant(lambdas, mbar.u_kn)
    work_stddev_estimator = construct_work_stddev_estimator(reference_log_weights_n, vec_u_interp)

    for _ in range(10):
        prev_lam, next_lam = np.random.rand(2)
        assert work_stddev_estimator(prev_lam, next_lam) > 0

    for _ in range(10):
        prev_lam = np.random.rand()
        next_lams = np.linspace(prev_lam, 1.0, 5)
        next_stddevs = [work_stddev_estimator(prev_lam, next_lam) for next_lam in next_lams]
        assert (np.diff(next_stddevs) > 0).all()


def poorly_spaced_path(lam):
    """lam in [0,1] -> (offset in [0, 4], force_constant in [1, 16])"""
    lam_eff = lam ** 4

    offset = 4 * lam_eff
    force_constant = 2 ** (4 * lam_eff)

    return offset, force_constant


def simulate_protocol(lambdas_k, n_samples_per_window=100, seed=None):
    """Generate samples from each lambda window, plug into MBAR"""
    O_k, K_k = poorly_spaced_path(lambdas_k)
    testsystem = HarmonicOscillatorsTestCase(O_k, K_k)
    N_k = [n_samples_per_window] * len(O_k)
    xs, u_kn, N_k, s_n = testsystem.sample(N_k, seed=seed)
    mbar = MBAR(u_kn, N_k)
    return mbar
