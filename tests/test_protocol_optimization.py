from optimize.protocol import rebalance_initial_protocol

from pymbar import MBAR
from pymbar.testsystems import HarmonicOscillatorsTestCase

import numpy as np

np.random.seed(2021)


def test_rebalance_initial_protocol():
    """Integration test: assert that protocol optimization improves run-to-run variance in free energy estimates"""
    initial_protocol = np.linspace(0, 1, 64)
    mbar = simulate_protocol(initial_protocol)
    new_protocol = rebalance_initial_protocol(
        initial_protocol,
        mbar.f_k,
        mbar.u_kn,
        mbar.N_k,
        work_stddev_threshold=0.5,
    )

    # for a fair comparison, re-run initial protocol with same number of lambda windows as new_protocol
    new_K = len(new_protocol)
    old_protocol = np.linspace(0, 1, new_K)

    # run several replicates to measure run-to-run variability
    n_replicates = 10
    old_mbars = []
    new_mbars = []
    for _ in range(n_replicates):
        old_mbars.append(simulate_protocol(old_protocol))
        new_mbars.append(simulate_protocol(new_protocol))

    old_stddev = np.std([mbar.f_k[-1] for mbar in old_mbars])
    new_stddev = np.std([mbar.f_k[-1] for mbar in new_mbars])

    print(f'empirical free energy estimate stddev across {n_replicates} replicates, with # of lambda windows = {new_K}')
    print(f'\told stddev: {old_stddev:.3f}')
    print(f'\tnew stddev: {new_stddev:.3f}')

    assert new_stddev < old_stddev


def poorly_spaced_path(lam):
    """lam in [0,1] -> (offset in [0, 4], force_constant in [1, 16])"""
    lam_eff = lam ** 4

    offset = 4 * lam_eff
    force_constant = 2 ** (4 * lam_eff)

    return offset, force_constant


def simulate_protocol(lambdas_k, n_samples_per_window=100):
    """Generate samples from each lambda window, plug into MBAR"""
    O_k, K_k = poorly_spaced_path(lambdas_k)
    testsystem = HarmonicOscillatorsTestCase(O_k, K_k)
    N_k = [n_samples_per_window] * len(O_k)
    xs, u_kn, N_k, s_n = testsystem.sample(N_k)
    mbar = MBAR(u_kn, N_k)
    return mbar
