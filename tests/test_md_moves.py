from typing import Tuple

import numpy as np
from scipy.stats import ks_2samp

from timemachine.md.moves import MetropolisHastingsMove


def test_rwmh():
    """Test sampling from a 1-d normal distribution using Random Walk Metropolis-Hastings"""

    np.random.seed(2023)
    n_samples = 100_000
    dx = 0.1

    class RWMH1D(MetropolisHastingsMove[float]):
        def propose_with_dlogq(self, x: float) -> Tuple[float, float]:
            x_prop = x + np.random.uniform(-dx, dx)

            def log_q(x):
                return -(x ** 2) / 2

            dlogq = log_q(x_prop) - log_q(x)

            return x_prop, dlogq

    rwmh = RWMH1D()
    x_0 = np.random.uniform(-1.0, 1.0)
    rw_samples = rwmh.sample_chain(x_0, n_samples)

    tau = round(2 / dx ** 2)
    decorrelated_rw_samples = rw_samples[tau::tau]

    target_samples = np.random.normal(0, 1, size=(len(decorrelated_rw_samples),))

    _, pvalue = ks_2samp(decorrelated_rw_samples, target_samples)

    assert pvalue >= 0.05
