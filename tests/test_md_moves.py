from typing import Callable, Tuple

import numpy as np
import pytest
from scipy.stats import ks_2samp

from timemachine.md.moves import MetropolisHastingsMove


class RWMH1D(MetropolisHastingsMove[float]):
    def __init__(self, log_q: Callable[[float], float], proposal_radius: float):
        super().__init__()
        self.log_q = log_q
        self.proposal_radius = proposal_radius

    def propose_with_log_q_diff(self, x: float) -> Tuple[float, float]:
        dx = self.proposal_radius
        x_prop = x + np.random.uniform(-dx, dx)
        log_q_diff = self.log_q(x_prop) - self.log_q(x)
        return x_prop, log_q_diff


@pytest.mark.parametrize("dist", ["normal", "uniform"])
@pytest.mark.parametrize("seed", [2023, 2024, 2025])
def test_random_walk_metropolis_hastings(dist, seed):
    """Test sampling from a 1-d normal distribution using Random Walk Metropolis-Hastings"""

    np.random.seed(seed)
    n_samples = 200_000
    dx = 0.1

    # estimate autocorrelation time, number of independent samples
    tau = round(2 / dx ** 2)
    n_independent_samples = n_samples // tau

    if dist == "normal":
        log_q = lambda x: -(x ** 2) / 2
        target_samples = np.random.normal(0, 1, size=(n_independent_samples,))
    else:
        log_q = lambda x: 0.0 if -1 < x < 1 else -float("inf")
        target_samples = np.random.uniform(-1, 1, size=(n_independent_samples,))

    sampler = RWMH1D(log_q, dx)
    x_0 = np.random.uniform(-1.0, 1.0)
    rw_samples = sampler.sample_chain(x_0, n_samples)

    decorrelated_rw_samples = rw_samples[::tau]

    _, pvalue = ks_2samp(decorrelated_rw_samples, target_samples)

    assert pvalue >= 0.05
