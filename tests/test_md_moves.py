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
        x_prop = np.random.normal(x, self.proposal_radius)
        log_q_diff = self.log_q(x_prop) - self.log_q(x)
        return x_prop, log_q_diff


@pytest.mark.parametrize("seed", [2023, 2024, 2025])
@pytest.mark.parametrize("dist", ["uniform", "normal"])
def test_random_walk_metropolis_hastings(dist, seed):
    """Test sampling from a 1-d normal distribution using Random Walk Metropolis-Hastings"""

    np.random.seed(seed)
    n_samples = 100_000
    dx = 0.1

    # estimate autocorrelation time, number of independent samples
    tau = round(1 / dx ** 2)
    n_independent_samples = n_samples // tau - 1

    d_log_q = np.random.uniform(-1.0, 1.0)  # arbitrary offset added to log_q

    if dist == "normal":
        log_q = lambda x: -(x ** 2) / 2 + d_log_q
        target_samples = np.random.normal(0, 1, size=(n_independent_samples,))
    else:
        log_q = lambda x: d_log_q if -1 < x < 1 else -float("inf")
        target_samples = np.random.uniform(-1, 1, size=(n_independent_samples,))

    sampler = RWMH1D(log_q, dx)
    x_0 = np.random.uniform(-1.0, 1.0)
    rw_samples = sampler.sample_chain(x_0, n_samples)

    decorrelated_rw_samples = rw_samples[tau::tau]

    _, pvalue = ks_2samp(decorrelated_rw_samples, target_samples)

    assert pvalue >= 0.01
