"""
Assert accurate estimates for free energy differences between 1D Gaussians using perfect maps
"""

from dataclasses import dataclass

import numpy as np
import pytest
from pymbar import bar, exp
from pymbar.mbar import MBAR

from timemachine.fe.bar import DG_KEY
from timemachine.maps.estimators import compute_mapped_reduced_work, compute_mapped_u_kn

pytestmark = [pytest.mark.nocuda]


@dataclass
class UnnormalizedGaussian:
    mean: float
    stddev: float
    reduced_free_energy: float

    def _normalized_logpdf(self, x):
        return -((x - self.mean) ** 2) / (self.stddev**2) - np.log(self.stddev * np.sqrt(2 * np.pi))

    def reduced_potential(self, x):
        return -self._normalized_logpdf(x) + self.reduced_free_energy

    def sample(self, n_samples=1000):
        return np.random.randn(n_samples) * self.stddev + self.mean

    @classmethod
    def initialize_randomly(cls):
        return cls(
            mean=np.random.randn(),
            stddev=np.exp(np.random.randn()),  # positive
            reduced_free_energy=np.random.randn(),
        )


def construct_map(a: UnnormalizedGaussian, b: UnnormalizedGaussian):
    shift = b.mean - a.mean
    scale = b.stddev / a.stddev

    def map_fxn(x):
        centered = x - a.mean
        scaled = scale * centered + a.mean
        shifted = scaled + shift

        logdetjacs = np.log(scale) * np.ones_like(x)

        return shifted, logdetjacs

    return map_fxn


def test_one_sided_estimates():
    np.random.seed(2022)

    src_state = UnnormalizedGaussian(mean=0, stddev=1, reduced_free_energy=2)
    src_samples = src_state.sample(1000)
    u_src = src_state.reduced_potential

    dst_states = [UnnormalizedGaussian.initialize_randomly() for _ in range(10)]

    eps = 1e-10

    for dst_state in dst_states:
        u_dst = dst_state.reduced_potential

        map_fxn = construct_map(src_state, dst_state)

        # should have pretty high variance
        naive_w_F = u_dst(src_samples) - u_src(src_samples)
        assert np.std(naive_w_F) > eps

        # since map_fxn is perfect, variance should be == 0...
        mapped_w_F = compute_mapped_reduced_work(src_samples, u_src, u_dst, map_fxn)
        assert np.std(mapped_w_F) < eps

        # ... and estimated_delta_f should be == exact_delta_f
        estimated_delta_f = exp(mapped_w_F)[DG_KEY]
        exact_delta_f = dst_state.reduced_free_energy - src_state.reduced_free_energy

        np.testing.assert_allclose(estimated_delta_f, exact_delta_f)


def test_two_sided_estimates():
    np.random.seed(2022)

    for _ in range(10):
        state_a = UnnormalizedGaussian.initialize_randomly()
        state_b = UnnormalizedGaussian.initialize_randomly()

        map_fxn = construct_map(state_a, state_b)
        inv_map_fxn = construct_map(state_b, state_a)

        u_a = state_a.reduced_potential
        u_b = state_b.reduced_potential

        x_a = state_a.sample(1000)
        x_b = state_b.sample(500)

        w_F = compute_mapped_reduced_work(x_a, u_a, u_b, map_fxn)
        w_R = compute_mapped_reduced_work(x_b, u_b, u_a, inv_map_fxn)

        # estimated_delta_f = bar(w_F, w_R)[0] #  default solver -> BoundsError: Cannot determine bound on free energy
        estimated_delta_f = bar(w_F, w_R, method="self-consistent-iteration", compute_uncertainty=False)[DG_KEY]

        exact_delta_f = state_b.reduced_free_energy - state_a.reduced_free_energy

        np.testing.assert_allclose(estimated_delta_f, exact_delta_f)


def test_multistate_estimates():
    np.random.seed(2022)

    # define a collection of states
    K = 10
    states = [UnnormalizedGaussian.initialize_randomly() for _ in range(K)]
    u_fxns = [state.reduced_potential for state in states]

    # define invertible maps between all pairs of states, and put these in a container supporting [i, j] indexing
    map_fxns = np.zeros((K, K), dtype=object)
    for i in range(K):
        for j in range(K):
            map_fxns[i, j] = construct_map(states[i], states[j])

    # collect a different number of samples from each state
    N_k = np.random.randint(100, 1000, size=K)
    samples = [state.sample(n) for state, n in zip(states, N_k)]

    # compute MBAR estimate
    u_kn = compute_mapped_u_kn(samples, u_fxns, map_fxns)
    mbar = MBAR(u_kn, N_k)

    exact_f_k = np.array([state.reduced_free_energy for state in states])
    exact_f_k -= exact_f_k[0]

    np.testing.assert_allclose(mbar.f_k, exact_f_k)
