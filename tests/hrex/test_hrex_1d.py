from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, List, Protocol, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy
from numpy.typing import NDArray
from scipy.special import logsumexp

from timemachine.fe.plots import (
    plot_hrex_replica_state_distribution,
    plot_hrex_replica_state_distribution_convergence,
    plot_hrex_replica_state_distribution_heatmap,
    plot_hrex_swap_acceptance_rates_convergence,
    plot_hrex_transition_matrix,
)
from timemachine.md.hrex import HREXDiagnostics, NeighborSwapMove, ReplicaIdx, StateIdx, run_hrex
from timemachine.md.moves import MixtureOfMoves, MonteCarloMove

DEBUG = False

pytestmark = [pytest.mark.nogpu]


class Distribution(Protocol):
    def sample(self, n_samples: int) -> NDArray:
        ...

    def log_q(self, x: float) -> float:
        ...


@dataclass
class Uniform:
    x_1: float
    x_2: float

    def sample(self, n_samples: int) -> NDArray:
        return np.random.uniform(self.x_1, self.x_2, size=(n_samples,))

    def log_q(self, x: float) -> float:
        return 0.0 if self.x_1 < x <= self.x_2 else -np.inf


@dataclass
class GaussianMixture:
    locs: NDArray
    scales: NDArray
    log_weights: NDArray

    def __post_init__(self):
        assert len(self.locs) == len(self.scales) == len(self.log_weights)

    def sample(self, n_samples: int) -> NDArray:
        (n_components,) = self.locs.shape
        probs = np.exp(self.log_weights - logsumexp(self.log_weights))
        components = np.random.choice(n_components, p=probs, size=(n_samples,))
        xs = np.random.normal(self.locs, self.scales, size=(n_samples, n_components))
        return xs[np.arange(n_samples), components]

    def log_q(self, x: float) -> float:
        x_ = np.atleast_1d(np.asarray(x))
        log_q = -((x_[:, None] - self.locs) ** 2) / (2 * self.scales**2)
        return logsumexp(log_q + self.log_weights, axis=1).item()


def gaussian(loc: float, scale: float, log_weight: float = 0.0) -> Distribution:
    return GaussianMixture(np.array([loc]), np.array([scale]), np.array([log_weight]))


@dataclass
class LocalMove(MonteCarloMove[float]):
    def __init__(self, proposal: Callable[[float], Distribution], target: Distribution):
        super().__init__()
        self.proposal = proposal
        self.target = target

    def propose(self, x: float) -> Tuple[float, float]:
        x_p = self.proposal(x).sample(1).item()
        log_q_diff = self.target.log_q(x_p) - self.target.log_q(x)
        log_acceptance_probability = np.minimum(log_q_diff, 0.0)
        return x_p, log_acceptance_probability


def run_hrex_with_local_proposal(
    states: Sequence[Distribution],
    initial_replicas: Sequence[float],
    proposal: Callable[[float], Distribution],
    seed: int,
    n_samples=10_000,
    n_samples_per_iter=20,
):
    assert len(states) == len(initial_replicas)

    state_idxs = [StateIdx(i) for i, _ in enumerate(states)]
    neighbor_pairs = list(zip(state_idxs, state_idxs[1:]))

    # Add (0, 0) to the list of neighbor pairs considered for swap moves to ensure that performing a fixed number of
    # neighbor swaps is aperiodic in cases where swap acceptance rates approach 100%
    neighbor_pairs = [(StateIdx(0), StateIdx(0))] + neighbor_pairs

    def sample_replica(replica: float, state_idx: StateIdx, n_samples: int) -> List[float]:
        """Sample replica using local moves in the specified state"""
        move = LocalMove(proposal, states[state_idx])
        samples = move.sample_chain(replica, n_samples)
        return samples

    def get_log_q(replicas: Sequence[float]) -> NDArray:
        log_q_kl = np.array(
            [[states[state_idx].log_q(replicas[replica_idx]) for state_idx in state_idxs] for replica_idx in state_idxs]
        )
        return log_q_kl

    def replica_from_samples(xs: List[float]) -> float:
        return xs[-1]

    samples_by_state_by_iter, diagnostics = run_hrex(
        initial_replicas,
        sample_replica,
        replica_from_samples,
        neighbor_pairs,
        get_log_q,
        seed=seed,
        n_samples=n_samples,
        n_samples_per_iter=n_samples_per_iter,
    )

    # Remove stats for (0, 0) pair
    diagnostics = replace(
        diagnostics,
        fraction_accepted_by_pair_by_iter=[
            fraction_accepted_by_pair[1:] for fraction_accepted_by_pair in diagnostics.fraction_accepted_by_pair_by_iter
        ],
    )

    if DEBUG:
        plot_hrex_diagnostics(diagnostics)
        plt.show()

    return samples_by_state_by_iter, diagnostics


@pytest.mark.parametrize("seed", range(5))
def test_hrex_different_distributions_same_free_energy(seed):
    np.random.seed(seed)

    locs = [0.0, 0.5, 1.0]
    states = [gaussian(loc, 0.3) for loc in locs]
    initial_replicas = locs

    proposal_radius = 0.1
    proposal = lambda x: gaussian(x, proposal_radius)

    samples_by_state_by_iter, diagnostics = run_hrex_with_local_proposal(states, initial_replicas, proposal, seed)

    samples_by_state = np.concatenate(samples_by_state_by_iter, axis=1)

    # KS test assumes independent samples
    # Use a rough estimate of autocorrelation time to subsample correlated MCMC samples
    tau = round(1 / proposal_radius**2)

    (n_samples,) = samples_by_state[0].shape

    ks_pvalues = [
        scipy.stats.ks_2samp(samples[tau::tau], state.sample(n_samples)).pvalue
        for samples, state in zip(samples_by_state, states)
    ]

    np.testing.assert_array_less(0.005, ks_pvalues)

    final_swap_acceptance_rates = diagnostics.cumulative_swap_acceptance_rates[-1]
    np.testing.assert_array_less(0.2, final_swap_acceptance_rates)

    # Swap acceptance rates should be approximately equal between pairs
    np.testing.assert_array_less(np.abs(final_swap_acceptance_rates - final_swap_acceptance_rates.mean()), 0.02)

    n_iters = diagnostics.cumulative_replica_state_counts.shape[0]
    final_replica_state_density = diagnostics.cumulative_replica_state_counts[-1] / n_iters

    # Fraction of time spent in each state for each replica should be close to uniform
    np.testing.assert_array_less(np.abs(final_replica_state_density - np.mean(final_replica_state_density)), 0.25)


@pytest.mark.parametrize("seed", range(5))
def test_hrex_same_distributions_different_free_energies(seed):
    np.random.seed(seed)

    states = [gaussian(0.0, 0.3, log_weight) for log_weight in [-1.0, 0.0, 1.0]]
    initial_replicas = [0.0] * len(states)

    proposal_radius = 0.1
    proposal = lambda x: gaussian(x, proposal_radius)

    samples_by_state_by_iter, diagnostics = run_hrex_with_local_proposal(states, initial_replicas, proposal, seed)

    samples_by_state = np.concatenate(samples_by_state_by_iter, axis=1)

    # KS test assumes independent samples
    # Use a rough estimate of autocorrelation time to subsample correlated MCMC samples
    tau = round(1 / proposal_radius**2)

    (n_samples,) = samples_by_state[0].shape

    ks_pvalues = [
        scipy.stats.ks_2samp(samples[tau::tau], state.sample(n_samples)).pvalue
        for samples, state in zip(samples_by_state, states)
    ]

    np.testing.assert_array_less(0.01, ks_pvalues)
    assert np.all(diagnostics.cumulative_swap_acceptance_rates == 1.0)  # difference in log(q) for swaps is always zero

    n_iters = diagnostics.cumulative_replica_state_counts.shape[0]
    final_replica_state_density = diagnostics.cumulative_replica_state_counts[-1] / n_iters

    # Fraction of time spent in each state for each replica should be close to uniform
    np.testing.assert_array_less(np.abs(final_replica_state_density - np.mean(final_replica_state_density)), 0.2)


@pytest.mark.parametrize("seed", range(5))
def test_hrex_gaussian_mixture(seed):
    """Use HREX to sample from a mixture of two gaussians with ~zero overlap."""

    np.random.seed(seed)

    states = [
        GaussianMixture(np.array([0.0, 1.0]), scales=np.array([0.1, 0.1]), log_weights=np.array([0.0, 0.0])),
        gaussian(0.5, 0.5),
    ]

    # start replicas at x=0
    initial_replicas = [0.0, 0.0]

    proposal_radius = 0.1
    proposal = lambda x: gaussian(x, proposal_radius)

    samples_by_state_by_iter, diagnostics = run_hrex_with_local_proposal(states, initial_replicas, proposal, seed)

    samples_by_state = np.concatenate(samples_by_state_by_iter, axis=1)
    hrex_samples = samples_by_state[0]  # samples from gaussian mixture

    (n_samples,) = hrex_samples.shape

    local_samples_ = LocalMove(proposal, states[0]).sample_chain(initial_replicas[0], n_samples)
    local_samples = np.array(local_samples_)

    # HREX should sample the energy well at x=1
    assert np.any(hrex_samples > 1.0)

    target_samples = states[0].sample(n_samples)

    if DEBUG:
        plt.figure()
        hist = partial(plt.hist, density=True, bins=50, alpha=0.7)
        hist(target_samples, label="target")
        hist(hrex_samples, label="hrex")
        hist(local_samples, label="local")
        plt.legend()
        plt.show()

    # KS test assumes independent samples
    # Use a rough estimate of autocorrelation time to subsample correlated MCMC samples
    tau = round(1 / proposal_radius**2)

    def compute_ks_pvalue(samples):
        return scipy.stats.ks_2samp(samples[tau::tau], target_samples).pvalue

    assert compute_ks_pvalue(local_samples) == pytest.approx(0.0, abs=1e-10)  # local sampling alone is insufficient
    assert compute_ks_pvalue(hrex_samples) > 0.005

    final_swap_acceptance_rates = diagnostics.cumulative_swap_acceptance_rates[-1]
    assert final_swap_acceptance_rates[0] > 0.2


def plot_hrex_diagnostics(diagnostics: HREXDiagnostics):
    plot_hrex_swap_acceptance_rates_convergence(diagnostics.cumulative_swap_acceptance_rates)
    plot_hrex_transition_matrix(diagnostics.transition_matrix)
    plot_hrex_replica_state_distribution(diagnostics.cumulative_replica_state_counts)
    plot_hrex_replica_state_distribution_heatmap(diagnostics.cumulative_replica_state_counts)
    plot_hrex_replica_state_distribution_convergence(diagnostics.cumulative_replica_state_counts)


@pytest.mark.parametrize("num_states", [5])
@pytest.mark.parametrize("swaps", [50_000])
@pytest.mark.parametrize("seed", range(5))
def test_batched_mixture_of_moves(num_states, seed, swaps):
    np.random.seed(seed)
    replicas = sorted(np.random.uniform(size=num_states))
    states = [gaussian(loc, 0.3) for loc in replicas]
    state_idxs = [StateIdx(i) for i, _ in enumerate(states)]
    replica_idxs = [ReplicaIdx(i) for i, _ in enumerate(replicas)]
    neighbor_pairs = list(zip(state_idxs, state_idxs[1:]))

    log_q_matrix = np.array(
        [[states[state_idx].log_q(replicas[replica_idx]) for state_idx in state_idxs] for replica_idx in state_idxs]
    )

    def log_q(replica_idx: ReplicaIdx, state_idx: StateIdx) -> float:
        return log_q_matrix[replica_idx, state_idx]

    neighbor_swaps = [NeighborSwapMove(log_q, s_a, s_b) for s_a, s_b in neighbor_pairs]

    def run_sequential_swaps(replica_idxs):
        move = MixtureOfMoves(neighbor_swaps)
        for _ in range(swaps):
            replica_idxs = move.move(replica_idxs)
        return move

    move_seq = run_sequential_swaps(replica_idxs)

    move_batched = MixtureOfMoves(neighbor_swaps)
    np.random.seed(seed)
    _ = move_batched.move_n(replica_idxs, swaps)

    assert move_seq.n_accepted_by_move == move_batched.n_accepted_by_move
    assert move_seq.n_proposed_by_move == move_batched.n_proposed_by_move
