from dataclasses import dataclass
from typing import Any, Callable, Generic, List, NewType, Optional, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray
from scipy.stats import entropy

from timemachine.md.moves import MixtureOfMoves, MonteCarloMove
from timemachine.utils import batches, not_ragged

Replica = TypeVar("Replica")
Samples = TypeVar("Samples")

StateIdx = NewType("StateIdx", int)
ReplicaIdx = NewType("ReplicaIdx", int)

PRNGKeyArray = Any


class NeighborSwapMove(MonteCarloMove[List[Replica]]):
    """Move that attempts to swap replicas at a fixed pair of states."""

    def __init__(self, log_q: Callable[[Replica, StateIdx], float], s_a: StateIdx, s_b: StateIdx):
        super().__init__()
        self.log_q = log_q
        self.s_a = s_a
        self.s_b = s_b

    def propose(self, state: List[Replica]) -> Tuple[List[Replica], float]:
        s_a = self.s_a
        s_b = self.s_b
        state_ = list(state)
        state_[s_a], state_[s_b] = state[s_b], state[s_a]
        proposed_state = state_

        r_a = state[s_a]
        r_b = state[s_b]
        log_q_diff = self.log_q(r_a, s_b) + self.log_q(r_b, s_a) - self.log_q(r_a, s_a) - self.log_q(r_b, s_b)

        log_acceptance_probability = np.minimum(log_q_diff, 0.0)

        return proposed_state, log_acceptance_probability


@jax.jit
def _run_neighbor_swaps(
    replica_idx_by_state: Array,
    neighbor_pairs: Array,
    log_q_kl: Array,
    pair_idxs: Array,
    uniform_samples: Array,
) -> Tuple[Array, Array, Array]:
    """Efficient implementation of a batch of neighbor swap moves.

    Conceptually equivalent to

    .. code-block:: python
        MixtureOfMoves([NeighborSwapMove(log_q, s_a, s_b) for s_a, s_b in neighbor_pairs])

    but implemented in JAX for performance.

    Parameters
    ----------
    replica_idx_by_state : Array
        (n_states,) array of replica indices by state

    neighbor_pairs : Array
        (n_pairs, 2) array representing allowed swaps

    log_q_kl : Array
        (n_replicas, n_states) array with the (r, s) element giving the log unnormalized probability of replica r in state s

    pair_idxs : Array
        (n_swap_attempts,) array of indices of pairs for which to attempt swap moves

    uniform_samples : Array
        (n_swap_attempts,) array of random samples drawn from uniform(0, 1)

    Returns
    -------
    Tuple[Array, Array, Array]
        Final replica_idx_by_state, number of proposals by neighbor pair, number of accepted moves by neighbor pair
    """

    def run_neighbor_swap(
        carry: Tuple[Array, Array, Array], pair_idx_and_uniform_sample: Tuple[Array, Array]
    ) -> Tuple[Tuple[Array, Array, Array], None]:
        replica_idx_by_state, proposed, accepted = carry
        pair_idx, uniform_sample = pair_idx_and_uniform_sample

        s_a, s_b = neighbor_pairs[pair_idx]
        proposed_next = proposed.at[pair_idx].add(1)

        r_a = replica_idx_by_state[s_a]
        r_b = replica_idx_by_state[s_b]

        log_q_before = log_q_kl[r_a, s_a] + log_q_kl[r_b, s_b]
        log_q_after = log_q_kl[r_a, s_b] + log_q_kl[r_b, s_a]

        log_q_diff = log_q_after - log_q_before

        log_acceptance_probability = jnp.minimum(log_q_diff, 0.0)
        acceptance_probability = jnp.exp(log_acceptance_probability)
        is_accepted = uniform_sample < acceptance_probability

        def accept():
            replica_idx_by_state_next = replica_idx_by_state.at[s_a].set(r_b).at[s_b].set(r_a)
            accepted_next = accepted.at[pair_idx].add(1)
            return (replica_idx_by_state_next, proposed_next, accepted_next)

        def reject():
            return (replica_idx_by_state, proposed_next, accepted)

        result = jax.lax.cond(is_accepted, accept, reject)

        return (result, None)

    n_pairs, _ = neighbor_pairs.shape
    init = (replica_idx_by_state, jnp.zeros(n_pairs, jnp.uint32), jnp.zeros(n_pairs, jnp.uint32))

    (replica_idx_by_state, proposed, accepted), _ = jax.lax.scan(run_neighbor_swap, init, (pair_idxs, uniform_samples))

    return replica_idx_by_state, proposed, accepted


@dataclass(frozen=True)
class HREX(Generic[Replica]):
    replicas: List[Replica]
    replica_idx_by_state: List[ReplicaIdx]

    @classmethod
    def from_replicas(cls, replicas: Sequence[Replica]) -> "HREX":
        return HREX(list(replicas), [ReplicaIdx(i) for i, _ in enumerate(replicas)])

    def sample_replicas(
        self,
        sample_replica: Callable[[Replica, StateIdx], Samples],
        replica_from_samples: Callable[[Samples], Replica],
    ) -> Tuple["HREX[Replica]", List[Samples]]:
        samples_by_state = [sample_replica(replica, state_idx) for state_idx, replica in self.state_replica_pairs]
        replicas_by_state = [replica_from_samples(samples) for samples in samples_by_state]

        replicas = list(self.replicas)
        for state_idx, replica in enumerate(replicas_by_state):
            replica_idx = self.replica_idx_by_state[state_idx]
            replicas[replica_idx] = replica

        return HREX(replicas, self.replica_idx_by_state), samples_by_state

    def attempt_neighbor_swaps(
        self,
        neighbor_pairs: Sequence[Tuple[StateIdx, StateIdx]],
        log_q: Callable[[ReplicaIdx, StateIdx], float],
        n_swap_attempts: int,
    ) -> Tuple["HREX[Replica]", List[Tuple[int, int]]]:
        """Run a batch of swap attempts.

        See :py:meth:`attempt_neighbor_swaps_fast` for a more efficient implementation with a similar signature.
        Note that these methods do not generate identical random sequences.

        Parameters
        ----------
        neighbor_pairs : Sequence[Tuple[StateIdx, StateIdx]]
            pairs of states between which to attempt swaps

        log_q : Callable[[ReplicaIdx, StateIdx], float]
            function to compute the log unnormalized probability of a given replica in a given state

        n_swap_attempts : int
            number of individual swap attempts, each between a randomly-selected neighbor pair

        Returns
        -------
        Tuple[HREX[Replica], List[Tuple[int, int]]]
            Updated HREX state, list of (accepted, proposed) counts by neighbor pair
        """
        move = MixtureOfMoves([NeighborSwapMove(log_q, s_a, s_b) for s_a, s_b in neighbor_pairs])

        replica_idx_by_state = move.move_n(list(self.replica_idx_by_state), n_swap_attempts)

        fraction_accepted_by_pair = list(zip(move.n_accepted_by_move, move.n_proposed_by_move))

        return HREX(self.replicas, replica_idx_by_state), fraction_accepted_by_pair

    def attempt_neighbor_swaps_fast(
        self, neighbor_pairs: Sequence[Tuple[StateIdx, StateIdx]], log_q_kl: ArrayLike, n_swap_attempts: int, seed: int
    ) -> Tuple["HREX[Replica]", List[Tuple[int, int]]]:
        """Run a batch of swap attempts.

        See :py:meth:`attempt_neighbor_swaps` for a (typically slower) reference version.
        Note that these methods do not generate identical random sequences.

        Parameters
        ----------
        neighbor_pairs : Sequence[Tuple[StateIdx, StateIdx]]
            pairs of states between which to attempt swaps

        log_q_kl : ArrayLike
            (n_replicas, n_states) array with the (r, s) element giving the log unnormalized probability of replica r in state s

        n_swap_attempts : int
            number of individual swap attempts, each between a randomly-selected neighbor pair

        seed : int
            PRNG seed

        Returns
        -------
        Tuple[HREX[Replica], List[Tuple[int, int]]]
            Updated HREX state, list of (accepted, proposed) counts by neighbor pair
        """

        key = jax.random.key(seed)
        key, subkey = jax.random.split(key)
        pair_idxs = jax.random.choice(subkey, len(neighbor_pairs), (n_swap_attempts,))
        uniform_samples = jax.random.uniform(key, (n_swap_attempts,))

        replica_idx_by_state_, proposed, accepted = _run_neighbor_swaps(
            jnp.asarray(self.replica_idx_by_state),
            jnp.asarray(neighbor_pairs),
            jnp.asarray(log_q_kl),
            pair_idxs,
            uniform_samples,
        )

        replica_idx_by_state = replica_idx_by_state_.tolist()
        fraction_accepted_by_pair = list(zip(accepted.tolist(), proposed.tolist()))

        return HREX(self.replicas, replica_idx_by_state), fraction_accepted_by_pair

    @property
    def state_replica_pairs(self) -> List[Tuple[StateIdx, Replica]]:
        return [(StateIdx(i), self.replicas[replica_idx]) for i, replica_idx in enumerate(self.replica_idx_by_state)]


def get_normalized_kl_divergence(replica_idx_by_state_by_iter: Sequence[Sequence[ReplicaIdx]]) -> float:
    r"""Heuristic for the how uniformly windows were sampled within an HREX simulation.
    Based on Eq. 5 from [1], but after summing up the divergences of each state, take the mean
    so that it is possible to compare the values between simulations of different numbers of windows.

    A value closer to 0.0 indicates more uniform sampling.

    References
    ----------
    [1]: https://doi.org/10.1021/acs.jctc.0c00660, https://chemrxiv.org/engage/chemrxiv/article-details/60c74d2e702a9b007018b7ef

    Notes
    -----
    * Avoid having to generate a uniform distribution by using the identity
      $\sum_i p_i \log (p_i / u_i) =  \sum_i p_i \log p_i + \log N = -H_p + \log N$, where $H_p$ is the entropy of $p$
    """
    cumulative_counts = get_cumulative_replica_state_counts(replica_idx_by_state_by_iter)
    n_iters, n_states, n_replicas = cumulative_counts.shape
    count_by_replica_by_state = cumulative_counts[-1]
    fraction_by_replica_by_state = count_by_replica_by_state / n_iters

    return -np.mean(entropy(fraction_by_replica_by_state, axis=0)) + np.log(n_states)


def get_cumulative_replica_state_counts(replica_idx_by_state_by_iter: Sequence[Sequence[ReplicaIdx]]) -> NDArray:
    """Given a mapping of state index to replica index by iteration, returns an array of cumulative counts by iteration, replica, and state.

    Returns
    -------
    NDArray
        (iter, state, replica) -> cumulative occupancy count: int
    """
    replica_idx_by_state_by_iter_ = np.array(replica_idx_by_state_by_iter)  # (iter, state) -> replica
    _, n_states = replica_idx_by_state_by_iter_.shape
    states = np.arange(n_states)
    replica_in_state = replica_idx_by_state_by_iter_[:, :, None] == states  # (iter, state, replica) -> bool
    cumulative_count = np.cumsum(replica_in_state.astype(int), axis=0)  # (iter, state, replica) -> int
    return cumulative_count


def estimate_transition_matrix(replica_idx_by_state_by_iter: Sequence[Sequence[ReplicaIdx]]) -> NDArray:
    """Given a mapping of state index to replica index by iteration, returns an estimate of the transition matrix.

    The (i, j) element in the returned matrix represents the probability for a replica in state j to transition to state
    j in a single permutation move (consisting of many neighbor swap attempts).

    The resulting matrix is:
      * not necessarily symmetric
      * "doubly stochastic", i.e., all rows and columns sum to 1

    Returns
    -------
    NDArray
        (from state, to state) -> transition probability: float
    """
    replica_idx_by_state_by_iter_ = np.array(replica_idx_by_state_by_iter)  # (iter, state) -> replica
    n_iters, _ = replica_idx_by_state_by_iter_.shape

    # transition_by_iter: (iter, to state, from state) -> bool
    transition_by_iter = replica_idx_by_state_by_iter_[:-1, None, :] == replica_idx_by_state_by_iter_[1:, :, None]

    transition_count = np.sum(transition_by_iter, axis=0)  # (to state, from state) -> int
    transition_probability = transition_count / (n_iters - 1)  # (to state, from state) -> float

    return transition_probability


def estimate_relaxation_time(transition_matrix: NDArray) -> float:
    """Estimate the relaxation time of permutation moves as a function of the second-largest eigenvalue of the
    (symmetrized) transition matrix.

    Notes
    -----
    * Following [1] (section III.C.1), we assume that forward and time-reversed transitions are equally likely in the
      limit of infinite time, and symmetrize the transition matrix to ensure purely real eigenvalues.

    References
    ----------
    [1]: http://dx.doi.org/10.1063/1.3660669, https://arxiv.org/abs/1105.5749
    """

    assert np.allclose(np.sum(transition_matrix, axis=0), 1.0), "columns of transition matrix must sum to 1"

    def symmetrize(x):
        return (x + x.T) / 2.0

    transition_matrix_symmetric = symmetrize(transition_matrix)
    eigvals_ascending = np.linalg.eigvalsh(transition_matrix_symmetric)
    mu_2 = eigvals_ascending[-2]  # second-largest eigenvalue
    return 1 / (1 - mu_2)


def get_samples_by_iter_by_replica(
    samples_by_state_by_iter: List[List[Samples]], replica_idx_by_state_by_iter: List[List[ReplicaIdx]]
) -> List[List[Samples]]:
    """Permute and reshape samples returned by :py:func:`run_hrex` having shape

        (hrex_iter, state_idx) -> samples

    to a more convenient form for analyzing the trajectories of individual replicas

        (replica_idx, hrex_iter) -> samples
    """

    assert len(samples_by_state_by_iter) == len(replica_idx_by_state_by_iter)
    assert not_ragged(samples_by_state_by_iter)
    assert not_ragged(replica_idx_by_state_by_iter)

    samples_by_replica_by_iter = [
        [samples_by_state[state_idx] for state_idx in np.argsort(replica_idx_by_state)]
        for samples_by_state, replica_idx_by_state in zip(samples_by_state_by_iter, replica_idx_by_state_by_iter)
    ]

    samples_by_iter_by_replica = [list(xs) for xs in zip(*samples_by_replica_by_iter)]  # transpose

    return samples_by_iter_by_replica


@dataclass
class HREXDiagnostics:
    replica_idx_by_state_by_iter: List[List[ReplicaIdx]]
    fraction_accepted_by_pair_by_iter: List[List[Tuple[int, int]]]

    @property
    def cumulative_swap_acceptance_rates(self) -> NDArray:
        n_accepted, n_proposed = np.moveaxis(np.array(self.fraction_accepted_by_pair_by_iter), -1, 0)
        return np.cumsum(n_accepted, axis=0) / np.cumsum(n_proposed, axis=0)

    @property
    def cumulative_replica_state_counts(self) -> NDArray:
        return get_cumulative_replica_state_counts(self.replica_idx_by_state_by_iter)

    @property
    def transition_matrix(self) -> NDArray:
        return estimate_transition_matrix(self.replica_idx_by_state_by_iter)

    @property
    def relaxation_time(self) -> float:
        return estimate_relaxation_time(self.transition_matrix)

    @property
    def normalized_kl_divergence(self) -> float:
        return get_normalized_kl_divergence(self.replica_idx_by_state_by_iter)


def get_swap_attempts_per_iter_heuristic(n_states: int) -> int:
    """Heuristic for number of swap attempts per iteration derived from [1].

    References
    ----------
    [1]: http://dx.doi.org/10.1063/1.3660669, https://arxiv.org/abs/1105.5749
    """

    return n_states**3


def run_hrex(
    replicas: Sequence[Replica],
    sample_replica: Callable[[Replica, StateIdx, int], Samples],
    replica_from_samples: Callable[[Samples], Replica],
    neighbor_pairs: Sequence[Tuple[StateIdx, StateIdx]],
    get_log_q: Callable[[List[Replica]], ArrayLike | Callable[[ReplicaIdx, StateIdx], float]],
    n_samples: int,
    n_samples_per_iter: int,
    seed: int,
    n_swap_attempts_per_iter: Optional[int] = None,
) -> Tuple[List[List[Samples]], HREXDiagnostics]:
    r"""Sample from a sequence of states using Hamiltonian Replica EXchange (HREX).

    This implementation uses a method described in [1] (in section III.B.2) to generate effectively uncorrelated
    permutations by attempting many consecutive nearest-neighbor swap moves. By default, the number of swap moves is
    determined as a function of the number of states (:math:`K`) as :math`N_{\text{swaps}} = K^3`, a heuristic also
    described in [1].

    References
    ----------
    [1]: http://dx.doi.org/10.1063/1.3660669

    Parameters
    ----------
    replicas: sequence of Replica
        Sequence of initial states of each replica

    sample_replica: (Replica, StateIdx, n_samples: int) -> Samples
        Local sampling function. Should return n_samples samples from the given replica and state

    replica_from_samples: Samples -> Replica
        Function that returns a replica state given a sequence of local samples. This is used to update the state of
        individual replicas following local sampling.

    neighbor_pairs: sequence of (StateIdx, StateIdx)
        Pairs of states for which to attempt swap moves

    get_log_q: (sequence of Replica) -> (((ReplicaIdx, StateIdx) -> float) or (sequence of Replica -> array))
        Function that, given a list of replicas, returns either:

        1. a (n_replicas, n_states) array with the (r, s) element giving the log unnormalized probability of replica r in state s, or
        2. a function from replica-state pairs to log unnormalized probability

    n_samples: int
        Total number of local samples (e.g. MD frames)

    n_samples_per_iter: int
        Number of local samples (e.g. MD frames) per HREX iteration

    seed: int
        PRNG seed

    n_swap_attempts_per_iter: int or None, optional
        Number of neighbor swaps to attempt per iteration. Default is given by :py:func:`get_swap_attempts_per_iter_heuristic`.

    Returns
    -------
    List[List[Samples]]
        samples grouped by state and iteration

    HREXDiagnostics
        HREX statistics (e.g. swap rates, replica-state distribution)
    """

    n_replicas = len(replicas)

    if n_swap_attempts_per_iter is None:
        n_swap_attempts_per_iter = get_swap_attempts_per_iter_heuristic(n_replicas)

    hrex = HREX.from_replicas(replicas)

    samples_by_state_by_iter: List[List[Samples]] = []
    replica_idx_by_state_by_iter: List[List[ReplicaIdx]] = []
    fraction_accepted_by_pair_by_iter: List[List[Tuple[int, int]]] = []

    for iteration, n_samples_batch in enumerate(batches(n_samples, n_samples_per_iter)):
        log_q = get_log_q(hrex.replicas)
        log_q_kl = (
            jnp.array([[log_q(ReplicaIdx(r), StateIdx(s)) for s in range(n_replicas)] for r in range(n_replicas)])
            if callable(log_q)
            else log_q
        )

        hrex, fraction_accepted_by_pair = hrex.attempt_neighbor_swaps_fast(
            neighbor_pairs, log_q_kl, n_swap_attempts_per_iter, seed + iteration
        )

        sample_replica_ = lambda replica, state_idx: sample_replica(replica, state_idx, n_samples_batch)
        hrex, samples_by_state = hrex.sample_replicas(sample_replica_, replica_from_samples)

        fraction_accepted_by_pair_by_iter.append(fraction_accepted_by_pair)
        replica_idx_by_state_by_iter.append(hrex.replica_idx_by_state)
        samples_by_state_by_iter.append(samples_by_state)

    return samples_by_state_by_iter, HREXDiagnostics(replica_idx_by_state_by_iter, fraction_accepted_by_pair_by_iter)
