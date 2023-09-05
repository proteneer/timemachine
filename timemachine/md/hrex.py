from dataclasses import dataclass
from typing import Callable, Generic, List, NewType, Optional, Sequence, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from timemachine.md.moves import MetropolisHastingsMove, MixtureOfMoves
from timemachine.utils import batches

_Replica = TypeVar("_Replica")

StateIdx = NewType("StateIdx", int)
ReplicaIdx = NewType("ReplicaIdx", int)


@dataclass
class NeighborSwapMove(MetropolisHastingsMove[List[_Replica]]):
    def __init__(self, log_q: Callable[[_Replica, StateIdx], float], s_a: StateIdx, s_b: StateIdx):
        super().__init__()
        self.log_q = log_q
        self.s_a = s_a
        self.s_b = s_b

    def propose_with_log_q_diff(self, state: List[_Replica]) -> Tuple[List[_Replica], float]:
        s_a = self.s_a
        s_b = self.s_b
        state_ = list(state)
        state_[s_a], state_[s_b] = state[s_b], state[s_a]
        proposed_state = state_

        r_a = state[s_a]
        r_b = state[s_b]
        log_q_diff = self.log_q(r_a, s_b) + self.log_q(r_b, s_a) - self.log_q(r_a, s_a) - self.log_q(r_b, s_b)

        return proposed_state, log_q_diff


_Samples = TypeVar("_Samples")


@dataclass
class HREX(Generic[_Replica]):
    replicas: List[_Replica]
    replica_idx_by_state: List[ReplicaIdx]

    @classmethod
    def from_replicas(cls, replicas: Sequence[_Replica]) -> "HREX":
        return HREX(list(replicas), [ReplicaIdx(i) for i, _ in enumerate(replicas)])

    def sample_replicas(
        self,
        sample_replica: Callable[[_Replica, StateIdx], _Samples],
        replica_from_samples: Callable[[_Samples], _Replica],
    ) -> Tuple["HREX[_Replica]", List[_Samples]]:

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
    ) -> Tuple["HREX[_Replica]", List[Tuple[int, int]]]:

        move = MixtureOfMoves([NeighborSwapMove(log_q, s_a, s_b) for s_a, s_b in neighbor_pairs])

        replica_idx_by_state = list(self.replica_idx_by_state)
        for _ in range(n_swap_attempts):
            replica_idx_by_state = move.move(replica_idx_by_state)

        fraction_accepted_by_pair = list(zip(move.n_accepted_by_move, move.n_proposed_by_move))

        return HREX(self.replicas, replica_idx_by_state), fraction_accepted_by_pair

    @property
    def state_replica_pairs(self) -> List[Tuple[StateIdx, _Replica]]:
        return [(StateIdx(i), self.replicas[replica_idx]) for i, replica_idx in enumerate(self.replica_idx_by_state)]


def get_cumulative_replica_state_counts(replica_idx_by_state_by_iter: Sequence[Sequence[ReplicaIdx]]) -> NDArray:
    """Given a mapping of state index to replica index by iteration, returns an array of cumulative counts by iteration, replica, and state.

    Returns
    -------
    NDArray
        (iter, replica, state) -> cumulative occupancy count: int
    """
    replica_idx_by_state_by_iter_ = np.array(replica_idx_by_state_by_iter)  # (iter, state) -> replica
    _, n_states = replica_idx_by_state_by_iter_.shape
    states = np.arange(n_states)
    replica_in_state = replica_idx_by_state_by_iter_[:, :, None] == states  # (iter, replica, state) -> bool
    cumulative_count = np.cumsum(replica_in_state.astype(int), axis=0)  # (iter, replica, state) -> int
    return cumulative_count


def estimate_transition_matrix(replica_idx_by_state_by_iter: Sequence[Sequence[ReplicaIdx]]) -> NDArray:
    """Given a mapping of state index to replica index by iteration, returns an estimate of the transition matrix.

    Returns
    -------
    NDArray
        (state, state) -> transition_rate: float
    """
    replica_idx_by_state_by_iter_ = np.array(replica_idx_by_state_by_iter)  # (iter, state) -> replica index
    n_iters, _ = replica_idx_by_state_by_iter_.shape

    # transition_by_iter: (n_iters, n_states, n_states) -> bool
    transition_by_iter = replica_idx_by_state_by_iter_[:-1, None, :] == replica_idx_by_state_by_iter_[1:, :, None]

    transition_count = np.sum(transition_by_iter, axis=0)  # (to state, from state) -> int
    transition_rate = transition_count / n_iters  # (to state, from state) -> float

    return transition_rate


def estimate_relaxation_time(transition_matrix: NDArray) -> float:
    """Estimate the relaxation time of permutation moves (in number of iterations) as a function of the second-largest
    eigenvalue of the transition matrix."""

    eigvals_ascending = np.linalg.eigvals(transition_matrix)
    mu_2 = eigvals_ascending[-2]  # second-largest eigenvalue
    return 1 / (1 - mu_2)


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
    def relaxation_time(self):
        return estimate_relaxation_time(self.transition_matrix)


def get_swap_attempts_per_iter_heuristic(n_states: int) -> int:
    """Heuristic for number of swap attempts per iteration derived from [1].

    References
    ----------
    [1]: http://dx.doi.org/10.1063/1.3660669, https://arxiv.org/abs/1105.5749
    """

    return n_states ** 4


def run_hrex(
    replicas: Sequence[_Replica],
    sample_replica: Callable[[_Replica, StateIdx, int], _Samples],
    replica_from_samples: Callable[[_Samples], _Replica],
    neighbor_pairs: Sequence[Tuple[StateIdx, StateIdx]],
    get_log_q_fn: Callable[[List[_Replica]], Callable[[ReplicaIdx, StateIdx], float]],
    n_samples: int,
    n_samples_per_iter: int,
    n_swap_attempts_per_iter: Optional[int] = None,
) -> Tuple[List[List[_Samples]], HREXDiagnostics]:
    r"""Sample from a sequence of states using Hamiltonian Replica EXchange (HREX).

    This implementation uses a method described in [1] (in section III.B.2) to generate effectively uncorrelated
    permutations by attempting many consecutive nearest-neighbor swap moves. By default, the number of swap moves is
    determined as a function of the number of states (:math:`K`) as :math`N_{\text{swaps}} = K^4`, a heuristic also
    described in [1].

    References
    ----------
    [1]: http://dx.doi.org/10.1063/1.3660669

    Parameters
    ----------
    replicas: sequence of _Replica
        Sequence of initial states of each replica

    sample_replica: (_Replica, StateIdx, n_samples: int) -> _Samples
        Local sampling function. Should return n_samples samples from the given replica and state

    replica_from_samples: _Samples -> _Replica
        Function that returns a replica state given a sequence of local samples. This is used to update the state of
        individual replicas following local sampling.

    neighbor_pairs: sequence of (StateIdx, StateIdx)
        Pairs of states for which to attempt swap moves

    get_log_q_fn: sequence of _Replica -> ((ReplicaIdx, StateIdx) -> float)
        Function that returns a function from replica-state pairs to log unnormalized probability. Note that this is
        equivalent to the simpler signature (_Replica, StateIdx) -> float; the "curried" form here is to allow for the
        implementation to compute the full matrix as a batch operation when this is more efficient.

    n_samples: int
        Total number of local samples (e.g. MD frames)

    n_samples_per_iter: int
        Number of local samples (e.g. MD frames) per HREX iteration

    n_swap_attempts_per_iter: int or None, optional
        Number of neighbor swaps to attempt per iteration. Default is given by :py:func:`get_swap_attempts_per_iter_heuristic`.

    Returns
    -------
    List[List[_Samples]]
        samples grouped by state and iteration

    HREXDiagnostics
        HREX statistics (e.g. swap rates, replica-state distribution)
    """

    n_replicas = len(replicas)

    if n_swap_attempts_per_iter is None:
        n_swap_attempts_per_iter = get_swap_attempts_per_iter_heuristic(n_replicas)

    hrex = HREX.from_replicas(replicas)

    samples_by_state_by_iter: List[List[_Samples]] = []
    replica_idx_by_state_by_iter: List[List[ReplicaIdx]] = []
    fraction_accepted_by_pair_by_iter: List[List[Tuple[int, int]]] = []

    for n_samples_batch in batches(n_samples, n_samples_per_iter):
        sample_replica_ = lambda replica, state_idx: sample_replica(replica, state_idx, n_samples_batch)
        hrex, samples_by_state = hrex.sample_replicas(sample_replica_, replica_from_samples)
        log_q = get_log_q_fn(hrex.replicas)
        hrex, fraction_accepted_by_pair = hrex.attempt_neighbor_swaps(neighbor_pairs, log_q, n_swap_attempts_per_iter)

        samples_by_state_by_iter.append(samples_by_state)
        replica_idx_by_state_by_iter.append(hrex.replica_idx_by_state)
        fraction_accepted_by_pair_by_iter.append(fraction_accepted_by_pair)

    return samples_by_state_by_iter, HREXDiagnostics(replica_idx_by_state_by_iter, fraction_accepted_by_pair_by_iter)
