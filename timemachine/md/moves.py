from abc import ABC, abstractmethod
from functools import partial
from itertools import islice
from typing import Any, Generic, Iterator, List, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax.scipy.special import logsumexp as jlogsumexp
from numpy.typing import NDArray
from scipy.special import logsumexp

from timemachine import lib
from timemachine.lib import custom_ops
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import BoundPotential

_State = TypeVar("_State")


class Move(Generic[_State], ABC):
    @abstractmethod
    def move(self, _: _State) -> _State:
        ...

    def move_n(self, x: _State, n: int) -> _State:
        """Return the result of iterating the move n times.

        Subclasses may override this to use a more efficient implementation, e.g. to generate random numbers in batch.
        """
        for _ in range(n):
            x = self.move(x)
        return x

    def sample_chain_iter(self, x: _State) -> Iterator[_State]:
        """Given an initial state, returns an iterator over an infinite sequence of samples"""
        while True:
            x = self.move(x)
            yield x

    def sample_chain(self, x: _State, n_samples: int) -> List[_State]:
        """Given an initial state and number of samples, returns a finite sequence of samples

        Subclasses may override this to use a more efficient implementation, e.g. to generate random numbers in batch.
        """
        return list(islice(self.sample_chain_iter(x), n_samples))


class MonteCarloMove(Move[_State], ABC):
    def __init__(self):
        self._n_proposed = 0
        self._n_accepted = 0

    @abstractmethod
    def propose(self, x: _State) -> Tuple[_State, float]:
        """return proposed state and log acceptance probability"""

    def move(self, x: _State) -> _State:
        proposal, log_acceptance_probability = self.propose(x)
        self._n_proposed += 1

        alpha = np.random.rand()
        acceptance_probability = np.exp(log_acceptance_probability)
        if alpha < acceptance_probability:
            self._n_accepted += 1
            return proposal
        else:
            return x

    @property
    def n_proposed(self) -> int:
        return self._n_proposed

    @property
    def n_accepted(self) -> int:
        return self._n_accepted

    @property
    def acceptance_fraction(self) -> float:
        return self._n_accepted / self._n_proposed if self._n_proposed else np.nan


class CompoundMove(Move[_State]):
    def __init__(self, moves: Sequence[MonteCarloMove[_State]]):
        self.moves = moves

    @property
    def n_accepted_by_move(self) -> List[int]:
        return [m._n_accepted for m in self.moves]

    @property
    def n_proposed_by_move(self) -> List[int]:
        return [m._n_proposed for m in self.moves]


class MixtureOfMoves(CompoundMove[_State]):
    """Apply a single move uniformly selected from a list"""

    def __init__(self, moves: Sequence[MonteCarloMove[_State]]):
        self.moves = moves

    def move(self, x: _State) -> _State:
        idx = np.random.choice(len(self.moves))
        chosen_move = self.moves[idx]
        x = chosen_move.move(x)
        return x

    def move_n(self, x: _State, n: int):
        # Override default implementation to generate random selections in batch for efficiency
        idxs = np.random.choice(len(self.moves), size=n, replace=True)
        for idx in idxs:
            x = self.moves[idx].move(x)
        return x

    def sample_chain(self, x: _State, n_samples: int) -> List[_State]:
        # Override default implementation to generate random selections in batch for efficiency
        idxs = np.random.choice(len(self.moves), size=n_samples, replace=True)
        samples = []
        for idx in idxs:
            x = self.moves[idx].move(x)
            samples.append(x)
        return samples


class SequenceOfMoves(CompoundMove[_State]):
    """Apply each of a list of MonteCarloMoves in sequence"""

    def __init__(self, moves: Sequence[MonteCarloMove[_State]]):
        self.moves = moves

    def move(self, x: _State) -> _State:
        for individual_move in self.moves:
            x = individual_move.move(x)
        return x


class NVTMove(Move[CoordsVelBox]):
    def __init__(
        self,
        bps: List[BoundPotential],
        masses: NDArray,
        temperature: float,
        n_steps: int,
        seed: int,
        dt: float = 1.5e-3,
        friction: float = 1.0,
    ):
        intg = lib.LangevinIntegrator(temperature, dt, friction, masses, seed)
        self.integrator_impl = intg.impl()
        all_impls = [bp.to_gpu(np.float32).bound_impl for bp in bps]

        self.bound_impls = all_impls
        self.n_steps = n_steps

    def move(self, x: CoordsVelBox) -> CoordsVelBox:
        # note: context creation overhead here is actually very small!
        ctxt = custom_ops.Context(x.coords, x.velocities, x.box, self.integrator_impl, self.bound_impls)
        return self._steps(ctxt)

    def _steps(self, ctxt: "custom_ops.Context") -> CoordsVelBox:
        xs, boxes = ctxt.multiple_steps(self.n_steps, 0)
        x_t = xs[0]
        v_t = ctxt.get_v_t()
        box = boxes[0]

        after_steps = CoordsVelBox(x_t, v_t, box)

        return after_steps


class DeterministicMTMMove(Move):
    def __init__(self, rng_key):
        self.rng_key = rng_key
        self._n_proposed = 0
        self._n_accepted = 0

    @property
    def n_proposed(self):
        return self._n_proposed

    @property
    def n_accepted(self):
        return self._n_accepted

    @abstractmethod
    def acceptance_probability(self, x, box, key) -> Tuple[Any, Any, Any]:
        pass

    def move(self, xvb: CoordsVelBox) -> CoordsVelBox:
        self._n_proposed += 1
        y_proposed, acceptance_probability, key = self.acceptance_probability(xvb.coords, xvb.box, self.rng_key)
        # this may not be strictly necessary since the acceptance_probability should split the keys internally
        # but it never hurts to do an extra split.
        _, key = jrandom.split(key)
        alpha = jrandom.uniform(key)
        _, key = jrandom.split(key)
        self.rng_key = key
        if alpha < acceptance_probability:
            self._n_accepted += 1
            return CoordsVelBox(y_proposed, xvb.velocities, xvb.box)
        else:
            return xvb


class OptimizedMTMMove(DeterministicMTMMove):
    def __init__(self, K, batch_proposal_fn, batched_log_weights_fn, seed):
        """
        This is an optimized variant of the MTMMove, using a specialized choice lambda,
        proposal functions, and log_weights that take into account the proposal probability.

        Parameters
        ----------
        K: int
            Number of samples to generate for each move attempt.

        batch_proposal_fn: Callable(x: np.array, K: int, key: jrandom.key) -> List[np.array]
            Given the current coords, sample size, and a key, return a list of conformations

        batch_log_weights_fn: Callable(xs: List[np.array], box: np.array(3,3)) -> List[float]
            Return log weights given a list

        seed: int
            Seed to be used with jrandom.PRNGKey

        Returns
        -------
        float
            Acceptance rate probability.

        """
        self.K = K
        self.batch_proposal_fn = batch_proposal_fn
        self.batched_log_weights_fn = batched_log_weights_fn
        super().__init__(jrandom.PRNGKey(seed))

    @partial(jax.jit, static_argnums=(0,))
    def acceptance_probability(self, x, box, key):
        # split #1
        yj = self.batch_proposal_fn(x, self.K, key)
        _, key = jrandom.split(key)

        log_weights_yj = self.batched_log_weights_fn(yj, box)
        normalized_weights_yj = jnp.exp(log_weights_yj - jlogsumexp(log_weights_yj))

        # split #2
        y_jdx = jrandom.choice(key, jnp.arange(self.K), p=normalized_weights_yj)
        _, key = jrandom.split(key)

        y_proposed = yj[y_jdx]

        # split # 3
        xi_k_sub_1 = self.batch_proposal_fn(y_proposed, self.K - 1, key)
        _, key = jrandom.split(key)

        xi = jnp.concatenate([xi_k_sub_1, jnp.array([x])])
        log_weights_xi = self.batched_log_weights_fn(xi, box)
        log_ratio = jlogsumexp(log_weights_yj) - jlogsumexp(log_weights_xi)

        return y_proposed, jnp.exp(log_ratio), key


class ReferenceMTMMove(DeterministicMTMMove):
    def __init__(self, K, batch_proposal_fn, batch_log_Q_fn, batch_log_pi_fn, batch_log_lambda_a_b_fn, seed):
        """
        The recipe here roughly follows:
        https://www.stat.purdue.edu/~fmliang/papers/2000/Mtry.pdf

        Parameters
        ----------
        K: int
            Number of samples to generate for each move attempt.

        batch_proposal_fn: function that takes in CoordsVelBox -> List[CoordsVelBox]
            Batched proposal function g that proposes multiple new coordinates given a single x and associated
            log likelihoods

        batch_log_Q_fn: (List[a], b) -> List[Float]
            Batched proposal probability function.

        batch_log_pi_fn: function that takes in List[CoordsVelBox] -> List[float]
            Batched log likelihood function of a given state

        batch_log_lambda_a_b_fn: (List[a], b) -> List[Float]
            Batched log lambda function, must be symmetric for each element (a_i,b)

        """
        self.K = K
        self.batch_proposal_fn = batch_proposal_fn
        self.batch_log_Q_fn = batch_log_Q_fn
        self.batch_log_pi_fn = batch_log_pi_fn
        self.batch_log_lambda_fn = batch_log_lambda_a_b_fn
        super().__init__(jrandom.PRNGKey(seed))

    def acceptance_probability(self, xvb, key):
        # split # 1
        yj = self.batch_proposal_fn(xvb, self.K, key)
        _, key = jrandom.split(key)

        log_Q_y_x = self.batch_log_Q_fn(yj, xvb)
        log_pi_yj = self.batch_log_pi_fn(yj)

        log_weights_yj = log_pi_yj + log_Q_y_x + self.batch_log_lambda_fn(yj, xvb)
        assert self.K == len(yj)
        assert self.K == len(log_pi_yj)

        # (ytz): we can use these normalized weights here directly to compute observables
        # as a "cheap" importance sampling protocol
        normalized_weights_yj = np.exp(log_weights_yj - logsumexp(log_weights_yj))

        assert np.abs(np.sum(normalized_weights_yj) - 1) < 1e-9

        # split #2
        y_jdx = jrandom.choice(key, np.arange(self.K), p=normalized_weights_yj)
        _, key = jrandom.split(key)

        y_proposed = yj[y_jdx]

        # split #3
        xi_k_sub_1 = self.batch_proposal_fn(y_proposed, self.K - 1, key)
        _, key = jrandom.split(key)

        xi = xi_k_sub_1 + [xvb]
        log_Q_x_y = self.batch_log_Q_fn(xi, y_proposed)
        assert len(xi) == self.K

        log_pi_xi = self.batch_log_pi_fn(xi)
        log_weights_xi = log_pi_xi + log_Q_x_y + self.batch_log_lambda_fn(xi, y_proposed)
        log_ratio = logsumexp(log_weights_yj) - logsumexp(log_weights_xi)

        return y_proposed, np.exp(log_ratio), key
