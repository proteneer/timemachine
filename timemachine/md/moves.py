from abc import ABC, abstractmethod
from functools import partial
from itertools import islice
from typing import Callable, Generic, Iterator, List, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKeyArray
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
    def move(self, key: PRNGKeyArray, _: _State) -> Tuple[PRNGKeyArray, _State]:
        ...

    def sample_chain_iter(self, key: PRNGKeyArray, x: _State) -> Iterator[_State]:
        """Given an initial state, returns an iterator over an infinite sequence of samples"""
        while True:
            key, x = self.move(key, x)
            yield x

    def sample_chain(self, key: PRNGKeyArray, x: _State, n_samples: int) -> List[_State]:
        """Given an initial state and number of samples, returns a finite sequence of samples"""
        return list(islice(self.sample_chain_iter(key, x), n_samples))


class MonteCarloMove(Move[_State], ABC):
    def __init__(self):
        self._n_proposed = 0
        self._n_accepted = 0

    @abstractmethod
    def propose(self, key: PRNGKeyArray, x: _State) -> Tuple[PRNGKeyArray, _State, float]:
        """return proposed state and log acceptance probability"""

    def move(self, key: PRNGKeyArray, x: _State) -> Tuple[PRNGKeyArray, _State]:
        key, x_prop, log_acceptance_probability = self.propose(key, x)
        self._n_proposed += 1

        acceptance_probability = jnp.exp(log_acceptance_probability)
        accepted = jax.random.bernoulli(key, acceptance_probability)

        if accepted:
            self._n_accepted += 1
            x_next = x_prop
        else:
            x_next = x

        return key, x_next

    @property
    def n_proposed(self) -> int:
        return self._n_proposed

    @property
    def n_accepted(self) -> int:
        return self._n_accepted

    @property
    def acceptance_fraction(self) -> float:
        return self._n_accepted / self._n_proposed if self._n_proposed else np.nan


class MetropolisHastingsMove(MonteCarloMove[_State], ABC):
    @abstractmethod
    def propose_with_log_q_diff(self, key: PRNGKeyArray, x: _State) -> Tuple[PRNGKeyArray, _State, float]:
        """Return proposed state and the difference in log unnormalized probability, i.e.

        log_q_diff = log(q(x_proposed)) - log(q(x))
        """

    def propose(self, key: PRNGKeyArray, x: _State) -> Tuple[PRNGKeyArray, _State, float]:
        key, proposal, log_q_diff = self.propose_with_log_q_diff(key, x)
        log_acceptance_probability = jnp.minimum(log_q_diff, 0.0).item()
        return key, proposal, log_acceptance_probability


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

    def move(self, key: PRNGKeyArray, x: _State) -> Tuple[PRNGKeyArray, _State]:
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, len(self.moves)).item()
        chosen_move = self.moves[idx]
        key, x = chosen_move.move(key, x)
        return key, x


class SequenceOfMoves(CompoundMove[_State]):
    """Apply each of a list of MonteCarloMoves in sequence"""

    def __init__(self, moves: Sequence[MonteCarloMove[_State]]):
        self.moves = moves

    def move(self, key: PRNGKeyArray, x: _State) -> Tuple[PRNGKeyArray, _State]:
        for individual_move in self.moves:
            key, x = individual_move.move(key, x)
        return key, x


class NVTMove(Move[CoordsVelBox]):
    def __init__(
        self,
        bps: List[BoundPotential],
        masses: NDArray,
        temperature: float,
        n_steps: int,
        dt: float = 1.5e-3,
        friction: float = 1.0,
    ):
        intg = lib.LangevinIntegrator(temperature, dt, friction, masses, 0)
        self.integrator_impl = intg.impl()
        all_impls = [bp.to_gpu(np.float32).bound_impl for bp in bps]

        self.bound_impls = all_impls
        self.n_steps = n_steps

    def move(self, key: PRNGKeyArray, xvb: CoordsVelBox) -> Tuple[PRNGKeyArray, CoordsVelBox]:
        key, subkey = jax.random.split(key)
        seed = jax.random.randint(subkey, (), np.iinfo(np.int32).min, np.iinfo(np.int32).max).item()
        self.integrator_impl.set_seed(seed)

        # note: context creation overhead here is actually very small!
        ctxt = custom_ops.Context(xvb.coords, xvb.velocities, xvb.box, self.integrator_impl, self.bound_impls)
        return key, self._steps(ctxt)

    def _steps(self, ctxt: "custom_ops.Context") -> CoordsVelBox:
        xs, boxes = ctxt.multiple_steps(self.n_steps, 0)
        x_t = xs[0]
        v_t = ctxt.get_v_t()
        box = boxes[0]

        after_steps = CoordsVelBox(x_t, v_t, box)

        return after_steps


class OptimizedMTMMove(MetropolisHastingsMove[CoordsVelBox]):
    def __init__(
        self,
        K,
        batch_proposal_fn: Callable[[jax.Array, int, PRNGKeyArray], jax.Array],
        batched_log_weights_fn: Callable[[jax.Array, jax.Array], jax.Array],
    ):
        """
        This is an optimized variant of the MTMMove, using a specialized choice lambda,
        proposal functions, and log_weights that take into account the proposal probability.

        Parameters
        ----------
        K: int
            Number of samples to generate for each move attempt.

        batch_proposal_fn: Callable[[jax.Array, int, PRNGKeyArray], jax.Array]
            Given the current coords, sample size, and a key, return a list of conformations

        batch_log_weights_fn: Callable[[jax.Array, jax.Array], jax.Array]
            Return log weights given a list

        seed: int
            Seed to be used with PRNGKey

        Returns
        -------
        float
            Acceptance rate probability.

        """
        super().__init__()
        self.K = K
        self.batch_proposal_fn = batch_proposal_fn
        self.batched_log_weights_fn = batched_log_weights_fn

    @partial(jax.jit, static_argnums=(0,))
    def _propose_with_log_q_diff_impl(
        self, key: PRNGKeyArray, xvb: CoordsVelBox
    ) -> Tuple[PRNGKeyArray, CoordsVelBox, jax.Array]:
        key, subkey = jax.random.split(key)
        yj = self.batch_proposal_fn(xvb.coords, self.K, subkey)

        log_weights_yj = self.batched_log_weights_fn(yj, xvb.box)
        normalized_weights_yj = jnp.exp(log_weights_yj - jlogsumexp(log_weights_yj))

        key, subkey = jax.random.split(key)
        y_jdx = jax.random.choice(subkey, self.K, p=normalized_weights_yj)

        y_proposed = yj[y_jdx]

        key, subkey = jax.random.split(key)
        xi_k_sub_1 = self.batch_proposal_fn(y_proposed, self.K - 1, subkey)

        xi = jnp.concatenate([xi_k_sub_1, jnp.array([xvb.coords])])
        log_weights_xi = self.batched_log_weights_fn(xi, xvb.box)
        log_ratio = jlogsumexp(log_weights_yj) - jlogsumexp(log_weights_xi)

        return key, CoordsVelBox(y_proposed, xvb.velocities, xvb.box), log_ratio

    def propose_with_log_q_diff(self, key: PRNGKeyArray, xvb: CoordsVelBox) -> Tuple[PRNGKeyArray, CoordsVelBox, float]:
        key, xvb, log_q_diff = self._propose_with_log_q_diff_impl(key, xvb)
        return key, xvb, log_q_diff.item()


class ReferenceMTMMove(MetropolisHastingsMove[CoordsVelBox]):
    def __init__(
        self,
        K: int,
        batch_proposal_fn: Callable[[CoordsVelBox, int, PRNGKeyArray], List[CoordsVelBox]],
        batch_log_Q_fn: Callable[[List[CoordsVelBox], CoordsVelBox], jax.Array],
        batch_log_pi_fn: Callable[[List[CoordsVelBox]], jax.Array],
        batch_log_lambda_a_b_fn: Callable[[List[CoordsVelBox], CoordsVelBox], jax.Array],
    ):
        """
        The recipe here roughly follows:
        https://www.stat.purdue.edu/~fmliang/papers/2000/Mtry.pdf

        Parameters
        ----------
        K: int
            Number of samples to generate for each move attempt.

        batch_proposal_fn: Callable[[CoordsVelBox, int, PRNGKeyArray], List[CoordsVelBox]]
            Batched proposal function g that proposes multiple new CoordsVelBox given a single CoordsVelBox

        batch_log_Q_fn: Callable[[List[CoordsVelBox], CoordsVelBox], jax.Array]
            Batched proposal probability function.

        batch_log_pi_fn: Callable[[List[CoordsVelBox]], jax.Array]
            Batched log likelihood function of a given state

        batch_log_lambda_a_b_fn: Callable[[List[CoordsVelBox], CoordsVelBox], jax.Array]
            Batched log lambda function, must be symmetric for each element (a_i, b)

        """
        super().__init__()
        self.K = K
        self.batch_proposal_fn = batch_proposal_fn
        self.batch_log_Q_fn = batch_log_Q_fn
        self.batch_log_pi_fn = batch_log_pi_fn
        self.batch_log_lambda_fn = batch_log_lambda_a_b_fn

    @partial(jax.jit, static_argnums=(0,))
    def _propose_with_log_q_diff_impl(
        self, key: PRNGKeyArray, xvb: CoordsVelBox
    ) -> Tuple[PRNGKeyArray, CoordsVelBox, jax.Array]:
        key, subkey = jax.random.split(key)
        yj = self.batch_proposal_fn(xvb, self.K, subkey)

        log_Q_y_x = self.batch_log_Q_fn(yj, xvb)
        log_pi_yj = self.batch_log_pi_fn(yj)

        log_weights_yj = log_pi_yj + log_Q_y_x + self.batch_log_lambda_fn(yj, xvb)
        assert self.K == len(yj)
        assert self.K == len(log_pi_yj)

        # (ytz): we can use these normalized weights here directly to compute observables
        # as a "cheap" importance sampling protocol
        normalized_weights_yj = np.exp(log_weights_yj - logsumexp(log_weights_yj))

        assert np.abs(np.sum(normalized_weights_yj) - 1) < 1e-9

        key, subkey = jax.random.split(key)
        y_jdx = jax.random.choice(subkey, np.arange(self.K), p=normalized_weights_yj)

        y_proposed = yj[y_jdx]

        key, subkey = jax.random.split(key)
        xi_k_sub_1 = self.batch_proposal_fn(y_proposed, self.K - 1, subkey)

        xi = xi_k_sub_1 + [xvb]
        log_Q_x_y = self.batch_log_Q_fn(xi, y_proposed)
        assert len(xi) == self.K

        log_pi_xi = self.batch_log_pi_fn(xi)
        log_weights_xi = log_pi_xi + log_Q_x_y + self.batch_log_lambda_fn(xi, y_proposed)
        log_ratio = logsumexp(log_weights_yj) - logsumexp(log_weights_xi)

        return key, y_proposed, log_ratio

    def propose_with_log_q_diff(self, key: PRNGKeyArray, xvb: CoordsVelBox) -> Tuple[PRNGKeyArray, CoordsVelBox, float]:
        key, xvb_proposed, log_q_diff = self._propose_with_log_q_diff_impl(key, xvb)
        return key, xvb_proposed, log_q_diff.item()


def random_seed(key: PRNGKeyArray) -> Tuple[PRNGKeyArray, int]:
    key, subkey = jax.random.split(key)
    return key, jax.random.randint(subkey, (), np.iinfo(np.int32).min, np.iinfo(np.int32).max).item()
