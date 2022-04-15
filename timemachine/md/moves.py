from abc import abstractmethod
from functools import partial
from typing import Any, List, Tuple, cast

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax.scipy.special import logsumexp as jlogsumexp
from numpy.typing import NDArray
from scipy.special import logsumexp

from timemachine import lib
from timemachine.lib import custom_ops, potentials
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.states import CoordsVelBox


class MonteCarloMove:
    n_proposed: int = 0
    n_accepted: int = 0

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        """return proposed state and log acceptance probability"""
        raise NotImplementedError

    def move(self, x: CoordsVelBox) -> CoordsVelBox:
        proposal, log_acceptance_probability = self.propose(x)
        self.n_proposed += 1

        alpha = np.random.rand()
        acceptance_probability = np.exp(log_acceptance_probability)
        if alpha < acceptance_probability:
            self.n_accepted += 1
            return proposal
        else:
            return x

    @property
    def acceptance_fraction(self):
        if self.n_proposed > 0:
            return self.n_accepted / self.n_proposed
        else:
            return 0.0


class CompoundMove(MonteCarloMove):
    def __init__(self, moves: List[MonteCarloMove]):
        """Apply each of a list of moves in sequence"""
        self.moves = moves

    def move(self, x: CoordsVelBox) -> CoordsVelBox:
        for individual_move in self.moves:
            x = individual_move.move(x)
        return x

    @property
    def n_accepted_by_move(self):
        return np.array([m.n_accepted for m in self.moves])

    @property
    def n_proposed_by_move(self):
        return np.array([m.n_proposed for m in self.moves])

    @property
    def n_accepted(self):
        return np.sum(self.n_accepted_by_move)

    @property
    def n_proposed(self):
        return np.sum(self.n_proposed_by_move)


class NPTMove(MonteCarloMove):
    def __init__(
        self,
        ubps: List[potentials.CustomOpWrapper],
        lamb: float,
        masses: NDArray,
        temperature: float,
        pressure: float,
        n_steps: int,
        seed: int,
        dt: float = 1.5e-3,
        friction: float = 1.0,
        barostat_interval: int = 5,
    ):

        intg = lib.LangevinIntegrator(temperature, dt, friction, masses, seed)
        self.integrator_impl = intg.impl()
        all_impls = [bp.bound_impl(np.float32) for bp in ubps]

        bond_list = get_bond_list(cast(potentials.HarmonicBond, ubps[0]))
        group_idxs = get_group_indices(bond_list)

        barostat = lib.MonteCarloBarostat(len(masses), pressure, temperature, group_idxs, barostat_interval, seed + 1)
        barostat_impl = barostat.impl(all_impls)

        self.bound_impls = all_impls
        self.barostat_impl = barostat_impl
        self.lamb = lamb
        self.n_steps = n_steps

    def move(self, x: CoordsVelBox) -> CoordsVelBox:
        # note: context creation overhead here is actually very small!
        ctxt = custom_ops.Context(
            x.coords, x.velocities, x.box, self.integrator_impl, self.bound_impls, self.barostat_impl
        )

        # arguments: lambda_schedule, du_dl_interval, x_interval
        _, xs, boxes = ctxt.multiple_steps(self.lamb * np.ones(self.n_steps), 0, 0)
        x_t = xs[0]
        v_t = ctxt.get_v_t()
        box = boxes[0]

        after_npt = CoordsVelBox(x_t, v_t, box)

        self.n_proposed += 1
        self.n_accepted += 1

        return after_npt


class DeterministicMTMMove(MonteCarloMove):

    rng_key: Any

    @abstractmethod
    def acceptance_probability(self, x, box, key) -> Tuple[Any, Any, Any]:
        pass

    def move(self, xvb: CoordsVelBox) -> CoordsVelBox:
        self.n_proposed += 1
        y_proposed, acceptance_probability, key = self.acceptance_probability(xvb.coords, xvb.box, self.rng_key)
        # this may not be strictly necessary since the acceptance_probability should split the keys internally
        # but it never hurts to do an extra split.
        _, key = jrandom.split(key)
        alpha = jrandom.uniform(key)
        _, key = jrandom.split(key)
        self.rng_key = key
        if alpha < acceptance_probability:
            self.n_accepted += 1
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
        self.n_accepted = 0
        self.n_proposed = 0
        self.batch_proposal_fn = batch_proposal_fn
        self.batched_log_weights_fn = batched_log_weights_fn
        self.rng_key = jrandom.PRNGKey(seed)

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
        self.n_accepted = 0
        self.n_proposed = 0
        self.batch_proposal_fn = batch_proposal_fn
        self.batch_log_Q_fn = batch_log_Q_fn
        self.batch_log_pi_fn = batch_log_pi_fn
        self.batch_log_lambda_fn = batch_log_lambda_a_b_fn
        self.rng_key = jrandom.PRNGKey(seed)

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
