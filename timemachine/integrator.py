import time
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jrandom

from timemachine.constants import BOLTZ
from timemachine.lib.fixed_point import fixed_to_float, float_to_fixed


def langevin_coefficients(temperature, dt, friction, masses):
    """
    Compute coefficients for langevin dynamics

    Parameters
    ----------
    temperature: float
        units of Kelvin

    dt: float
        units of picoseconds

    friction: float
        collision rate in 1 / picoseconds

    masses: array
        mass of each atom in standard mass units. np.inf masses will
        effectively freeze the particles.

    Returns
    -------
    tuple (ca, cb, cc)
        ca is scalar, and cb and cc are n length arrays
        that are used during langevin dynamics as follows:

        during heat-bath update
        v -> ca * v + cc * gaussian

        during force update
        v -> v + cb * force
    """
    kT = BOLTZ * temperature
    nscale = np.sqrt(kT / masses)

    ca = np.exp(-friction * dt)
    cb = dt / masses
    cc = np.sqrt(1 - np.exp(-2 * friction * dt)) * nscale

    return ca, cb, cc


class Integrator(ABC):
    @abstractmethod
    def step(self, x, v) -> tuple[Any, Any]:
        """Return copies x and v, updated by a single timestep"""
        pass

    def multiple_steps(self, x, v, n_steps: int = 1000):
        """Return trajectories of x and v, advanced by n_steps"""
        xs, vs = [x], [v]

        for _ in range(n_steps):
            new_x, new_v = self.step(xs[-1], vs[-1])

            xs.append(new_x)
            vs.append(new_v)

        return np.array(xs), np.array(vs)


class StochasticIntegrator(ABC):
    @abstractmethod
    def step(self, x, v, rng: np.random.Generator) -> tuple[Any, Any]:
        """Return copies x and v, updated by a single timestep. Accepts a numpy Generator instance for determinism."""
        pass

    @abstractmethod
    def step_lax(self, key, x, v) -> tuple[Any, Any]:
        """Return copies x and v, updated by a single timestep. Accepts a jax PRNG key for determinism."""
        pass

    def multiple_steps(self, x, v, n_steps: int = 1000, rng: Optional[np.random.Generator] = None):
        """Return trajectories of x and v, advanced by n_steps"""

        rng = rng or np.random.default_rng()

        xs, vs = [x], [v]

        for _ in range(n_steps):
            new_x, new_v = self.step(xs[-1], vs[-1], rng)

            xs.append(new_x)
            vs.append(new_v)

        return np.array(xs), np.array(vs)

    @partial(jax.jit, static_argnums=(0, 4))
    def multiple_steps_lax(self, key, x, v, n_steps: int = 1000):
        """
        Return trajectories of x and v, advanced by n_steps. Implemented using jax.lax.scan to allow jax.jit to produce
        efficient code.

        Note: requires that force_fxn be jax-transformable
        """

        def f(xv, key):
            x, v = xv
            xv_ = self.step_lax(key, x, v)
            return xv_, xv_

        keys = jax.random.split(key, n_steps)
        _, (xs, vs) = jax.lax.scan(f, (x, v), keys)

        return (
            jnp.concatenate((x[jnp.newaxis, :], xs)),
            jnp.concatenate((v[jnp.newaxis, :], vs)),
        )


class LangevinIntegrator(StochasticIntegrator):
    def __init__(self, force_fxn, masses, temperature, dt, friction):
        """BAOAB (https://arxiv.org/abs/1203.5428), rotated by half a timestep"""
        self.dt = dt
        self.masses = masses
        self.temperature = temperature
        ca, cb, cc = langevin_coefficients(temperature, dt, friction, masses)
        self.force_fxn = force_fxn

        # make masses, frictions, etc. (scalar or (N,)) shape-compatible with coordinates (vector or (N,3))
        # note: per-atom frictions allowed
        self.ca, self.cb, self.cc = np.expand_dims(ca, -1), np.expand_dims(cb, -1), np.expand_dims(cc, -1)

    def _step(self, x, v, noise):
        """Intended to match https://github.com/proteneer/timemachine/blob/37e60205b3ae3358d9bb0967d03278ed184b8976/timemachine/cpp/src/integrator.cu#L71-L74"""
        v_mid = v + self.cb * self.force_fxn(x)

        new_v = (self.ca * v_mid) + (self.cc * noise)
        new_x = x + 0.5 * self.dt * (v_mid + new_v)

        return new_x, new_v

    def step(self, x, v, rng):
        return self._step(x, v, rng.normal(size=x.shape))

    def step_lax(self, key, x, v):
        return self._step(x, v, jax.random.normal(key, x.shape))


class VelocityVerletIntegrator(Integrator):
    def __init__(self, force_fxn, masses, dt):
        """WARNING: `.step` makes 2x more calls to force_fxn per timestep than `.multiple_steps`"""
        self.dt = dt
        self.masses = masses[:, np.newaxis]  # TODO: cleaner way to handle (n_atoms,) vs. (n_atoms, 3) mismatch?
        self.force_fxn = force_fxn
        self.cb = self.dt / self.masses

    def step(self, x, v):
        """WARNING: makes 2 calls to force_fxn per timestep -- prefer `.multiple_steps` in most cases"""
        v_mid = float_to_fixed(v) + float_to_fixed((0.5 * self.cb) * self.force_fxn(x))
        fixed_x = float_to_fixed(x) + float_to_fixed(self.dt * fixed_to_float(v_mid))
        fixed_v = v_mid + float_to_fixed((0.5 * self.cb) * self.force_fxn(fixed_to_float(fixed_x)))

        return fixed_to_float(fixed_x), fixed_to_float(fixed_v)

    def multiple_steps(self, x, v, n_steps=1000):
        # note: intermediate timesteps are staggered
        #    xs[0], vs[0] = x_0, v_0
        #    xs[1], vs[1] = x_2, v_{1.5}
        #    xs[2], vs[2] = x_3, v_{2.5}
        #    ...
        #    xs[T-1], vs[T-1] = x_T, v_{T-0.5}
        #    xs[T],   vs[T]   = x_T, v_T

        # note: reorders loop slightly to avoid ~n_steps extraneous calls to force_fxn
        x_fixed = float_to_fixed(x)
        v_fixed = float_to_fixed(v)

        zs = [(x_fixed, v_fixed)]
        # initialize traj
        v_fixed = v_fixed + float_to_fixed((0.5 * self.cb) * self.force_fxn(fixed_to_float(x_fixed)))
        x_fixed = x_fixed + float_to_fixed(self.dt * fixed_to_float(v_fixed))

        # run n_steps-1 steps
        for t in range(n_steps - 1):
            v_fixed = v_fixed + float_to_fixed(self.cb * self.force_fxn(fixed_to_float(x_fixed)))
            x_fixed = x_fixed + float_to_fixed(self.dt * fixed_to_float(v_fixed))

            zs.append((x_fixed, v_fixed))

        # finalize traj
        v_fixed = v_fixed + float_to_fixed((0.5 * self.cb) * self.force_fxn(fixed_to_float(x_fixed)))

        zs.append((x_fixed, v_fixed))

        xs = np.array([x for (x, _) in zs])
        vs = np.array([v for (_, v) in zs])
        return fixed_to_float(xs), fixed_to_float(vs)

    def _update_via_fori_loop(self, x, v, n_steps=1000):
        # initialize

        v_fixed = float_to_fixed(v) + float_to_fixed((0.5 * self.cb) * self.force_fxn(x))
        x_fixed = float_to_fixed(x) + float_to_fixed(self.dt * fixed_to_float(v_fixed))

        def velocity_verlet_loop_body(_, val):
            x_prev, v_prev = val

            v_fixed = v_prev + float_to_fixed(self.cb * self.force_fxn(fixed_to_float(x_prev)))
            x_fixed = x_prev + float_to_fixed(self.dt * fixed_to_float(v_fixed))
            return x_fixed, v_fixed

        # run n_steps - 1 steps
        x_fixed, v_fixed = jax.lax.fori_loop(0, n_steps - 1, velocity_verlet_loop_body, (x_fixed, v_fixed))

        # finalize
        v_fixed = v_fixed + float_to_fixed((0.5 * self.cb) * self.force_fxn(fixed_to_float(x_fixed)))

        return fixed_to_float(x_fixed), fixed_to_float(v_fixed)


def _fori_steps(x0, v0, key0, grad_fn, num_steps, dt, ca, cbs, ccs):
    def body_fn(_, val):
        # BAOAB integrator
        x_t, v_t, key = val
        du_dx = grad_fn(x_t)[0]
        v_mid = v_t + cbs * du_dx
        noise = jrandom.normal(key, v_t.shape)
        _, sub_key = jrandom.split(key)
        v_t = ca * v_mid + ccs * noise
        x_t += 0.5 * dt * (v_mid + v_t)
        return x_t, v_t, sub_key

    return jax.lax.fori_loop(0, num_steps, body_fn, (x0, v0, key0))


def simulate(x0, U_fn, temperature, masses, steps_per_batch, num_batches, num_workers, seed=None):
    """
    Simulate a gas-phase system using a reference jax implementation.

    Parameters
    ----------

    x0: (N,3) np.ndarray
        initial coordinates

    U_fn: function
        Potential energy function

    temperature: float
        Temperature in Kelvin

    steps_per_batch: int
        number of steps we run for each batch

    num_batches: int
        number of batches we run

    num_workers: int
        How many jobs to run in parallel

    seed: int
        used for the random number generated

    Returns
    -------

    """
    dt = 1.5e-3
    friction = 1.0
    ca, cbs, ccs = langevin_coefficients(temperature, dt, friction, masses)
    cbs = np.expand_dims(cbs * -1, axis=-1)
    ccs = np.expand_dims(ccs, axis=-1)

    grad_fn = jax.jit(jax.grad(U_fn, argnums=(0,)))
    U_fn = jax.jit(U_fn)

    if seed is None:
        seed = int(time.time())

    @jax.jit
    def multiple_steps(x0, v0, key0):
        return _fori_steps(x0, v0, key0, grad_fn, steps_per_batch, dt, ca, cbs, ccs)

    v0 = np.zeros_like(x0)

    # jitting a pmap will result in a warning about inefficient data movement
    batched_multiple_steps_fn = jax.pmap(multiple_steps)

    xs_t = np.array([x0] * num_workers)
    vs_t = np.array([v0] * num_workers)
    keys_t = np.array([jrandom.PRNGKey(seed + idx) for idx in range(num_workers)])

    all_xs = []
    all_vs = []

    for batch_step in range(num_batches):
        #                                             [B,N,3][B,N,3][B,2]
        xs_t, vs_t, keys_t = batched_multiple_steps_fn(xs_t, vs_t, keys_t)
        all_xs.append(xs_t)
        all_vs.append(vs_t)

    # result has shape [num_workers, num_batches, num_atoms, num_dimensions]
    return np.transpose(np.array(all_xs), axes=[1, 0, 2, 3]), np.transpose(np.array(all_vs), axes=[1, 0, 2, 3])
