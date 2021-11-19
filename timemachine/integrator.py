from timemachine.constants import BOLTZ
import numpy as np
import jax
from jax import random as jrandom
import time


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


class Integrator:
    def step(self, x, v):
        """Return copies x and v, updated by a single timestep"""
        raise NotImplementedError

    def multiple_steps(self, x, v, n_steps=1000):
        """Return trajectories of x and v, advanced by n_steps"""
        xs, vs = [x], [v]

        for _ in range(n_steps):
            new_x, new_v = self.step(xs[-1], vs[-1])

            xs.append(new_x)
            vs.append(new_v)

        return np.array(xs), np.array(vs)


class LangevinIntegrator(Integrator):
    def __init__(self, force_fxn, masses, temperature, dt, friction):
        """BAOAB (https://arxiv.org/abs/1203.5428), rotated by half a timestep"""
        self.dt = dt
        self.masses = masses
        ca, cb, cc = langevin_coefficients(temperature, dt, friction, masses)
        self.force_fxn = force_fxn

        # make masses, frictions, etc. (scalar or (N,)) shape-compatible with coordinates (vector or (N,3))
        # note: per-atom frictions allowed
        self.ca, self.cb, self.cc = np.expand_dims(ca, -1), np.expand_dims(cb, -1), np.expand_dims(cc, -1)

    def step(self, x, v):
        """Intended to match https://github.com/proteneer/timemachine/blob/37e60205b3ae3358d9bb0967d03278ed184b8976/timemachine/cpp/src/integrator.cu#L71-L74"""
        v_mid = v + self.cb * self.force_fxn(x)

        new_v = (self.ca * v_mid) + (self.cc * np.random.randn(*x.shape))
        new_x = x + 0.5 * self.dt * (v_mid + new_v)

        return new_x, new_v


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

    for batch_step in range(num_batches):
        #                                             [B,N,3][B,N,3][B,2]
        xs_t, vs_t, keys_t = batched_multiple_steps_fn(xs_t, vs_t, keys_t)
        all_xs.append(xs_t)

    # result has shape [num_workers, num_batches, num_atoms, num_dimensions]
    return np.transpose(np.array(all_xs), axes=[1, 0, 2, 3])
