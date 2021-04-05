# Generate test cases that make the force magnitude large

import numpy as np

from jax import jit, grad, numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

from scipy.optimize import minimize

from typing import Tuple
from functools import partial

from timemachine.potentials import bonded


def force_magnitude_loss_factory(restraint_potential: callable) -> callable:
    """Given a restraint_potential(x0, x1) -> float,
    define loss_fxn(x0, x1) = - magnitude_of_restraint_force(x0, x1)"""

    def loss_fxn(x0, x1):
        g = grad(restraint_potential, argnums=(0, 1))(x0, x1)
        return - (jnp.linalg.norm(g[0]) + jnp.linalg.norm(g[1]))

    return loss_fxn


# TODO: optimize for other things, like singular value degeneracy, or other quantities that might correspond to these cases:
#   https://github.com/proteneer/timemachine/pull/382#issuecomment-813426075


def optimize_point_sets(restraint_potential, loss_factory, N=10, norm_penalty=1.0, seed=0, verbose=True) -> Tuple[
    np.array, np.array]:
    """Use scipy optimize to find point sets x0, x1 that minimize

        loss_fxn(x0, x1) + norm_penalty * norm(flatten([x0, x1]))

        where
            loss_xn = loss_factory(restraint_potential)
            len(x0) = len(x1) = N
    """

    loss_fxn = loss_factory(restraint_potential)

    np.random.seed(seed)
    x0 = jnp.array(np.random.randn(N, 3))
    x1 = jnp.array(np.random.randn(N, 3))

    print(f'finding an instance with big forces where N particles = {N}...')

    def unflatten(x: np.array) -> Tuple[np.array, np.array]:
        x0_1 = jnp.reshape(x, (2 * N, 3))
        x0, x1 = x0_1[:N], x0_1[N:]
        return x0, x1

    @jit
    def f(x: np.array) -> float:
        return loss_fxn(*unflatten(x)) + norm_penalty * jnp.linalg.norm(x)

    def value_and_grad(x: np.array) -> Tuple[float, np.array]:
        return float(f(x)), np.array(grad(f)(x))

    x = np.vstack([x0, x1]).flatten()

    result = minimize(value_and_grad, x, jac=True, method='L-BFGS-B')

    x0, x1 = unflatten(result.x)

    if verbose:
        print(f'U = {restraint_potential(x0, x1)}')

        print(f'norm(x0, axis=1) = {np.linalg.norm(x0, axis=1)}')
        print(f'norm(x1, axis=1) = {np.linalg.norm(x1, axis=1)}')

        g = grad(restraint_potential, argnums=(0, 1))(x0, x1)

        print(f'norm(dU/x0, axis=1) = {np.linalg.norm(g[0], axis=1)}')
        print(f'norm(dU/x1, axis=1) = {np.linalg.norm(g[1], axis=1)}')

    return unflatten(result.x)


if __name__ == '__main__':

    # number of points in x0, x1
    N = 5

    U = partial(bonded.rmsd_restraint,
                group_a_idxs=jnp.arange(N),
                group_b_idxs=jnp.arange(N) + N,
                k=10.0,
                lamb=0.0,  # required
                box=np.eye(3),  # required
                params=jnp.array([], dtype=np.float64),  # required
                )


    def restraint_potential(x0, x1):
        return U(jnp.vstack([x0, x1]))


    x0, x1 = optimize_point_sets(restraint_potential, force_magnitude_loss_factory, N=N)

    print(f'x0 = {x0}')
    print(f'x1 = {x1}')
