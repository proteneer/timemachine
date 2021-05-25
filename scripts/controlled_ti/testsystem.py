from collections import namedtuple
from jax.config import config

config.update("jax_enable_x64", True)

import numpy as onp
from jax import numpy as np, jit
from jax.numpy.linalg import norm
from tqdm import tqdm

NonbondedParams = namedtuple(
    'NonbondedParams',
    ['cutoff', 'radius', 'sigma', 'epsilon']
)
default_params = NonbondedParams(
    cutoff=5.0, radius=5.0, sigma=1.0, epsilon=1.0,
)


def pdist(x):
    """array of pairwise distances"""
    i, j = np.triu_indices(len(x), k=1)
    return norm(x[i] - x[j], axis=1)


def apply_cutoff(distance, cutoff):
    """distance >= cutoff -> +inf"""
    return np.where(distance < cutoff, distance, np.inf)


def flat_bottom_restraint(distances, radius):
    """restraint 0 up until distances > radius, then quadratic"""
    return np.maximum(0, distances - radius) ** 2


def lennard_jones(distances, sigma, epsilon):
    return 4 * epsilon * ((sigma / distances) ** 12 - (sigma / distances) ** 6)


def U(x: np.array, lam: float = 0.0, params: NonbondedParams = default_params) -> float:
    alchemical, normal = x[0], x[1:]

    # distances
    pairwise_distances = apply_cutoff(pdist(normal), params.cutoff)

    w = (1 - lam) * params.cutoff
    _distances_to_alchemical_particle = norm(normal - alchemical, axis=1)
    _alchemical_distances = np.sqrt(_distances_to_alchemical_particle ** 2 + w ** 2)
    alchemical_distances = apply_cutoff(_alchemical_distances, params.cutoff)

    # LJ terms
    pairwise_lj = lennard_jones(pairwise_distances, params.sigma, params.epsilon)
    alchemical_lj = lennard_jones(alchemical_distances, params.sigma, params.epsilon)

    # restraint terms
    origin_distances = norm(x, axis=1)
    restraints = flat_bottom_restraint(origin_distances, params.radius)

    # sum up and return scalar
    return np.sum(pairwise_lj) + np.sum(alchemical_lj) + np.sum(restraints)


def md(x0, grad_log_pi, n_steps=1000, dt=1e-2, gamma=1.0, save_interval=10):
    traj = [np.array(x0)]

    sample_gaussian = lambda: onp.random.randn(*x0.shape)

    x = np.array(x0)
    v = sample_gaussian()

    g = grad_log_pi(x)

    a, b = np.exp(-gamma * dt), np.sqrt(1 - np.exp(-2 * gamma * dt))

    @jit
    def update(x, v, g, noise):
        v1 = v + 0.5 * dt * g
        x2 = x + 0.5 * dt * v1
        v3 = a * v1 + b * noise
        x4 = x2 + 0.5 * dt * v3
        g4 = grad_log_pi(x4)
        v5 = v3 + 0.5 * dt * g4

        return x4, v5, g4

    for i in tqdm(range(n_steps)):
        noise = sample_gaussian()
        x, v, g = update(x, v, g, noise)
        if i % save_interval == 0:
            traj.append(np.array(x))

    return traj
