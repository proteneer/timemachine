# TODO: molecular ideal gas test system

from md.barostat.moves import CentroidRescaler

import numpy as onp

from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as jnp


def test_compute_centroids():
    """test that CentroidRescaler's compute_centroids agrees with _slow_compute_centroids
    on random instances of varying size"""

    onp.random.seed(2021)

    for _ in range(10):
        # randomly generate point set of size between 50 and 1000
        n_particles = onp.random.randint(50, 1000)
        particle_inds = onp.arange(n_particles)

        # randomly generate group_inds with group sizes between 1 and 10
        group_inds = []
        onp.random.shuffle(particle_inds)
        i = 0
        while i < len(particle_inds):
            j = min(n_particles, i + onp.random.randint(1, 10))
            group_inds.append(jnp.array(particle_inds[i: j]))
            i = j

        # randomly generate coords
        coords = jnp.array(onp.random.randn(n_particles, 3))

        # assert compute_centroids agrees with _slow_compute_centroids
        rescaler = CentroidRescaler(group_inds)
        fast_centroids = rescaler.compute_centroids(coords)
        slow_centroids = rescaler._slow_compute_centroids(coords)
        onp.testing.assert_array_almost_equal(slow_centroids, fast_centroids)
