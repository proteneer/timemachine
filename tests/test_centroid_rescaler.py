import numpy as np
from md.barostat.moves import CentroidRescaler

def test_compute_centroids():
    """test that CentroidRescaler's compute_centroids agrees with _slow_compute_centroids
    on random instances of varying size"""

    np.random.seed(2021)

    for _ in range(10):
        # randomly generate point set of size between 50 and 1000
        n_particles = np.random.randint(50, 1000)
        particle_inds = np.arange(n_particles)

        # randomly generate group_inds with group sizes between 1 and 10
        group_inds = []
        np.random.shuffle(particle_inds)
        i = 0
        while i < len(particle_inds):
            j = min(n_particles, i + np.random.randint(1, 10))
            group_inds.append(np.array(particle_inds[i: j]))
            i = j

        # randomly generate coords
        coords = np.array(np.random.randn(n_particles, 3))

        # assert compute_centroids agrees with _slow_compute_centroids
        rescaler = CentroidRescaler(group_inds)
        fast_centroids = rescaler.compute_centroids(coords)
        slow_centroids = rescaler._slow_compute_centroids(coords)
        np.testing.assert_array_almost_equal(slow_centroids, fast_centroids)

