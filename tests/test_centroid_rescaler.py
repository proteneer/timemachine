import numpy as np
from md.barostat.moves import CentroidRescaler
from md.barostat.utils import compute_intramolecular_distances

np.random.seed(2021)


def _generate_random_instance():
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

    return coords, group_inds


def test_null_rescaling(n_instances=10):
    """scaling by a factor of 1.0x shouldn't change coordinates"""
    for _ in range(n_instances):
        coords, group_inds = _generate_random_instance()
        center = np.random.randn(3)

        rescaler = CentroidRescaler(group_inds)
        coords_prime = rescaler.scale_centroids(coords, center, 1.0)

        np.testing.assert_allclose(coords_prime, coords)


def test_intramolecular_distance(n_instances=10):
    """Test that applying a rescaling doesn't change intramolecular distances"""
    for _ in range(n_instances):
        coords, group_inds = _generate_random_instance()
        distances = compute_intramolecular_distances(coords, group_inds)

        center = np.random.randn(3)
        scale = np.random.rand() + 0.5

        rescaler = CentroidRescaler(group_inds)
        coords_prime = rescaler.scale_centroids(coords, center, scale)
        distances_prime = compute_intramolecular_distances(coords_prime, group_inds)

        np.testing.assert_allclose(np.hstack(distances_prime), np.hstack(distances))


def test_compute_centroids(n_instances=10):
    """test that CentroidRescaler's compute_centroids agrees with _slow_compute_centroids
    on random instances of varying size"""

    for _ in range(n_instances):
        coords, group_inds = _generate_random_instance()

        # assert compute_centroids agrees with _slow_compute_centroids
        rescaler = CentroidRescaler(group_inds)
        fast_centroids = rescaler.compute_centroids(coords)
        slow_centroids = rescaler._slow_compute_centroids(coords)
        np.testing.assert_array_almost_equal(slow_centroids, fast_centroids)
