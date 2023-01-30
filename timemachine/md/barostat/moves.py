import numpy as np
from jax import numpy as jnp
from jax.ops import segment_sum


def compute_centroid(group):
    return jnp.mean(group, axis=0)


def _scatter_inds_from_group_inds(group_inds):
    """
    given a list of arrays of ints, representing groups of particles,
    construct a flat array, representing which group index to sort each particle into

    [[0,1,2], [3,4,5]] --> [0, 0, 0, 1, 1, 1]
    """
    all_inds = np.hstack(group_inds)
    scatter_inds = np.zeros(len(all_inds))

    # assert group_inds not overlapping
    assert len(all_inds) == len(set(all_inds))

    for i, group in enumerate(group_inds):
        for j in group:
            scatter_inds[j] = i

    return np.array(scatter_inds, dtype=int)


class CentroidRescaler:
    def __init__(self, group_inds, weights=None):
        self.group_inds = group_inds
        self.group_sizes = jnp.array(list(map(len, self.group_inds)))
        assert jnp.min(self.group_sizes) > 0

        self.scatter_inds = _scatter_inds_from_group_inds(group_inds)

        if weights is not None:
            raise NotImplementedError("Weights are not implemented yet")

    def rescale(self, coords, center, scale=1.0):
        """scale distances of coords to center"""

        dx_initial = coords - center
        dx_updated = scale * dx_initial
        return center + dx_updated

    def compute_centroids(self, coords):
        """Returns an array containing the centroids of each group"""
        return segment_sum(coords, self.scatter_inds) / jnp.expand_dims(self.group_sizes, axis=1)

    def _slow_compute_centroids(self, coords):
        """For testing / reference"""
        return jnp.array([compute_centroid(coords[g]) for g in self.group_inds])

    def displace_by_group(self, coords, displacements):
        return coords + displacements[self.scatter_inds]

    def scale_centroids(self, coords, center, scale):
        """

        Notes
        -----
        * Currently ignores particle mass -- the centroid of a group of
            particles will be computed assuming all particles have equal weight
        * Later, particle weights could be set in some arbitrary way within each group
        """

        centroids = self.compute_centroids(coords)
        group_displacements = self.rescale(centroids, center, scale) - centroids
        displaced_coords = self.displace_by_group(coords, group_displacements)

        return displaced_coords
