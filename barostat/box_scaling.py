import jax.numpy as jnp


def rescale(coords, scale=1.0):
    return coords * scale


def compute_centroid(group, weights=None):
    if weights is not None:
        assert (not (weights < 0).any())
        normalized_weights = weights / jnp.sum(weights)
        return jnp.mean(normalized_weights * group, axis=0)

    return jnp.mean(group, axis=0)


def compute_centroids(coords, group_inds, weights=None):
    return jnp.array([compute_centroid(coords[inds], weights) for inds in group_inds])


def displace_by_group(coords, group_inds, displacements):
    # note: this implementation will fail silently for overlapping group_inds
    # TODO: assert group_inds not overlapping

    displaced_coords = jnp.array(coords)
    for (inds, displacement) in zip(group_inds, displacements):
        displaced_coords[inds] += displacement

    return displaced_coords


def scale_centroids(coords, group_inds, scale, weights=None):
    """

    Notes
    -----
    * Currently ignores particle mass -- the centroid of a group of
        particles will be computed assuming all particles have equal weight
    * Later, particle weights could be set in some arbitrary way within each group
    """

    centroids = compute_centroids(coords, group_inds, weights)
    group_displacements = rescale(centroids, scale) - centroids
    displaced_coords = displace_by_group(coords, group_inds, group_displacements)

    return displaced_coords
