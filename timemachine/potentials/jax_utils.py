import jax.numpy as np
import numpy as onp
from jax import vmap

Array = onp.array


def get_all_pairs_indices(n: int) -> Array:
    """all indices i, j such that i < j < n"""
    n_interactions = n * (n - 1) / 2

    pairs = np.stack(np.triu_indices(n, k=1)).T

    assert pairs.shape == (n_interactions, 2)

    return pairs


def pairs_from_interaction_groups(group_a_indices: Array, group_b_indices: Array) -> Array:
    """(a, b) for a in group_a_indices, b in group_b_indices"""
    n_interactions = len(group_a_indices) * len(group_b_indices)

    pairs = np.stack(np.meshgrid(group_a_indices, group_b_indices)).reshape(2, -1).T

    assert pairs.shape == (n_interactions, 2)

    return pairs


def compute_lifting_parameter(lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff):
    """One way to compute a per-particle "4D" offset in terms of an adjustable lamb and
    constant per-particle parameters.

    Notes
    -----
    (ytz): this initializes the 4th dimension to a fixed plane adjust by an offset
    followed by a scaling by cutoff.

    lambda_plane_idxs are typically 0 or 1 and allows us to turn off an interaction
    independent of the lambda value.

    lambda_offset_idxs are typically 0 and 1, and allows us to adjust the w coordinate
    in a lambda-dependent way.
    """

    w = cutoff * (lambda_plane_idxs + lambda_offset_idxs * lamb)
    return w


def augment_dim(x3: Array, w: Array) -> Array:
    """(x,y,z) -> (x,y,z,w)"""

    d4 = np.expand_dims(w, axis=-1)
    x4 = np.concatenate((x3, d4), axis=1)

    assert len(x4) == len(x3)
    assert x4.shape[1] == 4

    return x4


def convert_to_4d(x3, lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff):
    """(x,y,z) -> (x,y,z,w) where w = cutoff * (lambda_plane_idxs + lambda_offset_idxs * lamb)"""
    w = compute_lifting_parameter(lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)
    return augment_dim(x3, w)


def delta_r(ri, rj, box=None):
    diff = ri - rj  # this can be either N,N,3 or B,3

    # box is None for harmonic bonds, not None for nonbonded terms
    if box is not None:
        box_diag = np.diag(box)
        diff -= box_diag * np.floor(diff / box_diag + 0.5)
    return diff


def distance_on_pairs(ri, rj, box=None):
    """O(n) where n = len(ri) = len(rj)

    Notes
    -----
    TODO [performance]: any difference if the signature is (conf, pairs) rather than (ri, rj)?
    """
    assert len(ri) == len(rj)

    diff = delta_r(ri, rj, box)
    dij = np.linalg.norm(diff, axis=-1)

    assert len(dij) == len(ri)

    return dij


def get_interacting_pair_indices_batch(confs, boxes, pairs, cutoff=1.2):
    """Given candidate interacting pairs, exclude most pairs whose distances are >= cutoff

    Parameters
    ----------
    confs: (n_snapshots, n_atoms, dim) float array
    boxes: (n_snapshots, dim, dim) float array
    pairs: (n_candidate_pairs, 2) integer array
    cutoff: float

    Returns
    -------
    batch_pairs : (len(confs), max_n_neighbors, 2) array
        where max_n_neighbors pairs are returned for each conf in confs

    Notes
    -----
    * Padding causes some amount of wasted effort, but keeps things nice and fixed-dimensional for later XLA steps
    """
    n_snapshots, n_atoms, dim = confs.shape
    assert boxes.shape == (n_snapshots, dim, dim)

    distances = vmap(distance_on_pairs)(confs[:, pairs[:, 0]], confs[:, pairs[:, 1]], boxes)
    assert distances.shape == (len(confs), len(pairs))

    neighbor_masks = distances < cutoff
    # how many total neighbors?

    n_neighbors = np.sum(neighbor_masks, 1)
    max_n_neighbors = max(n_neighbors)

    assert max_n_neighbors > 0

    # sorting in order of [falses, ..., trues]
    keep_inds = np.argsort(neighbor_masks, axis=1)[:, -max_n_neighbors:]
    batch_pairs = pairs[keep_inds]

    assert batch_pairs.shape == (len(confs), max_n_neighbors, 2)

    return batch_pairs


def distance(x, box):
    # nonbonded distances require the periodic box
    assert x.shape[1] == 3 or x.shape[1] == 4  # 3d or 4d
    ri = np.expand_dims(x, 0)
    rj = np.expand_dims(x, 1)
    d2ij = np.sum(np.power(delta_r(ri, rj, box), 2), axis=-1)
    N = d2ij.shape[0]
    d2ij = np.where(np.eye(N), 0, d2ij)
    dij = np.where(np.eye(N), 0, np.sqrt(d2ij))
    return dij
