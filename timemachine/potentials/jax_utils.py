"""Contents:

index stuff
* get_all_pairs_indices
* get_group_group_indices
* batched_neighbor_inds
* get_ligand_dependent_indices_batch

distance stuff
* delta_r
* distance
* distance_on_pairs

4D decoupling stuff
* compute_lifting_parameter
* augment_dim
* convert_to_4d
"""

from typing import Tuple

import jax
import jax.numpy as np
import numpy as onp
from jax import vmap

Array = onp.array


def get_all_pairs_indices(n: int) -> Tuple[Array, Array]:
    """all indices i, j such that i < j < n"""
    n_interactions = n * (n - 1) / 2

    inds_i, inds_j = np.triu_indices(n, k=1)

    assert len(inds_i) == n_interactions

    return inds_i, inds_j


def get_group_group_indices(n: int, m: int) -> Tuple[Array, Array]:
    """all indices i, j such that i < n, j < m"""
    n_interactions = n * m

    _inds_i, _inds_j = np.indices((n, m))
    inds_i, inds_j = _inds_i.flatten(), _inds_j.flatten()

    assert len(inds_i) == n_interactions

    return inds_i, inds_j


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


def rescale_coordinates(conf, indices, box, scales):
    """Note: scales unused"""

    mol_sizes = np.expand_dims(onp.bincount(indices), axis=1)
    mol_centers = jax.ops.segment_sum(conf, indices) / mol_sizes

    new_centers = mol_centers - box[2] * np.floor(np.expand_dims(mol_centers[..., 2], axis=-1) / box[2][2])
    new_centers -= box[1] * np.floor(np.expand_dims(new_centers[..., 1], axis=-1) / box[1][1])
    new_centers -= box[0] * np.floor(np.expand_dims(new_centers[..., 0], axis=-1) / box[0][0])

    offset = new_centers - mol_centers

    return conf + offset[indices]


def delta_r(ri, rj, box=None):
    diff = ri - rj  # this can be either N,N,3 or B,3

    # box is None for harmonic bonds, not None for nonbonded terms
    if box is not None:
        box_diag = np.diag(box)
        diff -= box_diag * np.floor(diff / box_diag + 0.5)
    return diff


def distance_on_pairs(ri, rj, box=None):
    """O(n) where n = len(ri) = len(rj)"""
    assert len(ri) == len(rj)

    diff = delta_r(ri, rj, box)
    dij = np.linalg.norm(diff, axis=-1)

    assert len(dij) == len(ri)

    return dij


def batched_neighbor_inds(confs, inds_l, inds_r, cutoff, boxes):
    """Given candidate interacting pairs (inds_l, inds_r),
        inds_l.shape == n_interactions
    exclude most pairs whose distances are >= cutoff (neighbor_inds_l, neighbor_inds_r)
        neighbor_inds_l.shape == (len(confs), max_n_neighbors)
        where the total number of neighbors returned for each conf in confs is the same
        max_n_neighbors

    This padding causes some amount of wasted effort, but keeps things nice and fixed-dimensional
        for later XLA steps
    """
    assert len(confs.shape) == 3
    distances = vmap(distance_on_pairs)(confs[:, inds_l], confs[:, inds_r], boxes)
    assert distances.shape == (len(confs), len(inds_l))

    neighbor_masks = distances < cutoff
    # how many total neighbors?

    n_neighbors = np.sum(neighbor_masks, 1)
    max_n_neighbors = max(n_neighbors)

    assert max_n_neighbors > 0

    # sorting in order of [falses, ..., trues]
    keep_inds = np.argsort(neighbor_masks, axis=1)[:, -max_n_neighbors:]
    neighbor_inds_l = inds_l[keep_inds]
    neighbor_inds_r = inds_r[keep_inds]

    assert neighbor_inds_l.shape == (len(confs), max_n_neighbors)
    assert neighbor_inds_l.shape == neighbor_inds_r.shape

    return neighbor_inds_l, neighbor_inds_r


def get_ligand_dependent_indices_batch(confs, boxes, ligand_indices, cutoff=1.2):
    """Find atom pairs that depend on ligand and contribute to nonbonded sum.

    For conf in confs, find atom pairs (i, j) where either:
    * i in ligand_indices, j in environment_indices, and distance(i, j) < cutoff
    * i in ligand_indices, j in ligand_indices, and i < j

    Parameters
    ----------
    confs: (n_snapshots, n_atoms, 3) float array
    boxes: (n_snapshots, 3, 3) float array
    ligand_indices: (n_ligand) int array
    cutoff: float

    Returns
    -------
    (batch_inds_l, batch_inds_r)
        each of shape (len(confs), n_pairs),
        where n_pairs = maximum number of interacting pairs in confs

    Notes
    -----
    * Index arrays are padded so each conf has the same number of interacting pairs -- a small fraction of the returned
        pairs ij will have distance(i, j) > cutoff, so these may need to be filtered / masked again at later steps
    * TODO [naming]: change to return a single [n_pairs, 2] array instead of pair of [n_pairs,] arrays?
    * TODO [flexibility]: accept environment_indices instead of inferring them?
    """
    n_snapshots, n_atoms, _ = confs.shape
    environment_indices = np.array(list(set(onp.arange(n_atoms)) - set(onp.array(ligand_indices))))

    # (ligand, environment) pairs within distance cutoff
    _inds_l, _inds_r = get_group_group_indices(len(ligand_indices), len(environment_indices))
    inds_l, inds_r = ligand_indices[_inds_l], environment_indices[_inds_r]
    neighbor_inds_l, neighbor_inds_r = batched_neighbor_inds(confs, inds_l, inds_r, cutoff, boxes)

    # (ligand, ligand) pairs
    _l, _r = get_all_pairs_indices(len(ligand_indices))
    ligand_inds_l, ligand_inds_r = ligand_indices[_l], ligand_indices[_r]

    # concatenate
    batch_inds_l = np.hstack([neighbor_inds_l, np.repeat(ligand_inds_l[np.newaxis, :], n_snapshots, 0)])
    batch_inds_r = np.hstack([neighbor_inds_r, np.repeat(ligand_inds_r[np.newaxis, :], n_snapshots, 0)])

    assert batch_inds_l.shape == batch_inds_r.shape

    return batch_inds_l, batch_inds_r


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
