import jax.numpy as np
import numpy as onp
import jax
from jax.ops import index_update

from typing import Tuple
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


def rescale_coordinates(
    conf,
    indices,
    box,
    scales):
    """Note: scales unused"""

    mol_sizes = np.expand_dims(onp.bincount(indices), axis=1)
    mol_centers = jax.ops.segment_sum(conf, indices)/mol_sizes

    new_centers = mol_centers - box[2]*np.floor(np.expand_dims(mol_centers[...,2], axis=-1)/box[2][2])
    new_centers -= box[1]*np.floor(np.expand_dims(new_centers[...,1], axis=-1)/box[1][1])
    new_centers -= box[0]*np.floor(np.expand_dims(new_centers[...,0], axis=-1)/box[0][0])

    offset = new_centers - mol_centers

    return conf + offset[indices]

def delta_r(ri, rj, box=None):
    diff = ri - rj # this can be either N,N,3 or B,3
    dims = ri.shape[-1]

    # box is None for harmonic bonds, not None for nonbonded terms
    if box is not None:

        # Equivalent to:
        #    for d in range(dims):
        #       diff -= box[d]*np.floor(np.expand_dims(diff[...,d], axis=-1)/box[d][d]+0.5)
        def loop_body(d, diff):
            return diff - box[d] * np.floor(np.expand_dims(diff[..., d], axis=-1) / box[d][d] + 0.5)
        diff = jax.lax.fori_loop(lower=0, upper=dims, body_fun=loop_body, init_val=diff)

    return diff


def distance_on_pairs(ri, rj, box=None):
    """O(n) where n = len(ri) = len(rj)"""
    assert len(ri) == len(rj)

    diff = delta_r(ri, rj, box)
    dij = np.linalg.norm(diff, axis=-1)

    assert len(dij) == len(ri)

    return dij


def distance(x, box):
    # nonbonded distances require the periodic box
    assert x.shape[1] == 3 or x.shape[1] == 4 # 3d or 4d
    ri = np.expand_dims(x, 0)
    rj = np.expand_dims(x, 1)
    d2ij = np.sum(np.power(delta_r(ri, rj, box), 2), axis=-1)
    N = d2ij.shape[0]
    d2ij = np.where(np.eye(N), 0, d2ij)
    dij = np.where(np.eye(N), 0, np.sqrt(d2ij))
    return dij