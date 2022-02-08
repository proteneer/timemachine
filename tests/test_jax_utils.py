import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp
from jax import jit
from jax import numpy as np
from jax import vmap

onp.random.seed(2021)

from timemachine.potentials.jax_utils import (
    augment_dim,
    batched_neighbor_inds,
    compute_lifting_parameter,
    delta_r,
    distance_on_pairs,
    get_all_pairs_indices,
    get_group_group_indices,
)


def test_delta_r():
    """assert that
    * delta_r(ri, rj, box) == - delta_r(rj, ri, box)
    * delta_r agrees with jit(delta_r)
    * jit(norm(delta_r)) symmetric
    on a few random inputs of varying size
    """

    @jit
    def _distances(ri, rj, box):
        return np.linalg.norm(delta_r(ri, rj, box), axis=1)

    for _ in range(5):
        n_atoms = onp.random.randint(50, 1000)
        dim = onp.random.randint(3, 5)
        ri, rj = onp.random.randn(2, n_atoms, dim)
        box = np.eye(dim)

        dr_ij_1 = delta_r(ri, rj, box)
        dr_ji_1 = delta_r(rj, ri, box)

        onp.testing.assert_allclose(dr_ij_1, -dr_ji_1)

        dr_ij_2 = jit(delta_r)(ri, rj, box)
        dr_ji_2 = jit(delta_r)(rj, ri, box)

        onp.testing.assert_allclose(dr_ij_1, dr_ij_2)
        onp.testing.assert_allclose(dr_ji_1, dr_ji_2)

        dij = _distances(ri, rj, box)
        dji = _distances(rj, ri, box)

        onp.testing.assert_allclose(dij, dji)


def test_get_all_pairs_indices():
    """check i < j < n"""
    ns = onp.random.randint(5, 50, 10)
    for n in ns:
        inds_i, inds_j = get_all_pairs_indices(n)
        assert (inds_i < inds_j).all()
        assert (inds_j < n).all()


def test_get_group_group_indices():
    """check i < n, j < m"""
    ns = onp.random.randint(5, 50, 10)
    ms = onp.random.randint(5, 50, 10)

    for n, m in zip(ns, ms):
        inds_i, inds_j = get_group_group_indices(n, m)
        assert (inds_i < n).all()
        assert (inds_j < m).all()

        assert len(inds_i) == n * m


def test_compute_lifting_parameter():
    """check expected behavior at lambda=0, lambda=1 for combinations of
    lambda_plane_idx, lambda_offset_idxs in [-1, 0, +1]"""
    cutoff = 5.0

    lambda_plane_idxs = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    lambda_offset_idxs = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])

    w0 = compute_lifting_parameter(0.0, lambda_plane_idxs, lambda_offset_idxs, cutoff)
    onp.testing.assert_allclose(w0, cutoff * lambda_plane_idxs)

    w1 = compute_lifting_parameter(1.0, lambda_plane_idxs, lambda_offset_idxs, cutoff)
    onp.testing.assert_allclose(w1, cutoff * (lambda_offset_idxs + lambda_plane_idxs))


def test_augment_dim():
    """check xyz -> xyzw stacking"""
    for _ in range(5):
        n = onp.random.randint(5, 10)
        xyz = onp.random.randn(n, 3)
        w = onp.random.randn(n)

        xyzw = augment_dim(xyz, w)
        onp.testing.assert_allclose(xyzw[:, :3], xyz)
        onp.testing.assert_allclose(xyzw[:, -1], w)


def test_batched_neighbor_inds():
    """compute n x m distances on each of a batch of confs,
    get fixed-length "neighbor lists" using padded index masks,
    and assert that the same number of pairs is within cutoff for
    original and masked variants
    """
    n_confs, n_particles, dim = 100, 1000, 3

    confs = onp.random.rand(n_confs, n_particles, dim)
    cutoff = 0.3

    boxes = np.array([np.eye(3)] * n_confs)

    n_alchemical = 50
    inds_l, inds_r = get_group_group_indices(n=n_alchemical, m=n_particles - n_alchemical)
    inds_r += n_alchemical
    n_possible_interactions = len(inds_l)

    full_distances = vmap(distance_on_pairs)(confs[:, inds_l], confs[:, inds_r], boxes)
    assert full_distances.shape == (n_confs, n_possible_interactions)

    neighbor_inds_l, neighbor_inds_r = batched_neighbor_inds(confs, inds_l, inds_r, cutoff, boxes)
    n_neighbor_pairs = neighbor_inds_l.shape[1]
    assert neighbor_inds_r.shape == (n_confs, n_neighbor_pairs)
    assert n_neighbor_pairs <= n_possible_interactions

    def d(conf, inds_l, inds_r, box):
        return distance_on_pairs(conf[inds_l], conf[inds_r], box)

    neighbor_distances = vmap(d)(confs, neighbor_inds_l, neighbor_inds_r, boxes)

    assert neighbor_distances.shape == (n_confs, n_neighbor_pairs)
    assert np.sum(neighbor_distances < cutoff) == np.sum(full_distances < cutoff)
