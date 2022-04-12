import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import vmap

np.random.seed(2021)

from timemachine.potentials.jax_utils import (
    augment_dim,
    compute_lifting_parameter,
    delta_r,
    distance_on_pairs,
    get_all_pairs_indices,
    get_interacting_pair_indices_batch,
    pairs_from_interaction_groups,
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
        return jnp.linalg.norm(delta_r(ri, rj, box), axis=1)

    for _ in range(5):
        n_atoms = np.random.randint(50, 1000)
        dim = np.random.randint(3, 5)
        ri, rj = np.random.randn(2, n_atoms, dim)
        box = np.eye(dim)

        dr_ij_1 = delta_r(ri, rj, box)
        dr_ji_1 = delta_r(rj, ri, box)

        np.testing.assert_allclose(dr_ij_1, -dr_ji_1)

        dr_ij_2 = jit(delta_r)(ri, rj, box)
        dr_ji_2 = jit(delta_r)(rj, ri, box)

        np.testing.assert_allclose(dr_ij_1, dr_ij_2)
        np.testing.assert_allclose(dr_ji_1, dr_ji_2)

        dij = _distances(ri, rj, box)
        dji = _distances(rj, ri, box)

        np.testing.assert_allclose(dij, dji)


def test_get_all_pairs_indices():
    """check i < j < n"""
    ns = np.random.randint(5, 50, 10)
    for n in ns:
        pairs = get_all_pairs_indices(n)
        assert (pairs[:, 0] < pairs[:, 1]).all()
        assert (pairs < n).all()


def test_get_pairs_from_interaction_groups_indices():
    """on random instances of varying size, assert expected number and identity of interacting pairs"""
    num_instances = 10
    ns = np.random.randint(5, 50, num_instances)
    ms = np.random.randint(5, 50, num_instances)

    for n, m in zip(ns, ms):
        atom_indices = np.arange(n + m)

        np.random.shuffle(atom_indices)  # non-contiguous group indices
        group_a_indices = atom_indices[:n]
        group_b_indices = atom_indices[n:]

        pairs = pairs_from_interaction_groups(group_a_indices, group_b_indices)
        assert set(pairs[:, 0]) == set(group_a_indices)
        assert set(pairs[:, 1]) == set(group_b_indices)

        assert len(pairs) == n * m


def test_compute_lifting_parameter():
    """check expected behavior at lambda=0, lambda=1 for combinations of
    lambda_plane_idx, lambda_offset_idxs in [-1, 0, +1]"""
    cutoff = 5.0

    lambda_plane_idxs = jnp.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    lambda_offset_idxs = jnp.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])

    w0 = compute_lifting_parameter(0.0, lambda_plane_idxs, lambda_offset_idxs, cutoff)
    np.testing.assert_allclose(w0, cutoff * lambda_plane_idxs)

    w1 = compute_lifting_parameter(1.0, lambda_plane_idxs, lambda_offset_idxs, cutoff)
    np.testing.assert_allclose(w1, cutoff * (lambda_offset_idxs + lambda_plane_idxs))


def test_augment_dim():
    """check xyz -> xyzw stacking"""
    for _ in range(5):
        n = np.random.randint(5, 10)
        xyz = np.random.randn(n, 3)
        w = np.random.randn(n)

        xyzw = augment_dim(xyz, w)
        np.testing.assert_allclose(xyzw[:, :3], xyz)
        np.testing.assert_allclose(xyzw[:, -1], w)


def test_batched_neighbor_inds():
    """compute n x m distances on each of a batch of confs,
    get fixed-length "neighbor lists" using padded index masks,
    and assert that the same number of pairs is within cutoff for
    original and masked variants
    """
    n_confs, n_particles, dim = 100, 1000, 3

    confs = np.random.rand(n_confs, n_particles, dim)
    cutoff = 0.3

    boxes = jnp.array([np.eye(3)] * n_confs)

    n_alchemical = 50
    pairs = pairs_from_interaction_groups(np.arange(n_alchemical), np.arange(n_alchemical, n_particles))
    n_possible_interactions = len(pairs)

    full_distances = vmap(distance_on_pairs)(confs[:, pairs[:, 0]], confs[:, pairs[:, 1]], boxes)
    assert full_distances.shape == (n_confs, n_possible_interactions)

    batch_pairs = get_interacting_pair_indices_batch(confs, boxes, pairs, cutoff)
    n_neighbor_pairs = batch_pairs.shape[1]
    assert batch_pairs.shape == (n_confs, n_neighbor_pairs, 2)
    assert n_neighbor_pairs <= n_possible_interactions

    def d(conf, pairs, box):
        return distance_on_pairs(conf[pairs[:, 0]], conf[pairs[:, 1]], box)

    neighbor_distances = vmap(d)(confs, batch_pairs, boxes)

    assert neighbor_distances.shape == (n_confs, n_neighbor_pairs)
    assert np.sum(neighbor_distances < cutoff) == np.sum(full_distances < cutoff)
