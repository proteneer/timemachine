import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as np, jit
import numpy as onp

onp.random.seed(2021)

from timemachine.potentials.jax_utils import (
    delta_r, get_all_pairs_indices, get_group_group_indices,
    compute_lifting_parameter, augment_dim
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
    ns = onp.random.randint(5, 50, 10)
    for n in ns:
        inds_i, inds_j = get_all_pairs_indices(n)
        assert (inds_i < inds_j).all()
        assert (inds_j < n).all()


def test_get_group_group_indices():
    ns = onp.random.randint(5, 50, 10)
    ms = onp.random.randint(5, 50, 10)

    for n, m in zip(ns, ms):
        inds_i, inds_j = get_group_group_indices(n, m)
        assert (inds_i < n).all()
        assert (inds_j < m).all()

        assert len(inds_i) == n * m


def test_compute_lifting_parameter():
    cutoff = 5.0

    lambda_plane_idxs = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    lambda_offset_idxs = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])

    w0 = compute_lifting_parameter(0.0, lambda_plane_idxs, lambda_offset_idxs, cutoff)
    onp.testing.assert_allclose(w0, cutoff * lambda_plane_idxs)

    w1 = compute_lifting_parameter(1.0, lambda_plane_idxs, lambda_offset_idxs, cutoff)
    onp.testing.assert_allclose(w1, cutoff * (lambda_offset_idxs + lambda_plane_idxs))


def test_augment_dim():
    for _ in range(5):
        n = onp.random.randint(5, 10)
        xyz = onp.random.randn(n, 3)
        w = onp.random.randn(n)

        xyzw = augment_dim(xyz, w)
        onp.testing.assert_allclose(xyzw[:, :3], xyz)
        onp.testing.assert_allclose(xyzw[:, -1], w)
