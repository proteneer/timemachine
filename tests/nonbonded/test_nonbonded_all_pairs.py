import functools

import numpy as np
import pytest
from common import GradientTest
from parameter_interpolation import gen_params

from timemachine.lib.potentials import NonbondedAllPairs, NonbondedAllPairsInterpolated
from timemachine.potentials import nonbonded

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_all_pairs_invalid_planes_offsets():
    with pytest.raises(RuntimeError) as e:
        NonbondedAllPairs([0], [0, 0], 2.0, 1.1).unbound_impl(np.float32)

    assert "lambda offset idxs and plane idxs need to be equivalent" in str(e)


def test_nonbonded_all_pairs_invalid_atom_idxs():
    with pytest.raises(RuntimeError) as e:
        NonbondedAllPairs([0, 1], [0], 2.0, 1.1, [0, 0]).unbound_impl(np.float32)

    assert "atom indices must be unique" in str(e)


def test_nonbonded_all_pairs_invalid_num_atoms():
    potential = NonbondedAllPairs([0], [0], 2.0, 1.1).unbound_impl(np.float32)
    with pytest.raises(RuntimeError) as e:
        potential.execute(np.zeros((2, 3)), np.zeros((1, 3)), np.eye(3), 0)

    assert "NonbondedAllPairs::execute_device(): expected N == N_, got N=2, N_=1" in str(e)


def test_nonbonded_all_pairs_invalid_num_params():
    potential = NonbondedAllPairs([0], [0], 2.0, 1.1).unbound_impl(np.float32)
    with pytest.raises(RuntimeError) as e:
        potential.execute(np.zeros((1, 3)), np.zeros((2, 3)), np.eye(3), 0)

    assert "NonbondedAllPairs::execute_device(): expected P == M*N_*3, got P=6, M*N_*3=3" in str(e)

    potential_interp = NonbondedAllPairsInterpolated([0], [0], 2.0, 1.1).unbound_impl(np.float32)
    with pytest.raises(RuntimeError) as e:
        potential_interp.execute(np.zeros((1, 3)), np.zeros((1, 3)), np.eye(3), 0)

    assert "NonbondedAllPairs::execute_device(): expected P == M*N_*3, got P=3, M*N_*3=6" in str(e)


def make_ref_potential(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, atom_idxs, interpolated):

    s = atom_idxs if atom_idxs is not None else slice(None)

    @functools.wraps(nonbonded.nonbonded_v3)
    def wrapped(conf, params, box, lamb):
        conf_ = conf[s, :]
        num_atoms, _ = conf_.shape
        no_rescale = np.ones((num_atoms, num_atoms))
        return nonbonded.nonbonded_v3(
            conf_,
            params[s, :],
            box,
            lamb,
            charge_rescale_mask=no_rescale,
            lj_rescale_mask=no_rescale,
            beta=beta,
            cutoff=cutoff,
            lambda_plane_idxs=lambda_plane_idxs[s],
            lambda_offset_idxs=lambda_offset_idxs[s],
        )

    return nonbonded.interpolated(wrapped) if interpolated else wrapped


def test_nonbonded_all_pairs_singleton_subset(rng: np.random.Generator):
    """Checks that energy and derivatives are all zero when called with a single-atom subset"""
    num_atoms = 231
    beta = 2.0
    lamb = 0.1
    cutoff = 1.1
    box = 3.0 * np.eye(3)
    conf = rng.uniform(0, 1, size=(num_atoms, 3))
    params = rng.uniform(0, 1, size=(num_atoms, 3))

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    for idx in rng.choice(num_atoms, size=(10,)):
        atom_idxs = np.array([idx], dtype=np.uint32)
        potential = NonbondedAllPairs(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, atom_idxs)
        du_dx, du_dp, du_dl, u = potential.unbound_impl(np.float64).execute(conf, params, box, lamb)

        assert (du_dx == 0).all()
        assert (du_dp == 0).all()
        assert du_dl == 0
        assert u == 0


def test_nonbonded_all_pairs_improper_subset(rng: np.random.Generator):
    """Checks for bitwise equivalence of the following cases:
    1. atom_idxs = None
    2. atom_idxs = range(num_atoms)
    """
    num_atoms = 231
    beta = 2.0
    lamb = 0.1
    cutoff = 1.1
    box = 3.0 * np.eye(3)
    conf = rng.uniform(0, 1, size=(num_atoms, 3))
    params = rng.uniform(0, 1, size=(num_atoms, 3))

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    def test_impl(atom_idxs):
        return (
            NonbondedAllPairs(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, atom_idxs)
            .unbound_impl(np.float64)
            .execute(conf, params, box, lamb)
        )

    du_dx_1, du_dp_1, du_dl_1, u_1 = test_impl(None)
    du_dx_2, du_dp_2, du_dl_2, u_2 = test_impl(np.arange(num_atoms, dtype=np.uint32))

    np.testing.assert_array_equal(du_dx_1, du_dx_2)
    np.testing.assert_array_equal(du_dp_1, du_dp_2)
    np.testing.assert_array_equal(du_dl_1, du_dl_2)
    assert u_1 == u_2


def test_nonbonded_all_pairs_set_atom_idxs(rng: np.random.Generator):
    num_atoms = 100
    beta = 2.0
    lamb = 0.1
    cutoff = 1.1
    box = 3.0 * np.eye(3)
    conf = rng.uniform(0, 1, size=(num_atoms, 3))
    params = rng.uniform(0, 1, size=(num_atoms, 3))

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    atom_idxs = np.arange(0, num_atoms // 2, dtype=np.uint32)

    potential = NonbondedAllPairs(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff)
    unbound_pot = potential.unbound_impl(np.float32)

    identity_idxs = np.arange(0, num_atoms, dtype=np.uint32)
    for num_idxs in [5, 25, 50, 80, num_atoms]:
        atom_idxs = rng.choice(num_atoms, size=(num_idxs,), replace=False).astype(np.uint32)
        ignored_idxs = np.delete(identity_idxs, atom_idxs)
        unbound_pot.set_atom_idxs(atom_idxs)

        ref_potential = NonbondedAllPairs(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, atom_idxs)
        unbound_ref = ref_potential.unbound_impl(np.float32)

        du_dx, du_dl, du_dp, u = unbound_pot.execute(conf, params, box, lamb)
        du_dx_2, du_dl_2, du_dp_2, u_2 = unbound_pot.execute(conf, params, box, lamb)
        ref_du_dx, ref_du_dl, ref_du_dp, ref_u = unbound_ref.execute(conf, params, box, lamb)

        np.testing.assert_array_equal(du_dx_2, du_dx)
        np.testing.assert_array_equal(du_dp_2, du_dp)
        np.testing.assert_equal(du_dp_2, du_dp)
        np.testing.assert_equal(u_2, u)

        # Atoms that are ignored by the potential, should always return 0.0
        assert np.all(du_dx[ignored_idxs] == 0.0)
        assert np.all(du_dl[ignored_idxs] == 0.0)

        np.testing.assert_array_equal(ref_du_dx, du_dx)
        np.testing.assert_array_equal(ref_du_dp, du_dp)
        np.testing.assert_equal(ref_du_dp, du_dp)
        np.testing.assert_equal(ref_u, u)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms_subset", [None, 33])
@pytest.mark.parametrize("num_atoms", [33, 65, 231])
@pytest.mark.parametrize("interpolated", [False, True])
def test_nonbonded_all_pairs_correctness(
    interpolated,
    num_atoms,
    num_atoms_subset,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    "Compares with jax reference implementation."

    conf = example_conf[:num_atoms]
    params_initial = example_nonbonded_potential.params[:num_atoms, :]
    params = gen_params(params_initial, rng) if interpolated else params_initial

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    atom_idxs = (
        rng.choice(num_atoms, size=(num_atoms_subset,), replace=False).astype(np.uint32) if num_atoms_subset else None
    )

    ref_potential = make_ref_potential(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, atom_idxs, interpolated)

    make_test_potential = NonbondedAllPairsInterpolated if interpolated else NonbondedAllPairs
    test_potential = make_test_potential(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, atom_idxs)
    lambda_values = [0.0, 0.1]
    GradientTest().compare_forces(
        conf,
        params,
        example_box,
        lambda_values,
        ref_potential,
        test_potential,
        precision=precision,
        rtol=rtol,
        atol=atol,
    )
