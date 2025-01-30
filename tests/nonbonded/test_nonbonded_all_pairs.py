import numpy as np
import pytest
from common import GradientTest, gen_nonbonded_params_with_4d_offsets

from timemachine.potentials import NonbondedAllPairs

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_all_pairs_invalid_atom_idxs():
    with pytest.raises(RuntimeError, match="indices can't be empty"):
        NonbondedAllPairs(3, 2.0, 1.1, []).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match="atom indices must be unique"):
        NonbondedAllPairs(3, 2.0, 1.1, [0, 0]).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match="index values must be greater or equal to zero"):
        NonbondedAllPairs(3, 2.0, 1.1, [0, -1]).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match="index values must be less than N"):
        NonbondedAllPairs(3, 2.0, 1.1, [0, 100]).to_gpu(np.float64).unbound_impl


def test_nonbonded_all_pairs_invalid_set_atom_idxs():
    nb_allpairs = NonbondedAllPairs(3, 2.0, 1.1).to_gpu(np.float32).unbound_impl
    with pytest.raises(RuntimeError, match="indices can't be empty"):
        nb_allpairs.set_atom_idxs([])

    with pytest.raises(RuntimeError, match="atom indices must be unique"):
        nb_allpairs.set_atom_idxs([0, 0])

    with pytest.raises(RuntimeError, match="index values must be greater or equal to zero"):
        nb_allpairs.set_atom_idxs([0, -1])

    with pytest.raises(RuntimeError, match="index values must be less than N"):
        nb_allpairs.set_atom_idxs([0, 100])


def test_nonbonded_all_pairs_invalid_num_atoms():
    potential = NonbondedAllPairs(1, 2.0, 1.1).to_gpu(np.float32).unbound_impl
    with pytest.raises(RuntimeError) as e:
        potential.execute(np.zeros((2, 3)), np.zeros((1, 3)), np.eye(3))
    assert "NonbondedAllPairs::execute_device(): expected N == N_, got N=2, N_=1" == str(e.value)


def test_nonbonded_all_pairs_invalid_num_params():
    potential = NonbondedAllPairs(1, 2.0, 1.1).to_gpu(np.float32).unbound_impl
    with pytest.raises(RuntimeError) as e:
        potential.execute(np.zeros((1, 3)), np.zeros((2, 3)), np.eye(3))
    assert "NonbondedAllPairs::execute_device(): expected P == N_*4, got P=6, N_*4=4" == str(e.value)


def test_nonbonded_all_pairs_get_and_set_atom_idxs(rng):
    """Verify that getting and setting atom idxs on the GPU implementation of NonbondedAllPairs
    behaves as expected
    """
    num_atoms = 231
    beta = 2.0
    cutoff = 1.1

    potential = NonbondedAllPairs(num_atoms, beta, cutoff)
    gpu_pot = potential.to_gpu(np.float32).unbound_impl
    ref_atom_idxs = gpu_pot.get_atom_idxs()
    assert len(ref_atom_idxs) == num_atoms
    assert num_atoms == gpu_pot.get_num_atom_idxs()
    assert len(set(ref_atom_idxs)) == len(ref_atom_idxs)
    identity_idxs = np.arange(0, num_atoms, dtype=np.int32)
    updated_idxs = rng.choice(identity_idxs, size=100, replace=False)
    gpu_pot.set_atom_idxs(updated_idxs)
    assert len(updated_idxs) == gpu_pot.get_num_atom_idxs()
    np.testing.assert_array_equal(gpu_pot.get_atom_idxs(), updated_idxs)


def test_nonbonded_all_pairs_singleton_subset(rng: np.random.Generator):
    """Checks that energy and derivatives are all zero when called with a single-atom subset"""
    num_atoms = 231
    beta = 2.0
    cutoff = 1.1
    box = 3.0 * np.eye(3)
    conf = rng.uniform(0, 1, size=(num_atoms, 3))
    params = rng.uniform(0, 1, size=(num_atoms, 4))

    for idx in rng.choice(num_atoms, size=(10,)):
        atom_idxs = np.array([idx], dtype=np.int32)
        potential = NonbondedAllPairs(num_atoms, beta, cutoff, atom_idxs)
        du_dx, du_dp, u = potential.to_gpu(np.float64).unbound_impl.execute(conf, params, box.astype(np.float64))

        assert (du_dx == 0).all()
        assert (du_dp == 0).all()
        assert u == 0


def test_nonbonded_all_pairs_improper_subset(rng: np.random.Generator):
    """Checks for bitwise equivalence of the following cases:
    1. atom_idxs = None
    2. atom_idxs = range(num_atoms)
    """
    num_atoms = 231
    beta = 2.0
    cutoff = 1.1
    box = 3.0 * np.eye(3)
    conf = rng.uniform(0, 1, size=(num_atoms, 3))
    params = rng.uniform(0, 1, size=(num_atoms, 4))

    def test_impl(atom_idxs):
        return (
            NonbondedAllPairs(num_atoms, beta, cutoff, atom_idxs)
            .to_gpu(np.float64)
            .unbound_impl.execute(conf, params, box)
        )

    du_dx_1, du_dp_1, u_1 = test_impl(None)
    du_dx_2, du_dp_2, u_2 = test_impl(np.arange(num_atoms, dtype=np.int32))

    np.testing.assert_array_equal(du_dx_1, du_dx_2)
    np.testing.assert_array_equal(du_dp_1, du_dp_2)
    # If energy overflows, returns a Nan
    if not np.isnan(u_1):
        assert u_1 == u_2
    else:
        assert np.isnan(u_2)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_nonbonded_all_pairs_set_atom_idxs(precision, cutoff, beta, rng: np.random.Generator):
    num_atoms = 100
    box = 3.0 * np.eye(3)
    conf = rng.uniform(0, 1, size=(num_atoms, 3))
    params = rng.uniform(0, 1, size=(num_atoms, 4))

    potential = NonbondedAllPairs(num_atoms, beta, cutoff)
    unbound_pot = potential.to_gpu(precision).unbound_impl

    identity_idxs = np.arange(0, num_atoms, dtype=np.int32)
    for num_idxs in [5, 25, 50, 80, num_atoms]:
        atom_idxs = rng.choice(num_atoms, size=(num_idxs,), replace=False).astype(identity_idxs.dtype)
        ignored_idxs = np.delete(identity_idxs, atom_idxs)
        unbound_pot.set_atom_idxs(atom_idxs)  # type: ignore
        ref_potential = NonbondedAllPairs(num_atoms, beta, cutoff, atom_idxs)
        unbound_ref = ref_potential.to_gpu(precision).unbound_impl

        du_dx, du_dp, u = unbound_pot.execute(conf, params, box.astype(np.float64))
        du_dx_2, du_dp_2, u_2 = unbound_pot.execute(conf, params, box.astype(np.float64))
        ref_du_dx, ref_du_dp, ref_u = unbound_ref.execute(conf, params, box.astype(np.float64))

        np.testing.assert_array_equal(du_dx_2, du_dx)
        np.testing.assert_array_equal(du_dp_2, du_dp)
        np.testing.assert_equal(du_dp_2, du_dp)
        np.testing.assert_equal(u_2, u)

        # Atoms that are ignored by the potential, should always return 0.0
        assert np.all(du_dx[ignored_idxs] == 0.0)

        np.testing.assert_array_equal(ref_du_dx, du_dx)
        np.testing.assert_array_equal(ref_du_dp, du_dp)
        np.testing.assert_equal(ref_u, u)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms_subset", [None, 33])
@pytest.mark.parametrize("num_atoms", [33, 65, 231])
def test_nonbonded_all_pairs_correctness(
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
    params = example_nonbonded_potential.params[:num_atoms, :]

    atom_idxs = (
        rng.choice(num_atoms, size=(num_atoms_subset,), replace=False).astype(np.int32) if num_atoms_subset else None
    )

    potential = NonbondedAllPairs(num_atoms, beta, cutoff, atom_idxs)

    test_impl = potential.to_gpu(precision)

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        GradientTest().compare_forces(conf, params, example_box, potential, test_impl, rtol=rtol, atol=atol)
        GradientTest().assert_differentiable_interface_consistency(conf, params, example_box, test_impl)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("num_atoms_subset", [None, 33])
@pytest.mark.parametrize("num_atoms", [33, 65, 231])
def test_nonbonded_all_pairs_order_independent(
    num_atoms,
    num_atoms_subset,
    precision,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    "Verifies that with and without hilbert sorting the nonbonded potential is bitwise deterministic."

    conf = example_conf[:num_atoms]
    params = example_nonbonded_potential.params[:num_atoms, :]

    atom_idxs = (
        rng.choice(num_atoms, size=(num_atoms_subset,), replace=False).astype(np.int32) if num_atoms_subset else None
    )

    sorted_pot = NonbondedAllPairs(num_atoms, beta, cutoff, atom_idxs)
    unsorted_pot = NonbondedAllPairs(num_atoms, beta, cutoff, atom_idxs, disable_hilbert_sort=True)

    sorted_impl = sorted_pot.to_gpu(precision).unbound_impl
    unsorted_impl = unsorted_pot.to_gpu(precision).unbound_impl

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        a_du_dx, a_du_dp, a_u = sorted_impl.execute(conf, params, example_box)
        b_du_dx, b_du_dp, b_u = unsorted_impl.execute(conf, params, example_box)
        np.testing.assert_array_equal(a_du_dx, b_du_dx)
        np.testing.assert_array_equal(a_du_dp, b_du_dp)
        assert a_u == b_u
