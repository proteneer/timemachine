from jax.config import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import pytest
from common import GradientTest
from parameter_interpolation import gen_params

from timemachine.potentials import NonbondedAllPairs, Precision

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_all_pairs_invalid_planes_offsets():
    with pytest.raises(RuntimeError) as e:
        NonbondedAllPairs([0], [0, 0], 2.0, 1.1).impl_cuda(Precision.F32)

    assert "lambda offset idxs and plane idxs need to be equivalent" in str(e)


def test_nonbonded_all_pairs_invalid_atom_idxs():
    with pytest.raises(RuntimeError) as e:
        NonbondedAllPairs([0, 1], [0], 2.0, 1.1, [0, 0]).impl_cuda(Precision.F32)

    assert "atom indices must be unique" in str(e)


def test_nonbonded_all_pairs_invalid_num_atoms():
    impl = NonbondedAllPairs([0], [0], 2.0, 1.1).impl_cuda(Precision.F32)
    with pytest.raises(RuntimeError) as e:
        impl.execute(np.zeros((2, 3)), np.zeros((1, 3)), np.eye(3), 0)

    assert "NonbondedAllPairs::execute_device(): expected N == N_, got N=2, N_=1" in str(e)


def test_nonbonded_all_pairs_invalid_num_params():
    impl = NonbondedAllPairs([0], [0], 2.0, 1.1).impl_cuda(Precision.F32)
    with pytest.raises(RuntimeError) as e:
        impl.execute(np.zeros((1, 3)), np.zeros((2, 3)), np.eye(3), 0)

    assert "NonbondedAllPairs::execute_device(): expected P == M*N_*3, got P=6, M*N_*3=3" in str(e)

    impl = NonbondedAllPairs([0], [0], 2.0, 1.1, interpolated=True).impl_cuda(Precision.F32)
    with pytest.raises(RuntimeError) as e:
        impl.execute(np.zeros((1, 3)), np.zeros((1, 3)), np.eye(3), 0)

    assert "NonbondedAllPairs::execute_device(): expected P == M*N_*3, got P=3, M*N_*3=6" in str(e)


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

    for idx in rng.choice(num_atoms, size=(3,)):
        atom_idxs = np.array([idx], dtype=np.int32)

        impl = NonbondedAllPairs(
            lambda_plane_idxs,
            lambda_offset_idxs,
            beta,
            cutoff,
            atom_idxs,
        ).impl_cuda(Precision.F64)

        du_dx, du_dp, du_dl, u = impl.execute(conf, params, box, lamb)

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
            .impl_cuda(Precision.F64)
            .execute(conf, params, box, lamb)
        )

    du_dx_1, du_dp_1, du_dl_1, u_1 = test_impl(None)
    du_dx_2, du_dp_2, du_dl_2, u_2 = test_impl(np.arange(num_atoms, dtype=np.int32))

    np.testing.assert_array_equal(du_dx_1, du_dx_2)
    np.testing.assert_array_equal(du_dp_1, du_dp_2)
    np.testing.assert_array_equal(du_dl_1, du_dl_2)
    assert u_1 == u_2


@pytest.mark.parametrize("lamb", [0.0, 0.1])
@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(Precision.F64, 1e-8, 1e-8), (Precision.F32, 1e-4, 5e-4)])
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
    lamb,
    example_nonbonded_params,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    "Compares with jax reference implementation."

    conf = example_conf[:num_atoms]
    params_initial = example_nonbonded_params[:num_atoms, :]
    params = gen_params(params_initial, rng) if interpolated else params_initial

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    atom_idxs = (
        rng.choice(num_atoms, size=(num_atoms_subset,), replace=False).astype(np.int32) if num_atoms_subset else None
    )

    potential = NonbondedAllPairs(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff, atom_idxs)

    GradientTest().compare_forces(
        conf,
        params,
        example_box,
        lamb,
        potential=potential,
        precision=precision,
        rtol=rtol,
        atol=atol,
    )
