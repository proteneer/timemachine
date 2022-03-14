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


def make_ref_potential(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff):
    @functools.wraps(nonbonded.nonbonded_v3)
    def wrapped(conf, params, box, lamb):
        num_atoms, _ = conf.shape
        no_rescale = np.ones((num_atoms, num_atoms))
        return nonbonded.nonbonded_v3(
            conf,
            params,
            box,
            lamb,
            charge_rescale_mask=no_rescale,
            lj_rescale_mask=no_rescale,
            beta=beta,
            cutoff=cutoff,
            lambda_plane_idxs=lambda_plane_idxs,
            lambda_offset_idxs=lambda_offset_idxs,
        )

    return wrapped


@pytest.mark.parametrize("lamb", [0.0, 0.1])
@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms", [33, 65, 231, 1050, 4080])
def test_nonbonded_all_pairs_correctness(
    num_atoms,
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
    params = example_nonbonded_params[:num_atoms, :]

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    ref_potential = make_ref_potential(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff)
    test_potential = NonbondedAllPairs(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff)

    GradientTest().compare_forces(
        conf, params, example_box, lamb, ref_potential, test_potential, precision=precision, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("lamb", [0.0, 0.1, 0.9, 1.0])
@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 2e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms", [33, 231, 4080])
def test_nonbonded_all_pairs_interpolated_correctness(
    num_atoms,
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
    "Compares with jax reference implementation, with parameter interpolation."

    conf = example_conf[:num_atoms]
    params_initial = example_nonbonded_params[:num_atoms, :]
    params = gen_params(params_initial, rng)

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    ref_potential = nonbonded.interpolated(make_ref_potential(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff))
    test_potential = NonbondedAllPairsInterpolated(lambda_plane_idxs, lambda_offset_idxs, beta, cutoff)

    GradientTest().compare_forces(
        conf, params, example_box, lamb, ref_potential, test_potential, precision=precision, rtol=rtol, atol=atol
    )
