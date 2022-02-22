import jax

jax.config.update("jax_enable_x64", True)

import functools

import numpy as np
import pytest
from common import GradientTest

from timemachine.lib.potentials import NonbondedPairList
from timemachine.potentials import jax_utils, nonbonded


def test_nonbonded_pair_list_invalid_pair_idxs():
    with pytest.raises(RuntimeError) as e:
        NonbondedPairList([0], [0], [0], [0], 2.0, 1.1).unbound_impl(np.float32)

    assert "pair_idxs.size() must be even, but got 1" in str(e)

    with pytest.raises(RuntimeError) as e:
        NonbondedPairList([(0, 0)], [(1, 1)], [0], [0], 2.0, 1.1).unbound_impl(np.float32)

    assert "illegal pair with src == dst: 0, 0" in str(e)

    with pytest.raises(RuntimeError) as e:
        NonbondedPairList([(0, 1)], [(1, 1), (2, 2)], [0], [0], 2.0, 1.1).unbound_impl(np.float32)

    assert "expected same number of pairs and scale tuples, but got 1 != 2" in str(e)


def make_ref_potential(pair_idxs, scales, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff):
    @functools.wraps(nonbonded.nonbonded_v3_on_specific_pairs)
    def wrapped(conf, params, box, lamb):

        # compute 4d coordinates
        w = jax_utils.compute_lifting_parameter(lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)
        conf_4d = jax_utils.augment_dim(conf, w)
        box_4d = (1000 * jax.numpy.eye(4)).at[:3, :3].set(box)

        vdW, electrostatics = nonbonded.nonbonded_v3_on_specific_pairs(
            conf_4d, params, box_4d, pair_idxs[:, 0], pair_idxs[:, 1], beta, cutoff
        )
        return jax.numpy.sum(scales[:, 1] * vdW + scales[:, 0] * electrostatics)

    return wrapped


@pytest.mark.parametrize("lamb", [0.0, 0.1])
@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("ixn_group_size", [2, 33, 231])
def test_nonbonded_pair_list_correctness(
    ixn_group_size,
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

    num_atoms, _ = example_conf.shape

    # randomly select 2 interaction groups and construct all pairwise interactions
    atom_idxs = rng.choice(
        num_atoms,
        size=(
            2,
            ixn_group_size,
        ),
        replace=False,
    ).astype(np.int32)

    pair_idxs = np.stack(np.meshgrid(atom_idxs[0, :], atom_idxs[1, :])).reshape(2, -1).T
    num_pairs, _ = pair_idxs.shape

    scales = rng.uniform(0, 1, size=(num_pairs, 2))

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    ref_potential = make_ref_potential(pair_idxs, scales, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff)
    test_potential = NonbondedPairList(pair_idxs, scales, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff)

    GradientTest().compare_forces(
        example_conf,
        example_nonbonded_params,
        example_box,
        lamb,
        ref_potential,
        test_potential,
        precision=precision,
        rtol=rtol,
        atol=atol,
    )
