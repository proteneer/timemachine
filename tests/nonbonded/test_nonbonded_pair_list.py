import numpy as np
import pytest
from common import GradientTest
from nonbonded import gen_params_with_4d_offsets

from timemachine.lib.potentials import NonbondedPairList
from timemachine.potentials import generic

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_pair_list_invalid_pair_idxs():
    with pytest.raises(RuntimeError) as e:
        NonbondedPairList([0], [0], 2.0, 1.1).unbound_impl(np.float32)

    assert "pair_idxs.size() must be even, but got 1" in str(e)

    with pytest.raises(RuntimeError) as e:
        NonbondedPairList([(0, 0)], [(1, 1)], 2.0, 1.1).unbound_impl(np.float32)

    assert "illegal pair with src == dst: 0, 0" in str(e)

    with pytest.raises(RuntimeError) as e:
        NonbondedPairList([(0, 1)], [(1, 1), (2, 2)], 2.0, 1.1).unbound_impl(np.float32)

    assert "expected same number of pairs and scale tuples, but got 1 != 2" in str(e)


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
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    "Compares with jax reference implementation."

    num_atoms, _ = example_conf.shape

    # randomly select 2 interaction groups and construct all pairwise interactions
    atom_idxs = rng.choice(num_atoms, size=(2, ixn_group_size), replace=False).astype(np.int32)

    pair_idxs = np.stack(np.meshgrid(atom_idxs[0, :], atom_idxs[1, :])).reshape(2, -1).T
    num_pairs, _ = pair_idxs.shape

    rescale_mask = rng.uniform(0, 1, size=(num_pairs, 2))

    potential = generic.NonbondedPairList(pair_idxs, rescale_mask, beta, cutoff)
    GradientTest().compare_forces_gpu_vs_reference(
        example_conf,
        gen_params_with_4d_offsets(rng, example_nonbonded_potential.params, -2 * cutoff, 2 * cutoff, 3),
        example_box,
        potential,
        precision=precision,
        rtol=rtol,
        atol=atol,
    )
