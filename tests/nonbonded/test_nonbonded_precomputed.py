import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from common import GradientTest

from timemachine.lib.potentials import NonbondedPairListPrecomputed
from timemachine.potentials import generic

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_precomputed_pair_list_invalid_pair_idxs():

    with pytest.raises(RuntimeError) as e:
        NonbondedPairListPrecomputed([0], [0], 2.0, 1.1).unbound_impl(np.float32)

    assert "idxs.size() must be exactly 2*B" in str(e)

    with pytest.raises(RuntimeError) as e:
        NonbondedPairListPrecomputed([(0, 0)], [0.3], 2.0, 1.1).unbound_impl(np.float32)

    assert "illegal pair with src == dst: 0, 0" in str(e)

    with pytest.raises(RuntimeError) as e:
        NonbondedPairListPrecomputed([(0, 1)], [0.3, 0.4], 2.0, 1.1).unbound_impl(np.float32)

    assert "offset size does not match idxs size" in str(e)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1, 10000.0])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("ixn_group_size", [4, 33, 231])
@pytest.mark.parametrize("num_atoms", [25358])
def test_nonbonded_pair_list_precomputed_correctness(
    ixn_group_size,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    num_atoms,
    rng: np.random.Generator,
):
    "Compares with jax reference implementation."

    pair_idxs = []
    for _ in range(ixn_group_size):
        pair_idxs.append(rng.choice(np.arange(num_atoms), 2, replace=False))
    pair_idxs = np.array(pair_idxs, dtype=np.int32)
    num_pairs, _ = pair_idxs.shape

    params = rng.uniform(0, 1, size=(num_pairs, 3))
    params[:, 0] -= 0.5  # get some positive and negative charges
    params[:, 1] /= 5  # shrink lj sigma to avoid huge repulsive forces

    w_offsets = rng.uniform(0, 1, size=num_pairs) / 4
    w_offsets = w_offsets.astype(np.float64)

    conf = rng.uniform(0, 1, size=(num_atoms, 3)) * 3

    box = np.diag(
        1 + rng.uniform(0, 1, size=3) * 3
    )  # box should be fully ignored tbh (just like all other bonded forces)

    potential = generic.NonbondedPairListPrecomputed(pair_idxs, w_offsets, beta, cutoff)

    GradientTest().compare_forces_gpu_vs_reference(
        conf,
        params,
        box,
        [0.0],
        potential,
        precision=precision,
        rtol=rtol,
        atol=atol,
    )
