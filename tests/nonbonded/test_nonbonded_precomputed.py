import jax

jax.config.update("jax_enable_x64", True)

import functools

import jax.numpy as jnp
import numpy as np
import pytest
from common import GradientTest

from timemachine.lib.potentials import NonbondedPairListPrecomputed
from timemachine.potentials import nonbonded

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
def test_nonbonded_pair_list_precomputed_correctness(
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

    pair_idxs = []
    for _ in range(ixn_group_size):
        pair_idxs.append(np.random.choice(np.arange(num_atoms), 2))
    pair_idxs = np.array(pair_idxs, dtype=np.int32)
    num_pairs, _ = pair_idxs.shape

    params = np.random.rand(num_pairs, 3)
    params[:, 1] /= 5  # shrink lj ixns to avoid huge repulsive forces
    # params[:, 2] = 0

    w_offsets = np.random.rand(num_pairs) / 3
    w_offsets = w_offsets.astype(np.float64)
    conf = np.random.rand(num_atoms, 3) * 5

    box = np.diag(1 + np.random.rand(3) * 3)  # box should be fully ignored tbh (just like all other bonded forces)

    ref_nb = functools.partial(
        nonbonded.nonbonded_v3_on_precomputed_pairs,
        pairs=pair_idxs,
        offsets=w_offsets,
        beta=beta,
        cutoff=cutoff,
    )

    def ref_potential(conf, params, box, lamb):
        a, b = ref_nb(conf, params, box)
        return jnp.sum(a) + jnp.sum(b)

    test_potential = NonbondedPairListPrecomputed(pair_idxs, w_offsets, beta, cutoff)

    GradientTest().compare_forces(
        conf,
        params,
        box,
        [0.0],
        ref_potential,
        test_potential,
        precision=precision,
        rtol=rtol,
        atol=atol,
    )
