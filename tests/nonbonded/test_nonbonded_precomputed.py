import numpy as np
import pytest
from common import GradientTest, gen_nonbonded_params_with_4d_offsets

from timemachine.potentials import NonbondedPairListPrecomputed

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_precomputed_pair_list_invalid_pair_idxs():

    with pytest.raises(RuntimeError) as e:
        NonbondedPairListPrecomputed([0], 2.0, 1.1).to_gpu(np.float32).unbound_impl

    assert "idxs.size() must be exactly 2*B" in str(e)

    with pytest.raises(RuntimeError) as e:
        NonbondedPairListPrecomputed([(0, 0)], 2.0, 1.1).to_gpu(np.float32).unbound_impl

    assert "illegal pair with src == dst: 0, 0" in str(e)


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

    params = rng.uniform(0, 1, size=(num_pairs, 4))
    params[:, 0] -= 0.5  # get some positive and negative charges
    params[:, 1] /= 5  # shrink lj sigma to avoid huge repulsive forces

    conf = rng.uniform(0, 1, size=(num_atoms, 3)) * 3

    box = np.diag(
        1 + rng.uniform(0, 1, size=3) * 3
    )  # box should be fully ignored tbh (just like all other bonded forces)

    potential = NonbondedPairListPrecomputed(pair_idxs, beta, cutoff)

    # delta_w positive by convention
    test_impl = potential.to_gpu(precision)
    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff, w_min=0.0):
        GradientTest().compare_forces(
            conf,
            params,
            box,
            potential,
            test_impl,
            rtol=rtol,
            atol=atol,
        )
