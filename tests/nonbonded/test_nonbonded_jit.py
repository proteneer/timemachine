import jax
import jax.numpy as jnp
import numpy as np
import pytest

from timemachine.potentials import Nonbonded

pytestmark = [pytest.mark.nocuda]


@pytest.mark.parametrize("num_atom_idxs", [None, 15])
def test_nonbonded_reference_jittable(num_atom_idxs, rng: np.random.Generator):
    N = 30

    U_ref = Nonbonded(
        N,
        exclusion_idxs=np.zeros((0,), dtype=np.int32),
        scale_factors=np.zeros((0, 2)),
        beta=1.0,
        cutoff=0.1,
        atom_idxs=np.arange(num_atom_idxs) if num_atom_idxs is not None else None,
    )

    U_ref_jit = jax.jit(U_ref.__call__)

    _ = U_ref_jit(
        conf=jnp.array(rng.uniform(0, 1, size=(N, 3))),
        params=jnp.array(rng.uniform(0, 1, size=(N, 3))),
        box=10.0 * np.eye(3),
    )
    _ = jax.value_and_grad(U_ref_jit, argnums=(0, 1))(
        jnp.array(rng.uniform(0, 1, size=(N, 3))),
        jnp.array(rng.uniform(0, 1, size=(N, 3))),
        10.0 * np.eye(3),
    )
