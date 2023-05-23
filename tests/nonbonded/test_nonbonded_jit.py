import jax
import jax.numpy as jnp
import numpy as np

from timemachine.potentials import Nonbonded


def test_nonbonded_reference_jittable(rng: np.random.Generator):

    N = 30

    U_ref = Nonbonded(
        N,
        exclusion_idxs=jnp.zeros((0,)),
        scale_factors=jnp.zeros((0, 2)),
        beta=1.0,
        cutoff=0.1,
    )

    U_ref_jit = jax.jit(U_ref.__call__)

    _ = U_ref_jit(
        conf=rng.uniform(0, 1, size=(N, 3)),
        params=rng.uniform(0, 1, size=(N, 3)),
        box=10.0 * np.eye(3),
    )
