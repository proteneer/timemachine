import jax
import jax.numpy as jnp
import numpy as np

from timemachine.potentials import generic


def test_generic_nonbonded_jittable(rng: np.random.Generator):

    N = 30

    potential = generic.Nonbonded(
        exclusion_idxs=jnp.zeros((0,)),
        scale_factors=jnp.zeros((0, 1)),
        lambda_plane_idxs=jnp.zeros(N),
        lambda_offset_idxs=jnp.zeros(N),
        beta=1.0,
        cutoff=0.1,
    )

    U_ref = potential.to_reference()
    U_ref_jit = jax.jit(U_ref)

    _ = U_ref_jit(
        conf=rng.uniform(0, 1, size=(N, 3)),
        params=rng.uniform(0, 1, size=(N, 3)),
        box=10.0 * np.eye(3),
        lam=0.1,
    )
