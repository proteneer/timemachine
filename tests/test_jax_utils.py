import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as np, jit
import numpy as onp
onp.random.seed(2021)

from timemachine.potentials.jax_utils import delta_r


def test_jitted_delta_r_symmetric():
    """assert jit(delta_r)(ri, rj, box) == - jit(delta_r)(rj, ri, box)"""
    ri, rj = onp.random.randn(2, 1000, 3)
    box = np.eye(3)

    dr_ij = jit(delta_r)(ri, rj, box)
    dr_ji = jit(delta_r)(rj, ri, box)

    onp.testing.assert_allclose(dr_ij, -dr_ji)
