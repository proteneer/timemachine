import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as np, jit
import numpy as onp
onp.random.seed(2021)

from timemachine.potentials.jax_utils import delta_r


def test_delta_r():
    """assert that
    * delta_r(ri, rj, box) == - delta_r(rj, ri, box)
    * and that delta_r agrees with jit(delta_r)
    on a few random inputs of varying size
    """

    for _ in range(5):
        n_atoms = onp.random.randint(50, 1000)
        dim = onp.random.randint(3, 5)
        ri, rj = onp.random.randn(2,  n_atoms, dim)
        box = np.eye(dim)

        dr_ij_1 = delta_r(ri, rj, box)
        dr_ji_1 = delta_r(rj, ri, box)

        onp.testing.assert_allclose(dr_ij_1, -dr_ji_1)

        dr_ij_2 = jit(delta_r)(ri, rj, box)
        dr_ji_2 = jit(delta_r)(rj, ri, box)

        onp.testing.assert_allclose(dr_ij_1, dr_ij_2)
        onp.testing.assert_allclose(dr_ji_1, dr_ji_2)
