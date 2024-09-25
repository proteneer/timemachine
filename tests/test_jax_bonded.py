import jax
import numpy as np
import pytest
import scipy

from timemachine.potentials.bonded import harmonic_positional_restraint
from timemachine.potentials.jax_utils import delta_r


@pytest.mark.nocuda
@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("samples", [2, 5, 10, 600])
@pytest.mark.parametrize("k", [1000.0, 2000.0])
def test_harmonic_restraint_potential(seed, samples, k):
    rng = np.random.default_rng(seed)
    x_0 = rng.uniform(0.5, 1.5, (samples, 3))
    x_1 = rng.uniform(0.5, 1.5, (samples, 3))
    box = rng.uniform(1.0, 3.0, (3, 3))

    assert harmonic_positional_restraint(x_0, x_0, box, k=k) == 0.0
    assert harmonic_positional_restraint(x_1, x_1, box, k=k) == 0.0

    assert harmonic_positional_restraint(x_0, x_1, box, k=k) > 0.0

    val_and_grad = jax.value_and_grad(harmonic_positional_restraint, argnums=1)

    def minimize_func(x_updated_flattened):
        x_updated = x_updated_flattened.reshape((samples, 3))
        u, grad = val_and_grad(x_0, x_updated, box, k=k)
        return u, grad.reshape(-1)

    # Verify that the coordinates are not close to begin with
    assert not np.allclose(delta_r(x_0, x_1, box), 1e-14, atol=1e-9)

    # Minimize the second coordinates to the first
    res = scipy.optimize.minimize(
        minimize_func,
        x_1.reshape(-1),
        method="BFGS",
        jac=True,
    )

    x_final = res.x.reshape((samples, 3))
    # Check that under PBCs, values are nearly identical
    np.testing.assert_allclose(delta_r(x_0, x_final, box), 1e-14, atol=1e-9)
