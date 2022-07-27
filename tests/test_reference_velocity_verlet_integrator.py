import numpy as np
from jax import config, grad, jit
from jax import numpy as jnp

config.update("jax_enable_x64", True)

from functools import partial

from timemachine.integrator import VelocityVerletIntegrator


def assert_reversible(x0, v0, update_fxn):
    x1, v1 = update_fxn(x0, v0)
    x0_, v0_ = update_fxn(x1, -v1)

    assert np.isclose(x0_, x0).all()
    assert np.isclose(-v0_, v0).all()

    # also assert this is not a no-op
    assert not np.isclose(x1, x0).all()


def test_reversibility_on_quartic_potential():
    def U(x):
        return jnp.sum(x ** 4)

    def force(x):
        return -grad(U)(x)

    for n_steps in [1, 10, 100, 1000, 10000]:
        n = np.random.randint(10, 10000)  # Unif[10, 10000]
        masses = np.random.rand(n) + 1  # Unif[1, 2]
        dt = 0.09 * np.random.rand() + 0.01  # Unif[0.01, 0.1]
        x0 = np.random.randn(n, 3)
        v0 = np.random.randn(n, 3)

        intg = VelocityVerletIntegrator(force, masses, dt)

        @jit
        def update(x, v, n_steps=1000):
            return intg._update_via_fori_loop(x, v, n_steps)

        update_fxn = partial(update, n_steps=n_steps)
        assert_reversible(x0, v0, update_fxn)
