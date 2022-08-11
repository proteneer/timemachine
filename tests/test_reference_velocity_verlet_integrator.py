import numpy as np
from jax import config, grad, jit
from jax import numpy as jnp

config.update("jax_enable_x64", True)

from timemachine.integrator import FIXED_TO_FLOAT, FLOAT_TO_FIXED, VelocityVerletIntegrator
from timemachine.testsystems.relative import hif2a_ligand_pair


def assert_bitwise_reversiblility(x0, v0, update_fxn):
    """Define a fxn self_inverse as composition of flip_velocities and update_fxn,
    then assert that
    * self_inverse is its own inverse with bitwise determinism
    * self_inverse is not trivial (aka not the identity function)
    """

    def self_inverse(x, v):
        """integrate forward in time, flip v
        (expected to be an "involution" i.e. its own inverse)"""
        x_next, v_next = update_fxn(x, v)
        return x_next, -v_next

    # Bitwise determinisim is only guarenteed for x0s, v0s where FIXED_TO_FLOAT(FLOAT_TO_FIXED(x)) == x
    # This condition is not met for all floating point values and thus we roundtrip values initially.
    x0 = FIXED_TO_FLOAT(FLOAT_TO_FIXED(x0))
    v0 = FIXED_TO_FLOAT(FLOAT_TO_FIXED(v0))

    # assert "self_inverse" is really its own inverse
    x1, v1 = self_inverse(x0, v0)
    x0_, v0_ = self_inverse(x1, v1)
    np.testing.assert_array_equal(x0_, x0)
    np.testing.assert_array_equal(v0_, v0)

    # also assert this is not a no-op
    assert (not np.allclose(x1, x0)) and (not np.allclose(v1, v0))


def assert_reversibility_using_step_implementations(intg, x0, v0, n_steps=1000):
    """Assert reversibility of .step and .multiple_steps implementations"""

    # check step implementation
    def step_update(x, v):
        for t in range(n_steps):
            x, v = intg.step(x, v)
        return x, v

    assert_bitwise_reversiblility(x0, v0, step_update)

    # check multiple_steps implementation
    def multiple_steps_update(x, v):
        xs, vs = intg.multiple_steps(x, v, n_steps)
        return xs[-1], vs[-1]

    assert_bitwise_reversiblility(x0, v0, multiple_steps_update)


def test_reversibility_with_jax_potentials():
    """On a simple jax-transformable potential (quartic oscillators)
    with randomized parameters and initial conditions
    (n oscillators, masses, dt, x0, v0)
    assert all 3 update functions
    (public: .step and .multiple_steps, private: ._update_via_fori_loop)
    are reversible"""

    np.random.seed(2022)

    def U(x):
        return jnp.sum(x ** 4)

    @jit
    def force(x):
        return -grad(U)(x)

    for n_steps in [1, 10, 100, 1000, 10000]:
        n = np.random.randint(10, 10000)  # Unif[10, 10000]
        masses = np.random.rand(n) + 1  # Unif[1, 2]
        dt = 0.09 * np.random.rand() + 0.01  # Unif[0.01, 0.1]
        x0 = np.random.randn(n, 3)
        v0 = np.random.randn(n, 3)

        intg = VelocityVerletIntegrator(force, masses, dt)

        # check "public" .step and .multiple_steps implementations
        assert_reversibility_using_step_implementations(intg, x0, v0, n_steps)

        # also check "private" jax.lax.fori_loop implementation
        @jit
        def jax_update(x, v):
            return intg._update_via_fori_loop(x, v, n_steps)

        assert_bitwise_reversiblility(x0, v0, jax_update)


def test_reversibility_with_custom_ops_potentials():
    """Check reversibility of "public" .step and .multiple_steps implementations when `force_fxn`
    is a custom_op potential"""

    np.random.seed(2022)

    # define a Python force fxn that calls custom_ops
    rfe = hif2a_ligand_pair
    unbound_potentials, sys_params, masses = rfe.prepare_vacuum_edge(rfe.ff.get_ordered_params())
    coords = rfe.prepare_combined_coords()
    bound_potentials = [
        ubp.bind(params).bound_impl(np.float32) for (ubp, params) in zip(unbound_potentials, sys_params)
    ]
    box = 100 * np.eye(3)

    def force(coords):
        du_dxs = np.array([bp.execute(coords, box, 0.5)[0] for bp in bound_potentials])
        return -np.sum(du_dxs, 0)

    dt = 1.5e-3
    intg = VelocityVerletIntegrator(force, masses, dt)

    x0 = np.array(coords)

    for n_steps in [1, 10, 100, 500, 1000, 10000]:
        v0 = np.random.randn(*coords.shape)

        # check "public" .step and .multiple_steps implementations
        assert_reversibility_using_step_implementations(intg, x0, v0, n_steps)
