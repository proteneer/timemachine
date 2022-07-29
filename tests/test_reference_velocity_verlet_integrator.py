from functools import partial

import numpy as np
from jax import config, grad, jit
from jax import numpy as jnp

config.update("jax_enable_x64", True)

from timemachine.integrator import VelocityVerletIntegrator
from timemachine.testsystems.relative import hif2a_ligand_pair


def assert_reversible(x0, v0, update_fxn, atol=1e-10):
    """Define a fxn self_inverse as composition of flip_velocities and update_fxn,
    then assert that
    * self_inverse is its own inverse
    * self_inverse is not trivial (aka not the identity function)
    """

    def self_inverse(x, v):
        """integrate forward in time, flip v
        (expected to be an "involution" i.e. its own inverse)"""
        x_next, v_next = update_fxn(x, v)
        return x_next, -v_next

    close = partial(np.allclose, atol=atol)

    # assert "self_inverse" is really its own inverse
    x1, v1 = self_inverse(x0, v0)
    x0_, v0_ = self_inverse(x1, v1)

    assert close(x0_, x0), f"max(abs(x0 - x0_)) = {np.max(np.abs(x0 - x0_))}"
    assert close(v0_, v0), f"max(abs(v0 - v0_)) = {np.max(np.abs(v0 - v0_))}"

    # also assert this is not a no-op
    assert (not close(x1, x0)) and (not close(v1, v0))


def assert_reversibility_using_step_implementations(intg, x0, v0, n_steps=1000, atol=1e-10):
    """Assert reversibility of .step and .multiple_steps implementations"""

    # check step implementation
    def step_update(x, v):
        for t in range(n_steps):
            x, v = intg.step(x, v)
        return x, v

    assert_reversible(x0, v0, step_update, atol=atol)

    # check multiple_steps implementation
    def multiple_steps_update(x, v):
        xs, vs = intg.multiple_steps(x, v, n_steps)
        return xs[-1], vs[-1]

    assert_reversible(x0, v0, multiple_steps_update, atol=atol)


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

        assert_reversible(x0, v0, jax_update, atol=1e-10)


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

    for n_steps in [1, 10, 100, 500, 1000]:  # , 10000]:
        v0 = np.random.randn(*coords.shape)

        # check "public" .step and .multiple_steps implementations
        assert_reversibility_using_step_implementations(intg, x0, v0, n_steps, atol=1e-10)

    # TODO: possibly investigate why n_steps = 10000 fails
    # assert_reversibility_using_step_implementations(intg, x0, v0, n_steps=10000, atol=0.1)
    # * also fails with reduced dt = 1.0e-3
    # * passes with greatly reduced dt = 1.0e-4
    # TODO: does replacing np.sum(du_dxs, 0) with SummedPotential address this?


def test_trajectory_wise_self_consistency():
    """Assert that trajectories produced by .step and .multiple_steps agree at all timesteps,
    and that their final frames agree with ._update_via_fori_loop"""

    np.random.seed(2022)

    def u_fxn(x):
        return jnp.sum(x ** 4)

    force_fxn = jit(lambda x: -grad(u_fxn)(x))
    n_atoms = 100
    masses = np.ones(n_atoms)
    dt = 0.1
    intg = VelocityVerletIntegrator(force_fxn, masses, dt)

    x0, v0 = np.random.randn(2, n_atoms, 3)

    close = partial(np.allclose, atol=1e-10)  # would also pass with atol=0.0

    for num_steps in [1, 10, 100, 1000]:

        # reference: intg.step in for loop
        ref_traj = [(x0, v0)]
        for _ in range(num_steps):
            ref_traj.append(intg.step(*ref_traj[-1]))

        xs_ref = np.array([x for (x, v) in ref_traj])
        vs_ref = np.array([v for (x, v) in ref_traj])

        # test: multiple_steps
        xs, vs = intg.multiple_steps(x0, v0, num_steps)

        # assert trajectories are ~identical
        assert close(xs_ref, xs) and close(vs_ref, vs)

        # also that output of jax.lax loop matches last frame of ref traj
        x_lax, v_lax = intg._update_via_fori_loop(x0, v0, num_steps)
        assert close(xs_ref[-1], x_lax) and close(vs_ref[-1], v_lax)
