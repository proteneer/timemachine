from functools import partial

import numpy as np

from timemachine.integrator import VelocityVerletIntegrator as ReferenceVelocityVerlet
from timemachine.lib import VelocityVerletIntegrator, custom_ops
from timemachine.lib.potentials import SummedPotential
from timemachine.testsystems.relative import hif2a_ligand_pair


def setup_velocity_verlet(bps, x0, box, dt, masses):
    integrator = VelocityVerletIntegrator(dt, masses)
    # return integrator impl to avoid deallocating
    intg = integrator.impl()
    context = custom_ops.Context(x0, np.zeros_like(x0), box, intg, bps)
    return intg, context


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


def assert_reversibility_using_step_implementations(context, n_steps=1000, atol=1e-10):
    """Assert reversibility of .step and .multiple_steps implementations"""

    x0 = context.get_x_t()
    v0 = context.get_v_t()

    # check step implementation
    def step_update(x, v, lamb=0.0):
        context.set_x_t(x)
        context.set_v_t(v)

        for t in range(n_steps):
            context.step(lamb)
        # Must call finalize in case of using step
        context.finalize(lamb)
        x = context.get_x_t()
        v = context.get_v_t()
        return x, v

    assert_reversible(x0, v0, step_update, atol=atol)

    # check multiple_steps implementation
    def multiple_steps_update(x, v, lamb=0.0):
        context.set_x_t(x)
        context.set_v_t(v)
        _, xs, _ = context.multiple_steps(np.ones(n_steps) * lamb)
        v = context.get_v_t()
        return xs[-1], v

    assert_reversible(x0, v0, multiple_steps_update, atol=atol)


def test_reversibility():
    """Check reversibility of "public" .step and .multiple_steps implementations when constructing a Context with"""

    np.random.seed(2022)

    # define a Python force fxn that calls custom_ops
    rfe = hif2a_ligand_pair
    unbound_potentials, sys_params, masses = rfe.prepare_vacuum_edge(rfe.ff.get_ordered_params())
    coords = rfe.prepare_combined_coords()
    bound_potentials = [
        ubp.bind(params).bound_impl(np.float32) for (ubp, params) in zip(unbound_potentials, sys_params)
    ]
    box = 100 * np.eye(3)

    dt = 1.5e-3

    # Is not infinitely reversible, will fail after 5000 steps due to accumulation of coords/velos in float
    for n_steps in [1, 10, 100, 500, 1000, 3000]:
        v0 = np.random.randn(*coords.shape)
        intg, ctxt = setup_velocity_verlet(bound_potentials, coords, box, dt, masses)  # noqa
        ctxt.set_v_t(v0)

        # check "public" .step and .multiple_steps implementations
        assert_reversibility_using_step_implementations(ctxt, n_steps, atol=1e-10)


def test_matches_reference():
    np.random.seed(2022)

    rfe = hif2a_ligand_pair
    unbound_potentials, sys_params, masses = rfe.prepare_vacuum_edge(rfe.ff.get_ordered_params())
    coords = rfe.prepare_combined_coords()
    box = 100 * np.eye(3)

    dt = 1.5e-3

    lamb = 0.0

    summed_potential = SummedPotential(unbound_potentials, sys_params)
    summed_potential.bind(np.concatenate([param.reshape(-1) for param in sys_params]))
    bound_summed = summed_potential.bound_impl(np.float32)

    def force(coords):
        du_dxs = bound_summed.execute(coords, box, lamb)[0]
        return -du_dxs

    intg = ReferenceVelocityVerlet(force, masses, dt)

    v0 = np.random.randn(*coords.shape)

    intg_impl, ctxt = setup_velocity_verlet([bound_summed], coords, box, dt, masses)  # noqa

    ctxt.set_v_t(v0)

    n_steps = 10

    ref_xs, ref_vs = intg.multiple_steps(coords, v0, n_steps=n_steps + 1)
    np.testing.assert_array_almost_equal(ref_xs[0], coords, decimal=10)
    np.testing.assert_array_almost_equal(ref_xs[0], ctxt.get_x_t(), decimal=10)

    _, xs, _ = ctxt.multiple_steps(np.ones(n_steps) * lamb, 0, 1)
    assert xs.shape[0] == n_steps
    v1 = ctxt.get_v_t()
    atol = 1e-5
    np.testing.assert_allclose(ref_xs[-1], xs[-1], atol=atol)
    np.testing.assert_allclose(ref_vs[-1], v1, atol=atol)
