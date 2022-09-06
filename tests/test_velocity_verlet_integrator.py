from functools import partial

import numpy as np
import pytest

from timemachine.integrator import VelocityVerletIntegrator as ReferenceVelocityVerlet
from timemachine.lib import VelocityVerletIntegrator, custom_ops
from timemachine.lib.potentials import SummedPotential
from timemachine.md import builders
from timemachine.testsystems.relative import hif2a_ligand_pair


def setup_velocity_verlet(bps, x0, box, dt, masses):
    integrator = VelocityVerletIntegrator(dt, masses)
    # return integrator impl to avoid deallocating
    intg = integrator.impl()
    context = custom_ops.Context(x0, np.zeros_like(x0), box, intg, bps)
    return intg, context


def assert_reversible(x0, v0, update_fxn, lambdas, atol=1e-10):
    """Define a fxn self_inverse as composition of flip_velocities and update_fxn,
    then assert that
    * self_inverse is its own inverse
    * self_inverse is not trivial (aka not the identity function)
    """

    def self_inverse(x, v, lamb_sched):
        """integrate forward in time, flip v
        (expected to be an "involution" i.e. its own inverse)"""
        x_next, v_next = update_fxn(x, v, lamb_sched)
        return x_next, -v_next, lamb_sched[::-1]

    # assert "self_inverse" is really its own inverse
    x1, v1, rev_lamb = self_inverse(x0, v0, lambdas)
    x0_, v0_, _ = self_inverse(x1, v1, rev_lamb)

    np.testing.assert_allclose(x0_, x0, atol=atol)
    np.testing.assert_allclose(v0_, v0, atol=atol)

    close = partial(np.allclose, atol=atol)
    # also assert this is not a no-op
    assert (not close(x1, x0)) and (not close(v1, v0))


def assert_reversibility_using_step_implementations(context, schedule, atol=1e-10):
    """Assert reversibility of .step and .multiple_steps implementations"""

    x0 = context.get_x_t()
    v0 = context.get_v_t()

    # check step implementation
    def step_update(x, v, lamb_sched):
        context.set_x_t(x)
        context.set_v_t(v)
        context.initialize(lamb_sched[0])
        for lamb in lamb_sched:
            context.step(lamb)
        # Must call finalize in case of using step
        context.finalize(lamb_sched[-1])
        x = context.get_x_t()
        v = context.get_v_t()
        return x, v

    assert_reversible(x0, v0, step_update, schedule, atol=atol)

    # check multiple_steps implementation
    def multiple_steps_update(x, v, lamb_sched):
        context.set_x_t(x)
        context.set_v_t(v)
        _, xs, _ = context.multiple_steps(lamb_sched)
        v = context.get_v_t()
        return xs[-1], v

    assert_reversible(x0, v0, multiple_steps_update, schedule, atol=atol)

    def multiple_steps_U_update(x, v, lamb_sched):
        # Doesn't use the lamb sched, as multiple_steps_U is always run as a equilibrium simulation
        context.set_x_t(x)
        context.set_v_t(v)
        _, xs, _ = context.multiple_steps_U(0.0, len(lamb_sched), [], 0, 0)
        v = context.get_v_t()
        return xs[-1], v

    assert_reversible(x0, v0, multiple_steps_U_update, schedule, atol=atol)


def test_reversibility():
    """Check reversibility of "public" .step and .multiple_steps implementations for a Context using the VelocityVerlet integrator"""

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

    # Is not infinitely reversible, will fail after 3000 steps due to accumulation of coords/velos in floating point
    for n_steps in [1, 10, 100, 500, 1000, 2000]:
        lamb_sched = np.linspace(0, 1, n_steps)
        v0 = np.random.randn(*coords.shape)
        intg, ctxt = setup_velocity_verlet(bound_potentials, coords, box, dt, masses)  # noqa
        ctxt.set_v_t(v0)

        # check "public" .step and .multiple_steps implementations
        assert_reversibility_using_step_implementations(ctxt, lamb_sched, atol=1e-10)


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

    # Add one step as the C++ context does N steps + 2 half steps (initialize, finalize)
    ref_xs, ref_vs = intg.multiple_steps(coords, v0, n_steps=n_steps + 1)
    np.testing.assert_allclose(ref_xs[0], coords, atol=1e-10)
    np.testing.assert_allclose(ref_xs[0], ctxt.get_x_t(), atol=1e-10)

    _, xs, _ = ctxt.multiple_steps(np.ones(n_steps) * lamb, 0, 1)
    assert xs.shape[0] == n_steps
    v1 = ctxt.get_v_t()
    atol = 1e-5
    np.testing.assert_allclose(ref_xs[1:-1], xs, atol=atol)
    np.testing.assert_allclose(ref_vs[-1], v1, atol=atol)


def test_initialization_and_finalization():
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

    intg, ctxt = setup_velocity_verlet(bound_potentials, coords, box, dt, masses)  # noqa
    with pytest.raises(RuntimeError) as e:
        ctxt.finalize(0.0)
    assert "not initialized" in str(e.value)

    ctxt.initialize(0.0)

    with pytest.raises(RuntimeError) as e:
        ctxt.initialize(0.0)
    assert "initialized twice" in str(e.value)

    ctxt.finalize(0.0)

    with pytest.raises(RuntimeError) as e:
        ctxt.finalize(0.0)
    assert "not initialized" in str(e.value)


def test_verlet_with_multiple_steps_local():
    """Ensure Local MD can be run with the Velocity Verlet integrator and can handle a temperature passed in"""
    dt = 1.5e-3
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    rfe = hif2a_ligand_pair
    unbound_potentials, sys_params, masses = rfe.prepare_host_edge(rfe.ff.get_ordered_params(), solvent_system)
    coords = rfe.prepare_combined_coords(solvent_coords)
    bound_potentials = [
        ubp.bind(params).bound_impl(np.float32) for (ubp, params) in zip(unbound_potentials, sys_params)
    ]

    mol_a = rfe.mol_a

    local_idxs = np.arange(len(coords) - mol_a.GetNumAtoms(), len(coords), dtype=np.uint32)

    intg_impl, ctxt = setup_velocity_verlet(bound_potentials, coords, solvent_box, dt, masses)  # noqa

    n_steps = 10

    ctxt.multiple_steps_local(np.ones(n_steps), local_idxs)

    # Uses non-default temperature as the integrator is not a thermostat, only changes the selection probabilities for local MD
    ctxt.multiple_steps_local(np.ones(n_steps), local_idxs, temperature=100)
