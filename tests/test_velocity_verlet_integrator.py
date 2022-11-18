from functools import partial

import numpy as np
import pytest

from timemachine.ff import Forcefield
from timemachine.integrator import VelocityVerletIntegrator as ReferenceVelocityVerlet
from timemachine.lib import VelocityVerletIntegrator, custom_ops
from timemachine.lib.potentials import SummedPotential
from timemachine.md.enhanced import get_solvent_phase_system
from timemachine.testsystems.ligands import get_biphenyl
from timemachine.testsystems.relative import get_relative_hif2a_in_vacuum


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

    # assert "self_inverse" is really its own inverse
    x1, v1 = self_inverse(x0, v0)
    x0_, v0_ = self_inverse(x1, v1)

    np.testing.assert_allclose(x0_, x0, atol=atol)
    np.testing.assert_allclose(v0_, v0, atol=atol)

    close = partial(np.allclose, atol=atol)
    # also assert this is not a no-op
    assert (not close(x1, x0)) and (not close(v1, v0))


def assert_reversibility_using_step_implementations(context, n_steps, atol=1e-10):
    """Assert reversibility of .step and .multiple_steps implementations"""

    x0 = context.get_x_t()
    v0 = context.get_v_t()

    # check step implementation
    def step_update(x, v):
        context.set_x_t(x)
        context.set_v_t(v)
        context.initialize()
        for _ in range(n_steps):
            context.step()
        # Must call finalize in case of using step
        context.finalize()
        x = context.get_x_t()
        v = context.get_v_t()
        return x, v

    assert_reversible(x0, v0, step_update, atol=atol)

    # check multiple_steps implementation
    def multiple_steps_update(x, v):
        context.set_x_t(x)
        context.set_v_t(v)
        xs, _ = context.multiple_steps(n_steps)
        v = context.get_v_t()
        return xs[-1], v

    assert_reversible(x0, v0, multiple_steps_update, atol=atol)

    def multiple_steps_U_update(x, v):
        # Doesn't use the lamb sched, as multiple_steps_U is always run as a equilibrium simulation
        context.set_x_t(x)
        context.set_v_t(v)
        _, xs, _ = context.multiple_steps_U(n_steps, 0, 0)
        v = context.get_v_t()
        return xs[-1], v

    assert_reversible(x0, v0, multiple_steps_U_update, atol=atol)


def test_reversibility():
    """Check reversibility of "public" .step and .multiple_steps implementations for a Context using the VelocityVerlet integrator"""

    seed = 2022

    # define a Python force fxn that calls custom_ops
    unbound_potentials, sys_params, coords, masses = get_relative_hif2a_in_vacuum()
    bound_potentials = [pot.bound_impl(precision=np.float32) for pot in unbound_potentials]

    box = 100 * np.eye(3)

    dt = 1.5e-3

    # Is not infinitely reversible, will fail after 3000 steps due to accumulation of coords/vels in floating point
    for n_steps in [1, 10, 100, 500, 1000, 2000]:
        # Note: reversibility can fail depending on the
        # range of values in the velocities. Setting the seed
        # here keeps the range the same for all n_step values.
        np.random.seed(seed)
        v0 = np.random.randn(*coords.shape)
        intg, ctxt = setup_velocity_verlet(bound_potentials, coords, box, dt, masses)  # noqa
        ctxt.set_v_t(v0)

        # check "public" .step and .multiple_steps implementations
        assert_reversibility_using_step_implementations(ctxt, n_steps, atol=1e-10)


def test_matches_reference():
    np.random.seed(2022)

    unbound_potentials, sys_params, coords, masses = get_relative_hif2a_in_vacuum()
    box = 100 * np.eye(3)

    dt = 1.5e-3

    summed_potential = SummedPotential(unbound_potentials, sys_params)
    summed_potential.bind(np.concatenate([param.reshape(-1) for param in sys_params]))
    bound_summed = summed_potential.bound_impl(np.float32)

    def force(coords):
        du_dxs = bound_summed.execute(coords, box)[0]
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

    xs, _ = ctxt.multiple_steps(n_steps, 1)
    assert xs.shape[0] == n_steps
    v1 = ctxt.get_v_t()
    atol = 1e-5
    np.testing.assert_allclose(ref_xs[1:-1], xs, atol=atol)
    np.testing.assert_allclose(ref_vs[-1], v1, atol=atol)


def test_initialization_and_finalization():
    np.random.seed(2022)

    # define a Python force fxn that calls custom_ops
    unbound_potentials, sys_params, coords, masses = get_relative_hif2a_in_vacuum()
    bound_potentials = [
        ubp.bind(params).bound_impl(np.float32) for (ubp, params) in zip(unbound_potentials, sys_params)
    ]
    box = 100 * np.eye(3)

    dt = 1.5e-3

    intg, ctxt = setup_velocity_verlet(bound_potentials, coords, box, dt, masses)  # noqa
    with pytest.raises(RuntimeError) as e:
        ctxt.finalize()
    assert "not initialized" in str(e.value)

    ctxt.initialize()

    with pytest.raises(RuntimeError) as e:
        ctxt.initialize()
    assert "initialized twice" in str(e.value)

    ctxt.finalize()

    with pytest.raises(RuntimeError) as e:
        ctxt.finalize()
    assert "not initialized" in str(e.value)


def test_verlet_with_multiple_steps_local():
    """Ensure Local MD can be run with the Velocity Verlet integrator and can handle a temperature passed in"""
    np.random.seed(2022)

    mol, _ = get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    dt = 1.5e-3

    # Have to minimize, else there can be clashes and the local moves will cause crashes
    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(mol, ff, 0.0)
    bound_potentials = [
        ubp.bind(params).bound_impl(np.float32) for (ubp, params) in zip(unbound_potentials, sys_params)
    ]
    box = 100 * np.eye(3)

    dt = 1.5e-3

    local_idxs = np.arange(len(coords) // 2, len(coords), dtype=np.int32)

    intg_impl, ctxt = setup_velocity_verlet(bound_potentials, coords, box, dt, masses)  # noqa

    n_steps = 10

    ctxt.multiple_steps_local(n_steps, local_idxs)

    # Uses non-default temperature as the integrator is not a thermostat, only changes the selection probabilities for local MD
    ctxt.multiple_steps_local(n_steps, local_idxs, temperature=100)
