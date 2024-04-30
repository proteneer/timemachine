import gc
import re
import weakref

import jax
import numpy as np
import pytest
from common import prepare_nb_system

from timemachine import constants
from timemachine.ff import Forcefield
from timemachine.integrator import langevin_coefficients
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, VelocityVerletIntegrator, custom_ops
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.enhanced import get_solvent_phase_system
from timemachine.md.minimizer import check_force_norm
from timemachine.potentials import SummedPotential
from timemachine.testsystems.ligands import get_biphenyl

pytestmark = [pytest.mark.memcheck]


def test_multiple_steps_store_interval():
    np.random.seed(2022)

    N = 8
    D = 3

    x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

    E = 2

    params, potential = prepare_nb_system(x0, E, p_scale=3.0, cutoff=1.0)
    test_nrg = potential.to_gpu(precision=np.float64)

    masses = np.random.rand(N)
    v0 = np.random.rand(x0.shape[0], x0.shape[1])

    temperature = constants.DEFAULT_TEMP
    dt = 2e-3
    friction = 0.0

    box = np.eye(3) * 3.0
    intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

    bp = test_nrg.bind(params).bound_impl
    bps = [bp]

    ctxt = custom_ops.Context(x0, v0, box, intg, bps)
    test_xs, test_boxes = ctxt.multiple_steps(10, 10)
    assert len(test_xs) == 1
    assert len(test_xs) == len(test_boxes)
    # We should not get out the input frame
    assert np.any(np.not_equal(x0, test_xs[0]))

    # The current coordinates should match, as the number of steps and the interval match
    np.testing.assert_array_equal(test_xs[0], ctxt.get_x_t())
    _, _ = bps[0].execute(test_xs[0], test_boxes[0])

    # Given an interval greater than the number of steps, return empty arrays
    test_xs, test_boxes = ctxt.multiple_steps(10, 100)
    assert len(test_xs) == 0
    assert len(test_boxes) == 0

    # Given interval of 0, return the last frame
    test_xs, test_boxes = ctxt.multiple_steps(10, 0)
    assert len(test_xs) == 1
    assert len(test_boxes) == 1

    np.testing.assert_array_equal(test_xs[0], ctxt.get_x_t())
    _, _ = bps[0].execute(test_xs[0], test_boxes[0])


def test_set_and_get():
    """
    This test the setters and getters in the context.
    """

    np.random.seed(4321)

    N = 8
    D = 3

    x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

    E = 2

    params, potential = prepare_nb_system(x0, E, p_scale=3.0, cutoff=1.0)
    test_nrg = potential.to_gpu(precision=np.float64)

    masses = np.random.rand(N)
    v0 = np.random.rand(x0.shape[0], x0.shape[1])

    temperature = constants.DEFAULT_TEMP
    dt = 2e-3
    friction = 0.0

    box = np.eye(3) * 3.0
    intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

    bp = test_nrg.bind(params).bound_impl
    bps = [bp]

    ctxt = custom_ops.Context(x0, v0, box, intg, bps)

    np.testing.assert_equal(ctxt.get_x_t(), x0)
    np.testing.assert_equal(ctxt.get_v_t(), v0)
    np.testing.assert_equal(ctxt.get_box(), box)

    new_x = np.random.rand(N, 3)
    ctxt.set_x_t(new_x)

    with pytest.raises(RuntimeError, match="number of new coords disagree with current coords"):
        bad_x = np.random.rand(N + 1, 3)
        ctxt.set_x_t(bad_x)

    np.testing.assert_equal(ctxt.get_x_t(), new_x)

    new_v = np.random.rand(N, 3)
    ctxt.set_v_t(new_v)

    with pytest.raises(RuntimeError, match="number of new velocities disagree with current coords"):
        bad_v = np.random.rand(N - 1, 3)
        ctxt.set_v_t(bad_v)

    np.testing.assert_equal(ctxt.get_v_t(), new_v)

    new_box = np.eye(3) * np.random.rand(3, 3)
    ctxt.set_box(new_box)

    with pytest.raises(RuntimeError, match="box must be 3x3"):
        bad_box = np.random.rand(3, 4)
        ctxt.set_box(bad_box)

    np.testing.assert_equal(ctxt.get_box(), new_box)


def test_fwd_mode():
    """
    This test verifies that stepping forward in time matches whether using the
    reference or the GPU platform.
    """

    np.random.seed(4321)

    N = 8
    D = 3

    x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

    E = 2

    params, potential = prepare_nb_system(
        x0,
        E,
        p_scale=3.0,
        # cutoff=0.5,
        cutoff=1.0,
    )
    ref_nrg_fn = potential
    test_nrg = potential.to_gpu(precision=np.float64)

    masses = np.random.rand(N)

    v0 = np.random.rand(x0.shape[0], x0.shape[1])

    num_steps = 12
    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 0.0
    ca, cbs, ccs = langevin_coefficients(temperature, dt, friction, masses)

    # not convenient to simulate identical trajectories otherwise
    assert (ccs == 0).all()

    def integrate_once_through(x_t, v_t, box, params):
        dU_dx_fn = jax.grad(ref_nrg_fn, argnums=(0,))
        dU_dp_fn = jax.grad(ref_nrg_fn, argnums=(1,))

        all_du_dps = []
        all_xs = []
        all_du_dxs = []
        all_us = []

        def compute_reference_values():
            u = ref_nrg_fn(x_t, params, box)
            all_us.append(u)
            du_dp = dU_dp_fn(x_t, params, box)[0]
            all_du_dps.append(du_dp)
            du_dx = dU_dx_fn(x_t, params, box)[0]
            all_du_dxs.append(du_dx)
            all_xs.append(x_t)

        for step in range(num_steps):
            compute_reference_values()

            noise = np.random.randn(*v_t.shape)
            force = -all_du_dxs[-1]
            v_mid = v_t + np.expand_dims(cbs, axis=-1) * force

            v_t = ca * v_mid + np.expand_dims(ccs, axis=-1) * noise
            x_t += 0.5 * dt * (v_mid + v_t)

        # Compute them for the last set of coords
        compute_reference_values()
        return all_xs, all_du_dxs, all_du_dps, all_us

    box = np.eye(3) * 3.0

    # when we have multiple parameters, we need to set this up correctly
    ref_all_xs, ref_all_du_dxs, ref_all_du_dps, ref_all_us = integrate_once_through(x0, v0, box, params)

    intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

    bp = test_nrg.bind(params).bound_impl
    bps = [bp]

    ctxt = custom_ops.Context(x0, v0, box, intg, bps)
    ctxt.initialize()
    for step in range(num_steps):
        print("comparing step", step)
        test_x_t = ctxt.get_x_t()
        np.testing.assert_allclose(test_x_t, ref_all_xs[step])
        test_du_dx_t, _ = bp.execute(test_x_t, box)
        ctxt.step()
        # np.testing.assert_allclose(test_u_t, ref_all_us[step])
        np.testing.assert_allclose(test_du_dx_t, ref_all_du_dxs[step])
    ctxt.finalize()
    # test the multiple_steps method
    ctxt_2 = custom_ops.Context(x0, v0, box, intg, bps)

    x_interval = 2
    start_box = ctxt_2.get_box()
    test_xs, test_boxes = ctxt_2.multiple_steps(num_steps, x_interval)
    end_box = ctxt_2.get_box()

    np.testing.assert_allclose(test_xs, ref_all_xs[x_interval::x_interval])
    np.testing.assert_array_equal(start_box, end_box)
    for i in range(test_boxes.shape[0]):
        np.testing.assert_array_equal(start_box, test_boxes[i])
    assert test_boxes.shape[0] == test_xs.shape[0]
    assert test_boxes.shape[1] == D
    assert test_boxes.shape[2] == test_xs.shape[2]


@pytest.mark.parametrize("freeze_reference", [True, False])
def test_multiple_steps_local_validation(freeze_reference):
    seed = 2022
    np.random.seed(seed)

    N = 8
    D = 3

    coords = np.random.rand(N, D).astype(dtype=np.float64) * 2
    box = np.eye(3) * 3.0
    masses = np.random.rand(N)

    E = 2

    params, potential = prepare_nb_system(
        coords,
        E,
        p_scale=3.0,
        cutoff=1.0,
    )
    nb_pot = potential.to_gpu(np.float32)

    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    radius = 1.2

    v0 = np.zeros_like(coords)
    bps = [nb_pot.bind(params).bound_impl]

    verlet = VelocityVerletIntegrator(dt, masses)

    ctxt = custom_ops.Context(coords, v0, box, verlet.impl(), bps)
    # If setup_local_md called with a temperature explicitly, can't use VelocityVerlet
    with pytest.raises(RuntimeError, match="integrator must be LangevinIntegrator."):
        ctxt.multiple_steps_local(100, np.array([0], dtype=np.int32))

    # Verify that indices are correctly checked
    ctxt = custom_ops.Context(coords, v0, box, verlet.impl(), bps)

    # Must use a non-zero temperature
    with pytest.raises(RuntimeError, match="temperature must be greater than 0"):
        ctxt.setup_local_md(0.0, freeze_reference)

    ctxt.setup_local_md(temperature, freeze_reference)
    with pytest.raises(RuntimeError, match="indices can't be empty"):
        ctxt.multiple_steps_local(100, np.array([], dtype=np.int32), radius=radius)

    with pytest.raises(RuntimeError, match="index values must be less than N"):
        ctxt.multiple_steps_local(100, np.array([N * 2], dtype=np.int32), radius=radius)

    with pytest.raises(RuntimeError, match="index values must be greater or equal to zero"):
        ctxt.multiple_steps_local(100, np.array([-1], dtype=np.int32), radius=radius)

    with pytest.raises(RuntimeError, match="atom indices must be unique"):
        ctxt.multiple_steps_local(100, np.array([1, 1], dtype=np.int32), radius=radius)

    with pytest.raises(RuntimeError, match="radius must be greater or equal to 0.1"):
        ctxt.multiple_steps_local(100, np.array([1], dtype=np.int32), radius=0.01)

    with pytest.raises(RuntimeError, match="k must be at least one"):
        ctxt.multiple_steps_local(100, np.array([1], dtype=np.int32), k=0.0)

    with pytest.raises(RuntimeError, match=re.escape("k must be less than than 1e+06")):
        ctxt.multiple_steps_local(100, np.array([1], dtype=np.int32), k=1e7)


@pytest.mark.parametrize("freeze_reference", [True, False])
def test_multiple_steps_local_selection_validation(freeze_reference):
    seed = 2022
    np.random.seed(seed)

    N = 8
    D = 3

    coords = np.random.rand(N, D).astype(dtype=np.float64) * 2
    box = np.eye(3) * 3.0
    masses = np.random.rand(N)

    E = 2

    params, potential = prepare_nb_system(
        coords,
        E,
        p_scale=3.0,
        cutoff=1.0,
    )
    nb_pot = potential.to_gpu(np.float32)

    dt = 1.5e-3
    radius = 1.2

    v0 = np.zeros_like(coords)
    bps = [nb_pot.bind(params).bound_impl]

    # Compatible with local NVE since multiple_steps_local_selection doesn't depend on temperature
    verlet = VelocityVerletIntegrator(dt, masses)
    intg_impl = verlet.impl()

    reference_idx = 0

    # Verify that indices are correctly checked
    ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)
    ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)
    with pytest.raises(RuntimeError, match="indices can't be empty"):
        ctxt.multiple_steps_local_selection(100, reference_idx, np.array([], dtype=np.int32), radius=radius)

    with pytest.raises(RuntimeError, match="index values must be less than N"):
        ctxt.multiple_steps_local_selection(100, reference_idx, np.array([N * 2], dtype=np.int32), radius=radius)

    with pytest.raises(RuntimeError, match="index values must be greater or equal to zero"):
        ctxt.multiple_steps_local_selection(100, reference_idx, np.array([-1], dtype=np.int32), radius=radius)

    with pytest.raises(RuntimeError, match="atom indices must be unique"):
        ctxt.multiple_steps_local_selection(100, reference_idx, np.array([1, 1], dtype=np.int32), radius=radius)

    with pytest.raises(RuntimeError, match="radius must be greater or equal to 0.1"):
        ctxt.multiple_steps_local_selection(100, reference_idx, np.array([1], dtype=np.int32), radius=0.01)

    with pytest.raises(RuntimeError, match="k must be at least one"):
        ctxt.multiple_steps_local_selection(100, reference_idx, np.array([1], dtype=np.int32), k=0.0)

    with pytest.raises(RuntimeError, match=re.escape("k must be less than than 1e+06")):
        ctxt.multiple_steps_local_selection(100, reference_idx, np.array([1], dtype=np.int32), k=1e7)

    with pytest.raises(RuntimeError, match="reference idx must not be in selection idxs"):
        ctxt.multiple_steps_local_selection(100, reference_idx, np.array([reference_idx], dtype=np.int32))

    with pytest.raises(RuntimeError, match=f"reference idx must be at least 0 and less than {N}"):
        ctxt.multiple_steps_local_selection(100, N, np.array([3], dtype=np.int32))

    with pytest.raises(RuntimeError, match=f"reference idx must be at least 0 and less than {N}"):
        ctxt.multiple_steps_local_selection(100, -1, np.array([3], dtype=np.int32))


@pytest.mark.parametrize("freeze_reference", [True, False])
def test_multiple_steps_local_consistency(freeze_reference):
    """Verify that running multiple_steps_local is consistent.

    - Assert that particles nearby the local idxs move, particles far away do not
    - Assert that potentials used to run local md do return bit wise identical results.
      As multiple_steps_local modifies the potentials
    - Assert that running with a Barostat returns identical frames
    - Assert that wrapping potentials within a SummedPotential returns identical frames"""
    mol, _ = get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 0.0
    seed = 2022
    radius = 1.2
    num_steps = 500
    x_interval = 100

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(mol, ff, 0.0, minimize_energy=True)
    v0 = np.zeros_like(coords)
    bps = []
    for p, pot in zip(sys_params, unbound_potentials):
        bps.append(pot.bind(p).to_gpu(np.float32).bound_impl)

    reference_values = []
    for bp in bps:
        reference_values.append(bp.execute(coords, box))

    # Select the molecule as the local idxs
    local_idxs = np.arange(len(coords) - mol.GetNumAtoms(), len(coords), dtype=np.int32)

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

    intg_impl = intg.impl()

    ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)
    ctxt.setup_local_md(temperature, freeze_reference)
    # Run steps of local MD
    xs, boxes = ctxt.multiple_steps_local(num_steps, local_idxs, store_x_interval=x_interval, radius=radius)

    assert xs.shape[0] == num_steps // x_interval
    assert boxes.shape[0] == num_steps // x_interval

    expected_to_move = len(local_idxs) - 1 if freeze_reference else len(local_idxs)

    # Indices in local idxs should have moved, except for the one selected as frozen if freeze_reference
    assert np.all(coords[local_idxs] != xs[-1][local_idxs], axis=1).sum() == expected_to_move

    # Get the particles within a certain distance of local idxs
    nblist = custom_ops.Neighborlist_f32(len(coords))
    nblist.set_row_idxs(local_idxs.astype(np.uint32))
    # Add padding to the radius to account for probablistic selection
    ixn_list = nblist.get_nblist(coords, box, radius + 0.2)
    potential_selected_particles = np.concatenate(ixn_list)

    moving_idxs = np.concatenate([local_idxs, potential_selected_particles.reshape(-1)])
    assert np.any(coords[moving_idxs] != xs[-1][moving_idxs])

    frozen_idxs = np.delete(np.arange(0, len(coords)), moving_idxs)
    assert len(frozen_idxs) > 0
    assert np.all(coords[frozen_idxs] == xs[-1][frozen_idxs])

    # Verify that the bound potentials haven't been changed, as local md modifies potentials
    for ref_val, pot in zip(reference_values, bps):
        ref_du_dx, ref_u = ref_val
        test_du_dx, test_u = pot.execute(coords, box)
        np.testing.assert_array_equal(ref_du_dx, test_du_dx)
        np.testing.assert_equal(ref_u, test_u)
        check_force_norm(-ref_du_dx)

    # Verify that running with a barostat doesn't change the results
    group_idxs = get_group_indices(get_bond_list(unbound_potentials[0]), len(masses))

    pressure = 1.0

    barostat = MonteCarloBarostat(coords.shape[0], pressure, temperature, group_idxs, 1, seed)
    barostat_impl = barostat.impl(bps)

    intg_impl = intg.impl()

    ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps, movers=[barostat_impl])

    ctxt.setup_local_md(temperature, freeze_reference)
    baro_xs, baro_boxes = ctxt.multiple_steps_local(num_steps, local_idxs, store_x_interval=x_interval, radius=radius)

    assert baro_xs.shape == xs.shape
    assert baro_boxes.shape == boxes.shape

    # Results using a barostat should be identical.
    np.testing.assert_array_equal(baro_xs, xs)
    np.testing.assert_array_equal(baro_boxes, boxes)

    # Verify that wrapping up the potentials in a summed potential is identical
    summed_potential = SummedPotential(unbound_potentials, sys_params)
    # Flatten the arrays so we can concatenate them.
    summed_potential = summed_potential.bind(np.concatenate([p.reshape(-1) for p in sys_params]))
    bp = summed_potential.to_gpu(precision=np.float32).bound_impl

    intg_impl = intg.impl()

    # Rerun with the summed potential
    ctxt = custom_ops.Context(coords, v0, box, intg_impl, [bp])
    ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)
    summed_pot_xs, summed_pot_boxes = ctxt.multiple_steps_local(
        num_steps, local_idxs, store_x_interval=x_interval, radius=radius
    )

    assert summed_pot_xs.shape == xs.shape
    assert summed_pot_boxes.shape == boxes.shape

    # Results using a summed potential should be identical.
    np.testing.assert_array_equal(summed_pot_xs, xs)
    np.testing.assert_array_equal(summed_pot_boxes, boxes)


def test_get_movers():
    mol, _ = get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    pressure = constants.DEFAULT_PRESSURE
    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 0.0
    seed = 2022

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(mol, ff, 0.0, minimize_energy=False)
    v0 = np.zeros_like(coords)
    bps = []
    for p, pot in zip(sys_params, unbound_potentials):
        bps.append(pot.bind(p).to_gpu(np.float32).bound_impl)

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

    intg_impl = intg.impl()

    group_idxs = get_group_indices(get_bond_list(unbound_potentials[0]), len(masses))

    barostat = MonteCarloBarostat(coords.shape[0], pressure, temperature, group_idxs, 1, seed)
    barostat_impl = barostat.impl(bps)

    ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps, movers=[barostat_impl])

    assert barostat_impl == ctxt.get_barostat()

    movers = ctxt.get_movers()
    assert len(movers) == 1
    assert movers[0] == barostat_impl


@pytest.mark.parametrize("freeze_reference", [True, False])
def test_multiple_steps_local_entire_system(freeze_reference):
    """Verify that running multiple_steps_local is valid even when consuming the entire system, IE radius ~= inf.

    - If freeze_reference only a single particle should not move else all should move.
    """
    mol, _ = get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 1.0
    seed = 2022
    radius = np.inf
    num_steps = 5

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(mol, ff, 0.0, minimize_energy=False)
    v0 = np.zeros_like(coords)
    bps = []
    for p, pot in zip(sys_params, unbound_potentials):
        bps.append(pot.bind(p).to_gpu(np.float32).bound_impl)

    # Select the molecule as the local idxs
    local_idxs = np.arange(len(coords) - mol.GetNumAtoms(), len(coords), dtype=np.int32)

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

    intg_impl = intg.impl()

    ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)
    ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)

    xs, boxes = ctxt.multiple_steps_local(num_steps, local_idxs, radius=radius)

    assert xs.shape[0] == 1
    if freeze_reference:
        assert np.all(xs[0] == coords, axis=1).sum() == 1, "Expected only a single atom to be stationary"
    else:
        assert np.all(xs[0] != coords), "All coordinates should have moved"


def test_multiple_steps_local_no_free_particles():
    """Verify that running multiple_steps_local, with free_reference=True raises an exception if no free particles
    selected. In this case we can trigger this failure by having a single atom molecule, and moving it away from
    the water box. This is a pathological case, but to verify the exception

    This may need to be changed in the future if there are stochastic failures due to probabilistic selection
    of the free particle.
    """
    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 0.0
    seed = 2023
    N = 100
    D = 3
    radius = 0.1
    k = 1.0

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(1, size=(N, D)).astype(dtype=np.float64) * 2

    E = 2

    box = np.eye(3) * 1000.0

    params, potential = prepare_nb_system(x0, E, p_scale=3.0, cutoff=1.0)
    test_nrg = potential.to_gpu(np.float32)

    bps = [test_nrg.bind(params).bound_impl]

    masses = rng.uniform(1.0, size=N)
    v0 = rng.uniform(1.0, size=(x0.shape[0], x0.shape[1]))

    # Select the last particle the local idxs
    local_idxs = np.array([N - 1], dtype=np.int32)
    x0[local_idxs] += 100.0

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

    ctxt = custom_ops.Context(x0, v0, box, intg.impl(), bps)

    with pytest.raises(RuntimeError, match="no free particles"):
        xs, boxes = ctxt.multiple_steps_local(1, local_idxs, radius=radius, k=k, seed=seed)


@pytest.mark.parametrize("freeze_reference", [True, False])
def test_local_md_initialization(freeze_reference):
    """Verify that initialization of local md doesn't impact behavior of context."""
    seed = 2023
    np.random.seed(seed)

    N = 8
    D = 3

    coords = np.random.rand(N, D).astype(dtype=np.float64) * 2
    box = np.eye(3) * 3.0
    masses = np.random.rand(N)

    E = 2

    params, potential = prepare_nb_system(
        coords,
        E,
        p_scale=3.0,
        cutoff=1.0,
    )
    nb_pot = potential.to_gpu(np.float32)

    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 0.0

    # Select a single particle to use as the reference, will be frozen
    local_idxs = np.array([len(coords) - 1], dtype=np.int32)

    v0 = np.zeros_like(coords)
    bps = [nb_pot.bind(params).bound_impl]

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

    steps = 10

    # Construct context with no potentials, should fail to initialize.
    ctxt = custom_ops.Context(coords, v0, box, intg.impl(), [])
    with pytest.raises(RuntimeError, match="unable to find a NonbondedAllPairs potential"):
        ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)

    ctxt = custom_ops.Context(coords, v0, box, intg.impl(), bps * 2)

    # If you have multiple nonbonded potentials, should fail
    with pytest.raises(RuntimeError, match="found multiple NonbondedAllPairs potentials"):
        ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)

    # Verify that initializing local md doesn't modify global md behavior
    ctxt = custom_ops.Context(coords, v0, box, intg.impl(), bps)
    ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)

    # setup_local_md is idempotent if called with the same parameters
    ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)

    # If the parameters change, will raise an exception
    with pytest.raises(RuntimeError, match="local md configured with different parameters"):
        ctxt.setup_local_md(constants.DEFAULT_TEMP + 1, freeze_reference)
    with pytest.raises(RuntimeError, match="local md configured with different parameters"):
        ctxt.setup_local_md(constants.DEFAULT_TEMP, not freeze_reference)

    ref_xs, ref_boxes = ctxt.multiple_steps(steps)

    ctxt = custom_ops.Context(coords, v0, box, intg.impl(), bps)
    ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)
    comp_xs, comp_boxes = ctxt.multiple_steps(steps)

    np.testing.assert_array_equal(ref_xs, comp_xs)
    np.testing.assert_array_equal(ref_boxes, comp_boxes)

    # Verify that initializing local md doesn't modify local md behavior
    ctxt = custom_ops.Context(coords, v0, box, intg.impl(), bps)
    ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)
    ref_local_xs, ref_local_boxes = ctxt.multiple_steps_local(steps, local_idxs)

    ctxt = custom_ops.Context(coords, v0, box, intg.impl(), bps)
    ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)
    comp_local_xs, comp_local_boxes = ctxt.multiple_steps_local(steps, local_idxs)

    np.testing.assert_array_equal(ref_local_xs, comp_local_xs)
    np.testing.assert_array_equal(ref_local_boxes, comp_local_boxes)


@pytest.mark.parametrize("freeze_reference", [True, False])
def test_local_md_with_selection_mask(freeze_reference):
    """Verify that running local md with a selection mask works as expected"""
    seed = 2023
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    N = 8
    D = 3

    coords = rng.random(size=(N, D)).astype(dtype=np.float64)
    box = np.eye(3) * 3.0
    masses = rng.random(N)

    E = 2

    params, potential = prepare_nb_system(
        coords,
        E,
        p_scale=3.0,
        cutoff=1.0,
    )
    nb_pot = potential.to_gpu(np.float32)

    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 0.0

    v0 = np.zeros_like(coords)
    bps = [nb_pot.bind(params).bound_impl]

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

    idxs = np.arange(0, len(coords))

    reference_idx = rng.choice(idxs)

    free_particles = rng.choice(idxs, size=len(coords) // 2, replace=False)
    free_particles = np.delete(free_particles, free_particles == reference_idx)
    frozen_particles = np.delete(idxs, free_particles)

    steps = 10

    ctxt = custom_ops.Context(coords, v0, box, intg.impl(), bps)
    ctxt.setup_local_md(constants.DEFAULT_TEMP, freeze_reference)
    xs, boxes = ctxt.multiple_steps_local_selection(steps, reference_idx, free_particles.astype(np.int32))

    if not freeze_reference:
        free_particles = np.append(free_particles, reference_idx)
        frozen_particles = np.delete(frozen_particles, frozen_particles == reference_idx)

    # The free particles should have moved
    assert np.all(xs[-1][free_particles] != coords[free_particles])
    assert np.all(xs[-1][frozen_particles] == coords[frozen_particles])


def test_setup_context_with_references():
    mol, _ = get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 0.0
    seed = 2022
    pressure = constants.DEFAULT_PRESSURE

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(mol, ff, 1.0, minimize_energy=False)
    v0 = np.zeros_like(coords)

    def build_context(barostat_interval):
        """The context returned will segfault if any of the objects get cleaned up"""

        weak_refs = []
        bps = []
        for p, pot in zip(sys_params, unbound_potentials):
            bound_impl = pot.bind(p).to_gpu(np.float32).bound_impl
            bps.append(bound_impl)
            weak_refs.append(weakref.ref(bound_impl))

        intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

        movers = []
        if barostat_interval > 0:
            group_idxs = get_group_indices(get_bond_list(unbound_potentials[0]), len(masses))

            barostat = MonteCarloBarostat(coords.shape[0], pressure, temperature, group_idxs, 1, seed)
            barostat_impl = barostat.impl(bps)
            movers.append(barostat_impl)
            weak_refs.append(weakref.ref(barostat_impl))

        intg_impl = intg.impl()
        weak_refs.append(weakref.ref(intg_impl))

        return custom_ops.Context(coords, v0, box, intg_impl, bps, movers=movers), weak_refs

    # Without barostat
    ctxt, reffed_objs = build_context(0)
    xs, boxes = ctxt.multiple_steps(100)
    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(boxes))
    assert np.all(xs[-1] != coords)
    assert np.all(boxes[-1] == box)

    del ctxt
    gc.collect()
    for ref in reffed_objs:
        assert ref() is None

    # With Barostat
    ctxt, reffed_objs = build_context(10)
    xs, boxes = ctxt.multiple_steps(100)
    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(boxes))
    assert np.all(xs[-1] != coords)
    # Barostat should change box size
    assert np.all(np.diagonal(boxes[-1]) != np.diagonal(box))

    del ctxt
    gc.collect()
    for ref in reffed_objs:
        assert ref() is None


@pytest.mark.parametrize("freeze_reference", [True, False])
def test_local_md_nonbonded_all_pairs_subset(freeze_reference):
    """Test that if the nonbonded all pairs is set up on a subset of the system, that local MD can correctly
    simulate the local region and reset the idxs to the old value"""
    seed = 2022
    np.random.seed(seed)

    N = 30
    D = 3

    coords = np.random.rand(N, D).astype(dtype=np.float64) * 2
    box = np.eye(3) * 3.0
    masses = np.random.rand(N)

    E = 2

    params, potential = prepare_nb_system(
        coords,
        E,
        p_scale=3.0,
        cutoff=1.0,
    )

    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 0.0
    radius = 1.2

    # Select a single particle to use as the reference, will be frozen
    local_idxs = np.array([len(coords) - 1], dtype=np.int32)
    atom_idxs = np.random.choice(np.arange(N, dtype=np.int32), size=5, replace=False)
    potential.atom_idxs = atom_idxs
    nb_pot = potential.to_gpu(np.float32)

    v0 = np.zeros_like(coords)
    bps = [nb_pot.bind(params).bound_impl]

    ref_vals = [bp.execute(coords, box) for bp in bps]

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

    intg_impl = intg.impl()

    steps = 100

    ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)
    ctxt.setup_local_md(temperature, freeze_reference)
    ref_xs, ref_boxes = ctxt.multiple_steps_local(steps, local_idxs, radius=radius)

    for (ref_du_dx, ref_u), bp in zip(ref_vals, bps):
        test_du_dx, test_u = bp.execute(coords, box)
        np.testing.assert_array_equal(ref_du_dx, test_du_dx)
        np.testing.assert_equal(ref_u, test_u)
        check_force_norm(-ref_du_dx)
        test_du_dx, _ = bp.execute(ref_xs[-1], ref_boxes[-1])
        check_force_norm(-test_du_dx)


def test_context_invalid_boxes():
    """Verify that nonbonded all pairs potentials provided to the context will correctly validate the box size"""
    mol, _ = get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 0.0
    seed = 2022
    steps = 1
    rng = np.random.default_rng(seed)

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(mol, ff, 0.0, minimize_energy=False)
    v0 = np.zeros_like(coords)

    host_idxs = np.arange(len(coords) - mol.GetNumAtoms(), dtype=np.int32)
    ligand_idxs = np.arange(len(coords) - mol.GetNumAtoms(), len(coords), dtype=np.int32)
    reference_idx = rng.choice(ligand_idxs)
    selection = rng.choice(host_idxs, size=40, replace=False)

    bps = []
    for p, pot in zip(sys_params, unbound_potentials):
        bound_impl = pot.bind(p).to_gpu(np.float32).bound_impl
        bps.append(bound_impl)

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

    ctxt = custom_ops.Context(coords, v0, box, intg.impl(), bps)
    ctxt.multiple_steps(steps)
    ctxt.multiple_steps_local(steps, ligand_idxs)
    ctxt.multiple_steps_local_selection(steps, reference_idx, selection)

    # Make the box way too small, which should trigger the failure
    ctxt.set_box(box * 0.01)
    err_msg = "cutoff with padding is more than half of the box width, neighborlist is no longer reliable"
    with pytest.raises(RuntimeError, match=err_msg):
        _, boxes = ctxt.multiple_steps(steps)
        assert len(boxes) == 1
    with pytest.raises(RuntimeError, match=err_msg):
        _, boxes = ctxt.multiple_steps_local(steps, ligand_idxs)
        assert len(boxes) == 1
    with pytest.raises(RuntimeError, match=err_msg):
        _, boxes = ctxt.multiple_steps_local_selection(steps, reference_idx, selection)
        assert len(boxes) == 1

    # Without returning boxes no check will be performed
    _, boxes = ctxt.multiple_steps(steps, store_x_interval=steps + 1)
    assert len(boxes) == 0
    _, boxes = ctxt.multiple_steps_local(steps, ligand_idxs, store_x_interval=steps + 1)
    assert len(boxes) == 0
    _, boxes = ctxt.multiple_steps_local_selection(steps, reference_idx, selection, store_x_interval=steps + 1)
    assert len(boxes) == 0


def test_context_invalid_boxes_without_nonbonded_potentials():
    """Verify that without a nonbonded all pairs potentials the context performs no check"""
    mol, _ = get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    temperature = constants.DEFAULT_TEMP
    dt = 1.5e-3
    friction = 0.0
    seed = 2022
    steps = 1

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(mol, ff, 0.0, minimize_energy=False)
    v0 = np.zeros_like(coords)

    bps = []
    for p, pot in zip(sys_params, unbound_potentials):
        # Skip the nonbonded all pairs, which is a SummedPotential, will result in no check
        if isinstance(pot, SummedPotential):
            continue
        bound_impl = pot.bind(p).to_gpu(np.float32).bound_impl
        bps.append(bound_impl)

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

    ctxt = custom_ops.Context(coords, v0, box, intg.impl(), bps)

    # Make the box way too small, which should trigger the failure
    ctxt.set_box(box * 0.01)
    _, boxes = ctxt.multiple_steps(steps)
    assert len(boxes) == 1
