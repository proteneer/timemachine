import numpy as np
import pytest

from timemachine.constants import AVOGADRO, BAR_TO_KJ_PER_NM3, BOLTZ, DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe import model_utils
from timemachine.fe.free_energy import AbsoluteFreeEnergy, HostConfig
from timemachine.fe.topology import BaseTopology
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator, custom_ops
from timemachine.md.barostat.moves import CentroidRescaler
from timemachine.md.barostat.utils import compute_box_center, compute_box_volume, get_bond_list, get_group_indices
from timemachine.md.builders import build_water_system
from timemachine.md.enhanced import get_solvent_phase_system
from timemachine.md.thermostat.utils import sample_velocities
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


@pytest.mark.memcheck
def test_barostat_validation():
    temperature = DEFAULT_TEMP  # kelvin
    pressure = DEFAULT_PRESSURE  # bar
    barostat_interval = 3  # step count
    seed = 2023

    np.random.seed(seed)

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
        mol_a, ff, lamb=0.0, minimize_energy=False
    )

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        u_impls.append(unbound_pot.bind(params).to_gpu(precision=np.float32).bound_impl)

    # Invalid interval
    with pytest.raises(RuntimeError, match="interval must be greater than 0"):
        custom_ops.MonteCarloBarostat(coords.shape[0], pressure, temperature, [[0, 1]], -1, u_impls, seed, True, 0.0)

    # Atom index over N
    with pytest.raises(RuntimeError, match="Grouped indices must be between 0 and N"):
        custom_ops.MonteCarloBarostat(
            coords.shape[0],
            pressure,
            temperature,
            [[0, coords.shape[0] + 1]],
            barostat_interval,
            u_impls,
            seed,
            True,
            0.0,
        )

    # Atom index < 0
    with pytest.raises(RuntimeError, match="Grouped indices must be between 0 and N"):
        custom_ops.MonteCarloBarostat(
            coords.shape[0], pressure, temperature, [[-1, 0]], barostat_interval, u_impls, seed, True, 0.0
        )

    # Atom index in two groups
    with pytest.raises(RuntimeError, match="All grouped indices must be unique"):
        custom_ops.MonteCarloBarostat(
            coords.shape[0], pressure, temperature, [[0, 1], [1, 2]], barostat_interval, u_impls, seed, True, 0.0
        )


@pytest.mark.memcheck
def test_barostat_with_clashes():
    temperature = DEFAULT_TEMP  # kelvin
    pressure = DEFAULT_PRESSURE  # bar
    timestep = 1.5e-3  # picosecond
    barostat_interval = 3  # step count
    collision_rate = 1.0  # 1 / picosecond
    seed = 2023

    np.random.seed(seed)

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # Set the lambda to 0.0 and don't minimize, resulting in clashes in the system
    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
        mol_a, ff, lamb=0.0, minimize_energy=False
    )
    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    # Cut the number of groups in half
    group_indices = group_indices[len(group_indices) // 2 :]

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        u_impls.append(unbound_pot.bind(params).to_gpu(precision=np.float32).bound_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses, temperature, seed)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
    )

    # The clashes will result in overflows, so the box should never change as no move is accepted
    ctxt = custom_ops.Context(coords, v_0, box, integrator_impl, u_impls, movers=[baro])
    ctxt.multiple_steps(barostat_interval * 100)
    assert np.all(box == ctxt.get_box())


@pytest.mark.memcheck
def test_barostat_zero_interval():
    pressure = DEFAULT_PRESSURE  # bar
    temperature = DEFAULT_TEMP  # kelvin
    seed = 2021
    np.random.seed(seed)

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    unbound_potentials, sys_params, masses, coords, _ = get_solvent_phase_system(
        mol_a, ff, lamb=1.0, minimize_energy=False
    )

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    with pytest.raises(RuntimeError):
        custom_ops.MonteCarloBarostat(
            coords.shape[0], pressure, temperature, group_indices, 0, u_impls, seed, True, 0.0
        )
    # Setting it to 1 should be valid.
    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0], pressure, temperature, group_indices, 1, u_impls, seed, True, 0.0
    )
    # Setting back to 0 should raise another error
    with pytest.raises(RuntimeError):
        baro.set_interval(0)


@pytest.mark.memcheck
def test_barostat_partial_group_idxs():
    """Verify that the barostat can handle a subset of the molecules
    rather than all of them. This test only verify that it runs, not the behavior"""
    lam = 1.0
    temperature = DEFAULT_TEMP  # kelvin
    timestep = 1.5e-3  # picosecond
    barostat_interval = 3  # step count
    collision_rate = 1.0  # 1 / picosecond

    seed = 2021
    np.random.seed(seed)

    pressure = DEFAULT_PRESSURE  # bar
    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    unbound_potentials, sys_params, masses, coords, complex_box = get_solvent_phase_system(
        mol_a, ff, lam, minimize_energy=False
    )

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    # Cut the number of groups in half
    group_indices = group_indices[len(group_indices) // 2 :]

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses, temperature, seed)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
    )

    ctxt = custom_ops.Context(coords, v_0, complex_box, integrator_impl, u_impls, movers=[baro])
    ctxt.multiple_steps(barostat_interval * 100)


@pytest.mark.memcheck
def test_barostat_is_deterministic():
    """Verify that the barostat results in the same box size shift after a fixed number of steps
    This is important to debugging as well as providing the ability to replicate
    simulations
    """
    lam = 1.0
    temperature = DEFAULT_TEMP
    timestep = 1.5e-3
    barostat_interval = 3
    collision_rate = 1.0
    seed = 2021
    np.random.seed(seed)

    pressure = DEFAULT_PRESSURE

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    host_system, host_coords, host_box, host_top = build_water_system(3.0, ff.water_ff)
    bt = BaseTopology(mol_a, ff)
    afe = AbsoluteFreeEnergy(mol_a, bt)
    host_config = HostConfig(host_system, host_coords, host_box, host_coords.shape[0])
    unbound_potentials, sys_params, masses = afe.prepare_host_edge(ff.get_params(), host_config, lam)
    coords = afe.prepare_combined_coords(host_coords=host_coords)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(params)
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )

    v_0 = sample_velocities(masses, temperature, seed)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
    )

    ctxt = custom_ops.Context(coords, v_0, host_box, integrator.impl(), u_impls, movers=[baro])
    ctxt.multiple_steps(15)
    atm_box = ctxt.get_box()
    # Verify that the volume of the box has changed
    assert compute_box_volume(atm_box) != compute_box_volume(host_box)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
    )
    ctxt = custom_ops.Context(coords, v_0, host_box, integrator.impl(), u_impls, movers=[baro])
    ctxt.multiple_steps(15)
    # Verify that we get back bitwise reproducible boxes
    assert compute_box_volume(atm_box) == compute_box_volume(ctxt.get_box())


def test_barostat_varying_pressure():
    lam = 1.0
    temperature = DEFAULT_TEMP
    timestep = 1.5e-3
    barostat_interval = 3
    collision_rate = 1.0
    seed = 2021
    np.random.seed(seed)

    # Start out with a very large pressure
    pressure = 1013.0
    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    unbound_potentials, sys_params, masses, coords, complex_box = get_solvent_phase_system(mol_a, ff, lam, margin=0.0)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses, temperature, seed)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
    )

    ctxt = custom_ops.Context(coords, v_0, complex_box, integrator_impl, u_impls, movers=[baro])
    ctxt.multiple_steps(1000)
    ten_atm_box = ctxt.get_box()
    ten_atm_box_vol = compute_box_volume(ten_atm_box)
    # Expect the box to shrink thanks to the barostat
    assert compute_box_volume(complex_box) - ten_atm_box_vol > 0.4

    # Set the pressure to 1 atm
    baro.set_pressure(DEFAULT_PRESSURE)
    # Changing the barostat interval resets the barostat step.
    baro.set_interval(2)

    ctxt.multiple_steps(2000)
    atm_box = ctxt.get_box()
    # Box will grow thanks to the lower pressure
    assert compute_box_volume(atm_box) > ten_atm_box_vol


# test that barostat only proposes properly re-centered coordinates
def test_barostat_recentering_upon_acceptance():
    lam = 1.0
    temperature = DEFAULT_TEMP
    pressure = DEFAULT_PRESSURE
    timestep = 1.5e-3
    barostat_interval = 10
    collision_rate = 1.0
    seed = 2023
    np.random.seed(seed)

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    unbound_potentials, sys_params, masses, coords, complex_box = get_solvent_phase_system(mol_a, ff, lam, margin=0.0)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses, temperature, seed)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
    )
    ctxt = custom_ops.Context(coords, v_0, complex_box, integrator_impl, u_impls, movers=[baro])
    # mini equilibrate the system to get barostat proposals to be reasonable
    ctxt.multiple_steps(1000)
    num_accepted = 0
    for _ in range(100):
        ctxt.multiple_steps(100)
        x_t = ctxt.get_x_t()
        box_t = ctxt.get_box()
        new_x_t, new_box_t = baro.move(x_t, box_t)
        if not np.all(box_t == new_box_t):
            for atom_idxs in group_indices:
                xyz = np.mean(new_x_t[atom_idxs], axis=0)
                ref_xyz = np.mean(model_utils.image_molecule(new_x_t[atom_idxs], new_box_t), axis=0)
                np.testing.assert_allclose(xyz, ref_xyz)
                x, y, z = xyz
                assert x > 0 and x < new_box_t[0][0]
                assert y > 0 and y < new_box_t[1][1]
                assert z > 0 and z < new_box_t[2][2]

            num_accepted += 1
        else:
            np.testing.assert_array_equal(new_x_t, x_t)
            np.testing.assert_array_equal(new_box_t, box_t)

    assert num_accepted > 0


def test_molecular_ideal_gas():
    """


    References
    ----------
    OpenMM testIdealGas
    https://github.com/openmm/openmm/blob/d8ef57fed6554ec95684e53768188e1f666405c9/tests/TestMonteCarloBarostat.h#L86-L140
    """

    # simulation parameters
    timestep = 1.5e-3
    collision_rate = 1.0
    n_moves = 10000
    barostat_interval = 5
    seed = 2021

    # thermodynamic parameters
    temperatures = np.array([300, 600, 1000])
    pressure = 100.0  # very high pressure, to keep the expected volume small

    # generate an alchemical system of a waterbox + alchemical ligand:
    # effectively discard ligands by running in AbsoluteFreeEnergy mode at lambda = 1.0
    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    _unbound_potentials, _sys_params, masses, coords, complex_box = get_solvent_phase_system(
        mol_a, ff, lamb=1.0, margin=0.0
    )

    # drop the nonbonded potential
    unbound_potentials = _unbound_potentials[:-1]
    sys_params = _sys_params[:-1]

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    volume_trajs = []

    relative_tolerance = 1e-2
    initial_relative_box_perturbation = 2 * relative_tolerance

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    # expected volume
    n_water_mols = len(group_indices) - 1  # 1 for the ligand
    expected_volume_in_md = (n_water_mols + 1) * BOLTZ * temperatures / (pressure * AVOGADRO * BAR_TO_KJ_PER_NM3)

    for i, temperature in enumerate(temperatures):
        # define a thermostat
        integrator = LangevinIntegrator(
            temperature,
            timestep,
            collision_rate,
            masses,
            seed,
        )
        integrator_impl = integrator.impl()

        v_0 = sample_velocities(masses, temperature, seed)

        # rescale the box to be approximately the desired box volume already
        rescaler = CentroidRescaler(group_indices)
        initial_volume = compute_box_volume(complex_box)
        initial_center = compute_box_center(complex_box)
        length_scale = ((1 + initial_relative_box_perturbation) * expected_volume_in_md[i] / initial_volume) ** (
            1.0 / 3
        )
        new_coords = rescaler.scale_centroids(coords, initial_center, length_scale)
        new_box = complex_box * length_scale

        baro = custom_ops.MonteCarloBarostat(
            new_coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
        )

        ctxt = custom_ops.Context(new_coords, v_0, new_box, integrator_impl, u_impls, movers=[baro])
        vols = []
        for move in range(n_moves // barostat_interval):
            ctxt.multiple_steps(barostat_interval)
            new_box = ctxt.get_box()
            volume = np.linalg.det(new_box)
            vols.append(volume)
        volume_trajs.append(vols)

    equil_time = len(volume_trajs[0]) // 2  # TODO: don't hard-code this?
    actual_volume_in_md = np.array([np.mean(volume_traj[equil_time:]) for volume_traj in volume_trajs])

    np.testing.assert_allclose(actual=actual_volume_in_md, desired=expected_volume_in_md, rtol=relative_tolerance)


def convert_to_fzset(grp_idxs):
    all_items = set()
    for grp in grp_idxs:
        items = set()
        for idx in grp:
            items.add(idx)
        items = frozenset(items)
        all_items.add(items)
    all_items = frozenset(all_items)
    return all_items


def assert_group_idxs_are_equal(set_a, set_b):
    assert convert_to_fzset(set_a) == convert_to_fzset(set_b)


@pytest.mark.nocuda
def test_get_group_indices():
    """
    Test that we generate correct group indices even when there are disconnected atoms (eg. ions) present

    Note that indices must be consecutive within each mol
    """

    bond_idxs = [[1, 0], [1, 2], [5, 6]]
    test_idxs = get_group_indices(bond_idxs, num_atoms=7)

    ref_idxs = [(0, 1, 2), (5, 6), (3,), (4,)]
    assert_group_idxs_are_equal(ref_idxs, test_idxs)

    test_idxs = get_group_indices([], num_atoms=4)
    ref_idxs = [(0,), (1,), (2,), (3,)]
    assert_group_idxs_are_equal(ref_idxs, test_idxs)

    test_idxs = get_group_indices([], num_atoms=0)
    ref_idxs = []
    assert_group_idxs_are_equal(ref_idxs, test_idxs)

    # slightly larger connected group
    test_idxs = get_group_indices([[0, 1], [1, 3], [3, 2]], num_atoms=5)
    ref_idxs = [(0, 1, 2, 3), (4,)]
    assert_group_idxs_are_equal(ref_idxs, test_idxs)

    with pytest.raises(AssertionError):
        # num_atoms <  an atom's index in bond_idxs
        get_group_indices([[0, 3]], num_atoms=3)


@pytest.mark.memcheck
def test_barostat_scaling_behavior():
    """Verify that it is possible to retrieve and set the volume scaling factor. Also check that the adaptive behavior of the scaling can be disabled"""
    lam = 1.0
    temperature = DEFAULT_TEMP
    timestep = 1.5e-3
    barostat_interval = 3
    collision_rate = 1.0
    seed = 2021
    np.random.seed(seed)

    pressure = DEFAULT_PRESSURE

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    host_system, host_coords, host_box, host_top = build_water_system(3.0, ff.water_ff)
    bt = BaseTopology(mol_a, ff)
    afe = AbsoluteFreeEnergy(mol_a, bt)
    host_config = HostConfig(host_system, host_coords, host_box, host_coords.shape[0])
    unbound_potentials, sys_params, masses = afe.prepare_host_edge(ff.get_params(), host_config, lam)
    coords = afe.prepare_combined_coords(host_coords=host_coords)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(params)
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )

    v_0 = sample_velocities(masses, temperature, seed)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
    )
    # Initial volume scaling is 0
    assert baro.get_volume_scale_factor() == 0.0
    assert baro.get_adaptive_scaling()

    ctxt = custom_ops.Context(coords, v_0, host_box, integrator.impl(), u_impls, movers=[baro])
    ctxt.multiple_steps(15)

    # Verify that the volume scaling is non-zero
    scaling = baro.get_volume_scale_factor()
    assert scaling > 0

    # Set to an intentionally bad factor to ensure it adapts
    bad_scaling_factor = 0.5 * compute_box_volume(host_box)
    baro.set_volume_scale_factor(bad_scaling_factor)
    assert baro.get_volume_scale_factor() == bad_scaling_factor
    ctxt.multiple_steps(100)
    # The scaling should adapt between moves
    assert bad_scaling_factor > baro.get_volume_scale_factor()

    # Reset the scaling to the previous value
    baro.set_volume_scale_factor(scaling)
    assert scaling == baro.get_volume_scale_factor()

    # Set back to the initial volume scaling, effectively disabling the barostat
    baro.set_volume_scale_factor(0.0)
    baro.set_adaptive_scaling(False)
    assert not baro.get_adaptive_scaling()
    ctxt.multiple_steps(100)
    assert baro.get_volume_scale_factor() == 0.0

    # Turning adaptive scaling back on should change the scaling after some MD
    baro.set_adaptive_scaling(True)
    assert baro.get_adaptive_scaling()
    ctxt.multiple_steps(100)
    assert baro.get_volume_scale_factor() != 0.0

    # Check that the adaptive_scaling_enabled, initial_volume_scale_factor constructor arguments works as expected
    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0],
        pressure,
        temperature,
        group_indices,
        barostat_interval,
        u_impls,
        seed,
        False,
        initial_volume_scale_factor=1.23,
    )
    assert not baro.get_adaptive_scaling()
    assert baro.get_volume_scale_factor() == 1.23
