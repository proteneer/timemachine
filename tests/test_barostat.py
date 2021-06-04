import numpy as np
from simtk import unit
import time

from testsystems.relative import hif2a_ligand_pair

from md.builders import build_water_system
from md.minimizer import minimize_host_4d

from fe.free_energy import AbsoluteFreeEnergy

from md.states import CoordsVelBox
from md.ensembles import PotentialEnergyModel, NPTEnsemble
from md.thermostat.moves import UnadjustedLangevinMove
from md.barostat.moves import MonteCarloBarostat, CentroidRescaler
from md.barostat.utils import get_bond_list, get_group_indices, compute_box_volume, compute_box_center
from md.utils import simulate_npt_traj
from md.thermostat.utils import sample_velocities

from timemachine.lib import LangevinIntegrator, custom_ops

from functools import partial

from timemachine.constants import BOLTZ, ENERGY_UNIT, DISTANCE_UNIT


def test_barostat_partial_group_idxs():
    """Verify that the barostat can handle a subset of the molecules
    rather than all of them. This test only verify that it runs, not the behavior"""
    temperature = 300.0 * unit.kelvin
    initial_waterbox_width = 2.0 * unit.nanometer
    timestep = 1.5 * unit.femtosecond
    barostat_interval = 3
    collision_rate = 1.0 / unit.picosecond
    seed = 2021
    np.random.seed(seed)

    pressure = 1. * unit.atmosphere
    mol_a = hif2a_ligand_pair.mol_a
    ff = hif2a_ligand_pair.ff
    complex_system, complex_coords, complex_box, complex_top = build_water_system(
        initial_waterbox_width.value_in_unit(unit.nanometer))

    min_complex_coords = minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
    afe = AbsoluteFreeEnergy(mol_a, ff)

    unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
        ff.get_ordered_params(), complex_system, min_complex_coords
    )

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)

    # Cut the number of groups in half
    group_indices = group_indices[len(group_indices)//2:]
    lam = 1.0

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.bound_impl(precision=np.float32)
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature.value_in_unit(unit.kelvin),
        timestep.value_in_unit(unit.picosecond),
        collision_rate.value_in_unit(unit.picosecond**-1),
        masses,
        seed
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses * unit.amu, temperature)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0],
        pressure.value_in_unit(unit.bar),
        temperature.value_in_unit(unit.kelvin),
        group_indices,
        barostat_interval,
        u_impls,
        seed
    )

    ctxt = custom_ops.Context(coords, v_0, complex_box, integrator_impl, u_impls, barostat=baro)
    ctxt.multiple_steps(np.ones(1000)*lam)

def test_barostat_varying_pressure():
    temperature = 300.0 * unit.kelvin
    initial_waterbox_width = 2.0 * unit.nanometer
    timestep = 1.5 * unit.femtosecond
    barostat_interval = 3
    collision_rate = 1.0 / unit.picosecond
    seed = 2021
    np.random.seed(seed)

    # Start out with a very large pressure
    pressure = 1000. * unit.atmosphere
    mol_a = hif2a_ligand_pair.mol_a
    ff = hif2a_ligand_pair.ff
    complex_system, complex_coords, complex_box, complex_top = build_water_system(
        initial_waterbox_width.value_in_unit(unit.nanometer))

    min_complex_coords = minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
    afe = AbsoluteFreeEnergy(mol_a, ff)

    unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
        ff.get_ordered_params(), complex_system, min_complex_coords
    )


    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)

    trajs = []
    volume_trajs = []

    lam = 1.0

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.bound_impl(precision=np.float32)
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature.value_in_unit(unit.kelvin),
        timestep.value_in_unit(unit.picosecond),
        collision_rate.value_in_unit(unit.picosecond**-1),
        masses,
        seed
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses * unit.amu, temperature)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0],
        pressure.value_in_unit(unit.bar),
        temperature.value_in_unit(unit.kelvin),
        group_indices,
        barostat_interval,
        u_impls,
        seed
    )

    ctxt = custom_ops.Context(coords, v_0, complex_box, integrator_impl, u_impls, barostat=baro)
    ctxt.multiple_steps(np.ones(2000)*lam)
    ten_atm_box = ctxt.get_box()
    ten_atm_box_vol = compute_box_volume(ten_atm_box)
    # Expect the box to shrink thanks to the barostat
    assert ten_atm_box_vol < compute_box_volume(complex_box)

    # Set the pressure to 1 bar
    baro.set_pressure((1 * unit.atmosphere).value_in_unit(unit.bar))
    # Changing the barostat interval resets the barostat step.
    baro.set_interval(2)
    start = time.time()
    volumes = []
    while time.time() - start < 15:
        ctxt.multiple_steps(np.ones(1000)*lam)
        atm_box = ctxt.get_box()
        volumes.append(compute_box_volume(atm_box))

    # Box will grow thanks to the lower pressure
    assert np.abs(ten_atm_box_vol - np.mean(volumes[len(volumes)//2:])) >= 0.2

def test_molecular_ideal_gas():
    """


    References
    ----------
    OpenMM testIdealGas
    https://github.com/openmm/openmm/blob/d8ef57fed6554ec95684e53768188e1f666405c9/tests/TestMonteCarloBarostat.h#L86-L140
    """

    # simulation parameters
    initial_waterbox_width = 2.0 * unit.nanometer
    timestep = 1.5 * unit.femtosecond
    collision_rate = 1.0 / unit.picosecond
    n_moves = 10000
    barostat_interval = 5
    seed = 2021

    # thermodynamic parameters
    temperatures = np.array([300, 600, 1000]) * unit.kelvin
    pressure = 100. * unit.bar # very high pressure, to keep the expected volume small

    # generate an alchemical system of a waterbox + alchemical ligand:
    # effectively discard ligands by running in AbsoluteFreeEnergy mode at lambda = 1.0
    mol_a = hif2a_ligand_pair.mol_a
    ff = hif2a_ligand_pair.ff
    complex_system, complex_coords, complex_box, complex_top = build_water_system(
        initial_waterbox_width.value_in_unit(unit.nanometer))

    min_complex_coords = minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
    afe = AbsoluteFreeEnergy(mol_a, ff)

    _unbound_potentials, _sys_params, masses, coords = afe.prepare_host_edge(
        ff.get_ordered_params(), complex_system, min_complex_coords
    )

    # drop the nonbonded potential
    unbound_potentials = _unbound_potentials[:-1]
    sys_params = _sys_params[:-1]

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)

    volume_trajs = []

    lam = 1.0

    relative_tolerance = 1e-2
    initial_relative_box_perturbation = 2 * relative_tolerance

    n_molecules = complex_top.getNumResidues()

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.bound_impl(precision=np.float32)
        u_impls.append(bp_impl)

    # expected volume
    md_pressure_unit = ENERGY_UNIT / DISTANCE_UNIT ** 3
    pressure_in_md = (pressure * unit.AVOGADRO_CONSTANT_NA).value_in_unit(md_pressure_unit)
    expected_volume_in_md = (n_molecules + 1) * BOLTZ * temperatures.value_in_unit(unit.kelvin) / pressure_in_md


    for i, temperature in enumerate(temperatures):

        # define a thermostat
        integrator = LangevinIntegrator(
            temperature.value_in_unit(unit.kelvin),
            timestep.value_in_unit(unit.picosecond),
            collision_rate.value_in_unit(unit.picosecond**-1),
            masses,
            seed
        )
        integrator_impl = integrator.impl()

        v_0 = sample_velocities(masses * unit.amu, temperature)

        # rescale the box to be approximately the desired box volume already
        rescaler = CentroidRescaler(group_indices)
        initial_volume = compute_box_volume(complex_box)
        initial_center = compute_box_center(complex_box)
        length_scale = ((1 + initial_relative_box_perturbation) * expected_volume_in_md[i] / initial_volume) ** (1. / 3)
        new_coords = rescaler.scale_centroids(coords, initial_center, length_scale)
        new_box = complex_box * length_scale

        baro = custom_ops.MonteCarloBarostat(
            new_coords.shape[0],
            pressure.value_in_unit(unit.bar),
            temperature.value_in_unit(unit.kelvin),
            group_indices,
            barostat_interval,
            u_impls,
            seed
        )

        ctxt = custom_ops.Context(new_coords, v_0, new_box, integrator_impl, u_impls, barostat=baro)
        vols = []
        for move in range(n_moves // barostat_interval):
            ctxt.multiple_steps(np.ones(barostat_interval))
            new_box = ctxt.get_box()
            volume = np.linalg.det(new_box)
            vols.append(volume)
        volume_trajs.append(vols)

    equil_time = len(volume_trajs[0]) // 2  # TODO: don't hard-code this?
    actual_volume_in_md = np.array([np.mean(volume_traj[equil_time:]) for volume_traj in volume_trajs])

    np.testing.assert_allclose(actual=actual_volume_in_md, desired=expected_volume_in_md, rtol=relative_tolerance)
