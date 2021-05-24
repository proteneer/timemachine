import numpy as np
from simtk import unit

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

from timemachine.lib import LangevinIntegrator

from functools import partial

from timemachine.constants import BOLTZ, ENERGY_UNIT, DISTANCE_UNIT



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
    mol_a, _, core, ff = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core, hif2a_ligand_pair.ff
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

    trajs = []
    volume_trajs = []

    potential_energy_model = PotentialEnergyModel(sys_params, unbound_potentials)
    lam = 1.0

    relative_tolerance = 1e-2
    initial_relative_box_perturbation = 2 * relative_tolerance

    n_molecules = complex_top.getNumResidues()

    # expected volume
    md_pressure_unit = ENERGY_UNIT / DISTANCE_UNIT ** 3
    pressure_in_md = (pressure * unit.AVOGADRO_CONSTANT_NA).value_in_unit(md_pressure_unit)
    expected_volume_in_md = (n_molecules + 1) * BOLTZ * temperatures.value_in_unit(unit.kelvin) / pressure_in_md


    for i, temperature in enumerate(temperatures):
        # define NPT ensemble
        ensemble = NPTEnsemble(potential_energy_model, temperature, pressure)

        # define a thermostat
        integrator = LangevinIntegrator(
            temperature.value_in_unit(unit.kelvin),
            timestep.value_in_unit(unit.picosecond),
            collision_rate.value_in_unit(unit.picosecond**-1),
            masses,
            seed
        )
        integrator_impl = integrator.impl()

        def reduced_potential_fxn(x, box, lam):
            u, du_dx = ensemble.reduced_potential_and_gradient(x, box, lam)
            return u

        thermostat = UnadjustedLangevinMove(integrator_impl, potential_energy_model.all_impls, lam, n_steps=barostat_interval)
        barostat = MonteCarloBarostat(partial(reduced_potential_fxn, lam=lam), group_indices)

        v_0 = sample_velocities(masses * unit.amu, temperature)

        # rescale the box to be approximately the desired box volume already
        rescaler = CentroidRescaler(group_indices)
        initial_volume = compute_box_volume(complex_box)
        initial_center = compute_box_center(complex_box)
        length_scale = ((1 + initial_relative_box_perturbation) * expected_volume_in_md[i] / initial_volume) ** (1. / 3)
        new_coords = rescaler.rescale(coords, initial_center, length_scale)
        new_box = complex_box * length_scale

        initial_state = CoordsVelBox(new_coords, v_0, new_box)

        traj, extras = simulate_npt_traj(thermostat, barostat, initial_state, n_moves=n_moves)

        trajs.append(traj)
        volume_trajs.append(extras['volume_traj'])

    equil_time = n_moves // 2  # TODO: don't hard-code this?
    actual_volume_in_md = np.array([np.mean(volume_traj[equil_time:]) for volume_traj in volume_trajs])

    np.testing.assert_allclose(actual=actual_volume_in_md, desired=expected_volume_in_md, rtol=relative_tolerance)
