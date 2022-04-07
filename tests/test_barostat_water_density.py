# Test that a barostat'd waterbox has an equilibrated density very close to 1.0 kg/L
# Analogous to https://github.com/openmm/openmm/blob/master/tests/TestMonteCarloBarostat.h#L204-L269
# Expected runtime: ~ 10 minutes (10 replicates where each takes < 1 minute)

from functools import partial

import numpy as np
from simtk import unit

from timemachine.lib import LangevinIntegrator
from timemachine.md.barostat.moves import MonteCarloBarostat
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.enhanced import get_solvent_phase_system
from timemachine.md.ensembles import NPTEnsemble, PotentialEnergyModel
from timemachine.md.states import CoordsVelBox
from timemachine.md.thermostat.moves import UnadjustedLangevinMove
from timemachine.md.thermostat.utils import sample_velocities
from timemachine.md.utils import simulate_npt_traj
from timemachine.testsystems.relative import hif2a_ligand_pair

if __name__ == "__main__":
    # simulation parameters
    n_replicates = 10
    initial_waterbox_width = 3.0 * unit.nanometer
    timestep = 1.5 * unit.femtosecond
    collision_rate = 1.0 / unit.picosecond
    n_moves = 2000
    barostat_interval = 5
    seed = 2021

    # thermodynamic parameters
    temperature = 300 * unit.kelvin
    pressure = 1.013 * unit.bar

    # generate an alchemical system of a waterbox + alchemical ligand:
    # effectively discard ligands by running in AbsoluteFreeEnergy mode at lambda = 1.0
    mol_a, ff = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.ff
    unbound_potentials, sys_params, masses, coords, complex_box = get_solvent_phase_system(mol_a, ff, margin=0.0)

    # define NPT ensemble
    potential_energy_model = PotentialEnergyModel(sys_params, unbound_potentials)
    ensemble = NPTEnsemble(potential_energy_model, temperature, pressure)

    # define a thermostat
    integrator = LangevinIntegrator(
        temperature.value_in_unit(unit.kelvin),
        timestep.value_in_unit(unit.picosecond),
        collision_rate.value_in_unit(unit.picosecond ** -1),
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    def reduced_potential_fxn(x, box, lam):
        u, du_dx = ensemble.reduced_potential_and_gradient(x, box, lam)
        return u

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)

    trajs = []
    volume_trajs = []

    # run at lambda=1.0, n_replicates times
    lambdas = np.ones(n_replicates)

    for lam in lambdas:
        thermostat = UnadjustedLangevinMove(
            integrator_impl, potential_energy_model.all_impls, lam, n_steps=barostat_interval
        )
        barostat = MonteCarloBarostat(partial(reduced_potential_fxn, lam=lam), group_indices, max_delta_volume=3.0)

        v_0 = sample_velocities(masses * unit.amu, temperature)
        initial_state = CoordsVelBox(coords, v_0, complex_box)

        traj, extras = simulate_npt_traj(thermostat, barostat, initial_state, n_moves=n_moves)

        trajs.append(traj)
        volume_trajs.append(extras["volume_traj"])

    equil_time = n_moves // 2  # TODO: don't hard-code this?
    final_volumes = np.array([np.mean(volume_traj[equil_time:]) for volume_traj in volume_trajs])

    volume = final_volumes * unit.nanometer ** 3
    water_molecule_mass = 18.01528 * unit.amu
    n_water_mols = len(group_indices) - 1  # 1 for the ligand
    density = n_water_mols * water_molecule_mass / (volume * unit.AVOGADRO_CONSTANT_NA)

    density_in_kg_l = density.value_in_unit(unit.kilogram / unit.liter)

    # expected ~ 0.995 - 1.005
    print(density_in_kg_l)

    # absolute tolerance following OpenMM testWater
    np.testing.assert_allclose(density_in_kg_l, 1.0, atol=0.02)
