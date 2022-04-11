# Run a simulation at constant temperature and pressure, at a variety of values of lambda

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from simtk import unit

from timemachine.fe.lambda_schedule import construct_lambda_schedule
from timemachine.lib import LangevinIntegrator
from timemachine.md import enhanced
from timemachine.md.barostat.moves import MonteCarloBarostat
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.ensembles import NPTEnsemble, PotentialEnergyModel
from timemachine.md.states import CoordsVelBox
from timemachine.md.thermostat.moves import UnadjustedLangevinMove
from timemachine.md.thermostat.utils import sample_velocities
from timemachine.md.utils import simulate_npt_traj
from timemachine.testsystems.relative import hif2a_ligand_pair

# simulation parameters
n_lambdas = 40
initial_waterbox_width = 3.0 * unit.nanometer
timestep = 1.5 * unit.femtosecond
collision_rate = 1.0 / unit.picosecond
n_moves = 2000
barostat_interval = 5
seed = 2021

# thermodynamic parameters
temperature = 300 * unit.kelvin
pressure = 1.013 * unit.bar
lambdas = construct_lambda_schedule(n_lambdas)

# build an alchemical ligand in a water box
mol_a, ff = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.ff
unbound_potentials, sys_params, masses, coords, complex_box = enhanced.get_solvent_phase_system(mol_a, ff)

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


def plot_volume_trajs(volume_trajs):
    n_trajs = len(volume_trajs)
    cmap = plt.get_cmap("viridis")
    colors = cmap.colors[:: len(cmap.colors) // n_trajs][:n_trajs]

    for i, volume_traj in enumerate(volume_trajs):
        plt.plot(volume_traj, color=colors[i])
    plt.xlabel("# moves")
    plt.ylabel("volume (nm$^3$)")
    plt.savefig("volume_traj.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_density(volume_trajs, n_water_mols):
    equil_time = len(volume_trajs[0]) // 2  # TODO: don't hard-code this?
    final_volumes = np.array([np.median(volume_traj[equil_time:]) for volume_traj in volume_trajs])

    volume = final_volumes * unit.nanometer ** 3
    water_molecule_mass = 18.01528 * unit.amu
    density = n_water_mols * water_molecule_mass / (volume * unit.AVOGADRO_CONSTANT_NA)

    plt.scatter(lambdas, density.value_in_unit(unit.kilogram / unit.liter))
    plt.xlabel(r"$\lambda$")
    plt.ylabel("density (kg/L)")
    plt.savefig("density_vs_lambda.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)
    n_water_mols = len(group_indices) - 1  # 1 for the ligand

    # loop over lambdas, collecting NPT trajectories
    trajs = []
    volume_trajs = []
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

    # plot volume equilibration, final densities
    plot_volume_trajs(volume_trajs)
    plot_density(volume_trajs, n_water_mols)
