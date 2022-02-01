# Run a simulation at constant temperature and pressure, at a variety of values of lambda

import numpy as np
from simtk import unit

from testsystems.relative import hif2a_ligand_pair

from timemachine.md.builders import build_water_system
from timemachine.md.minimizer import minimize_host_4d

from timemachine.md.ensembles import PotentialEnergyModel, NPTEnsemble

from timemachine.md.thermostat.utils import sample_velocities

from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.barostat.moves import MonteCarloBarostat

from timemachine.md.thermostat.moves import UnadjustedLangevinMove

from timemachine.md.states import CoordsVelBox
from timemachine.md.utils import simulate_npt_traj

from fe.free_energy import AbsoluteFreeEnergy, construct_lambda_schedule

from timemachine.lib import LangevinIntegrator
from functools import partial

import matplotlib.pyplot as plt

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

# build a pair of alchemical ligands in a water box
mol_a, mol_b, core, ff = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core, hif2a_ligand_pair.ff
complex_system, complex_coords, complex_box, complex_top = build_water_system(
    initial_waterbox_width.value_in_unit(unit.nanometer)
)

min_complex_coords = minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
afe = AbsoluteFreeEnergy(mol_a, ff)

unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
    ff.get_ordered_params(), complex_system, min_complex_coords
)

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


def plot_density(volume_trajs):
    equil_time = len(volume_trajs[0]) // 2  # TODO: don't hard-code this?
    final_volumes = np.array([np.median(volume_traj[equil_time:]) for volume_traj in volume_trajs])

    volume = final_volumes * unit.nanometer ** 3
    n_molecules = complex_top.getNumResidues()
    water_molecule_mass = 18.01528 * unit.amu
    density = n_molecules * water_molecule_mass / (volume * unit.AVOGADRO_CONSTANT_NA)

    plt.scatter(lambdas, density.value_in_unit(unit.kilogram / unit.liter))
    plt.xlabel("$\lambda$")
    plt.ylabel("density (kg/L)")
    plt.savefig("density_vs_lambda.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)

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
    plot_density(volume_trajs)
