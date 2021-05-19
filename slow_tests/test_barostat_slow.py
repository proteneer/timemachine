# Test that a barostat'd waterbox has an equilibrated density very close to 1.0 kg/L
# Analogous to https://github.com/openmm/openmm/blob/master/tests/TestMonteCarloBarostat.h#L204-L269
# Expected runtime: ~ 10 minutes (10 replicates where each takes < 1 minute)

import numpy as np
from simtk import unit

from testsystems.relative import hif2a_ligand_pair

from md.builders import build_water_system
from md.minimizer import minimize_host_4d

from fe.free_energy import AbsoluteFreeEnergy

from md.ensembles import PotentialEnergyModel, NPTEnsemble
from md.barostat.moves import MonteCarloBarostat
from md.barostat.utils import get_group_indices, simulate_npt_traj
from md.thermostat.utils import sample_velocities

from timemachine.lib import LangevinIntegrator

from functools import partial

# simulation parameters
n_replicates = 10
initial_waterbox_width_in_nm = 3.0
timestep_in_ps = 1.5e-3
collision_rate_in_inv_ps = 1.0
n_moves = 2000
barostat_interval = 5

# thermodynamic parameters
temperature = 300 * unit.kelvin
pressure = 1.0 * unit.atmosphere

# generate an alchemical system of a waterbox + alchemical ligand:
# effectively discard ligands by running in AbsoluteFreeEnergy mode at lambda = 1.0
mol_a, _, core, ff = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core, hif2a_ligand_pair.ff
complex_system, complex_coords, complex_box, complex_top = build_water_system(3.0)

min_complex_coords = minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
afe = AbsoluteFreeEnergy(mol_a, ff)

unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
    ff.get_ordered_params(), complex_system, min_complex_coords
)

# define NPT ensemble
potential_energy_model = PotentialEnergyModel(sys_params, unbound_potentials)
ensemble = NPTEnsemble(potential_energy_model, temperature, pressure)


# define a thermostat
seed = 2021
integrator = LangevinIntegrator(
    temperature.value_in_unit(unit.kelvin),
    timestep_in_ps,
    collision_rate_in_inv_ps,
    masses,
    seed
)
integrator_impl = integrator.impl()

# (getting list of molecules is most easily done by looking at bond table)
harmonic_bond_potential = unbound_potentials[0]
group_indices = get_group_indices(harmonic_bond_potential)


def reduced_potential_fxn(x, box, lam):
    u, du_dx = ensemble.reduced_potential_and_gradient(x, box, lam)
    return u

if __name__ == '__main__':

    trajs = []
    volume_trajs = []

    # run at lambda=1.0, n_replicates times
    lambdas = np.ones(n_replicates)

    for lam in lambdas:
        barostat = MonteCarloBarostat(partial(reduced_potential_fxn, lam=lam), group_indices, max_delta_volume=3.0)
        v_0 = sample_velocities(masses * unit.amu, temperature)
        x_traj, box_traj, extras = simulate_npt_traj(
            ensemble, integrator_impl, barostat,
            coords, complex_box, v_0, lam,
            n_moves=n_moves, barostat_interval=barostat_interval)
        trajs.append(x_traj)
        volume_trajs.append(extras['volume_traj'])

    equil_time = n_moves // 2 # TODO: don't hard-code this?
    final_volumes = np.array([np.mean(volume_traj[equil_time:]) for volume_traj in volume_trajs])

    volume = final_volumes * unit.nanometer ** 3
    n_molecules = complex_top.getNumResidues()
    water_molecule_mass = 18.01528 * unit.amu
    density = n_molecules * water_molecule_mass / (volume * unit.AVOGADRO_CONSTANT_NA)

    density_in_kg_l = density.value_in_unit(unit.kilogram / unit.liter)

    # expected ~ 0.995 - 1.005
    print(density_in_kg_l)

    # absolute tolerance following OpenMM testWater
    np.testing.assert_allclose(density_in_kg_l, 1.0, atol=0.02)
