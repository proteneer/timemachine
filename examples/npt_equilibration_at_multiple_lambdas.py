# Run a simulation at constant temperature and pressure

import numpy as np
from simtk import unit
from tqdm import tqdm

from testsystems.relative import hif2a_ligand_pair

from md.builders import build_water_system
from md.minimizer import minimize_host_4d

from fe.topology import SingleTopology
from fe.free_energy import AbsoluteFreeEnergy

from barostat.ensembles import PotentialEnergyModel, NPTEnsemble
from barostat.moves import MonteCarloBarostat, CoordsAndBox
from barostat.utils import get_group_indices, compute_box_volume

from timemachine.lib import custom_ops, LangevinIntegrator
from timemachine.constants import BOLTZ

from fe.free_energy import construct_lambda_schedule

import matplotlib.pyplot as plt
from functools import partial

# thermodynamic parameters
temperature = 300 * unit.kelvin
pressure = 1.0 * unit.atmosphere
lambdas = construct_lambda_schedule(40)

# build a pair of alchemical ligands in a water box
mol_a, mol_b, core, ff = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core, hif2a_ligand_pair.ff
complex_system, complex_coords, complex_box, complex_top = build_water_system(3.0)

min_complex_coords = minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
afe = AbsoluteFreeEnergy(mol_a, ff)

unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
    ff.get_ordered_params(), complex_system, min_complex_coords
)

# define NPT ensemble
potential_energy_model = PotentialEnergyModel(sys_params, unbound_potentials)
ensemble = NPTEnsemble(potential_energy_model, temperature, pressure)


def sample_velocities():
    """ TODO: move this into integrator or something? """
    v_unscaled = np.random.randn(len(masses), 3)

    # intended to be consistent with timemachine.integrator:langevin_coefficients
    temperature = ensemble.temperature.value_in_unit(unit.kelvin)
    sigma = np.sqrt(BOLTZ * temperature) * np.sqrt(1 / masses)

    return v_unscaled * np.expand_dims(sigma, axis=1)


# define a thermostat
seed = 2021
integrator = LangevinIntegrator(
    temperature.value_in_unit(unit.kelvin),
    1.5e-3,
    1.0,
    masses,
    seed
)
integrator_impl = integrator.impl()

# define a barostat

# (getting list of molecules is most easily done by looking at bond table)
harmonic_bond_potential = unbound_potentials[0]
group_indices = get_group_indices(harmonic_bond_potential)


def run_thermostatted_md(x: CoordsAndBox, v: np.array, lam: float, n_steps=5) -> CoordsAndBox:
    # TODO: is there a way to set context coords, box, velocities without initializing a fresh Context?

    ctxt = custom_ops.Context(
        x.coords,
        v,
        x.box,
        integrator_impl,
        potential_energy_model.all_impls
    )

    # lambda schedule, du_dl_interval, x interval
    du_dls, xs = ctxt.multiple_steps(lam * np.ones(n_steps), 0, n_steps - 1)
    x_t = ctxt.get_x_t()
    v_t = ctxt.get_v_t()
    return CoordsAndBox(x_t, x.box), v_t


def reduced_potential_fxn(x, box, lam):
    u, du_dx = ensemble.reduced_potential_and_gradient(x, box, lam)
    return u


def simulate_npt_traj(coords, box, lam, n_moves=1000):
    barostat = MonteCarloBarostat(partial(reduced_potential_fxn, lam=lam), group_indices, max_delta_volume=3.0)

    barostat.reset()

    # alternate between thermostat moves and barostat moves
    traj = [CoordsAndBox(coords, box)]
    volume_traj = [compute_box_volume(traj[0].box)]

    trange = tqdm(range(n_moves))

    from time import time

    v_t = sample_velocities()

    for _ in trange:
        t0 = time()
        after_nvt, v_t = run_thermostatted_md(traj[-1], v_t, lam)
        t1 = time()
        after_npt = barostat.move(after_nvt)
        t2 = time()

        traj.append(after_npt)
        volume_traj.append(compute_box_volume(after_npt.box))

        trange.set_postfix(volume=f'{volume_traj[-1]:.3f}',
                           acceptance_fraction=f'{barostat.acceptance_fraction:.3f}',
                           md_proposal_time=f'{(t1 - t0):.3f}s',
                           barostat_proposal_time=f'{(t2 - t1):.3f}s',
                           proposal_scale=f'{barostat.max_delta_volume:.3f}',
                           )

    traj = np.array(traj)
    volume_traj = np.array(volume_traj)
    return traj, volume_traj


if __name__ == '__main__':

    trajs = []
    volume_trajs = []
    for lam in lambdas:
        traj, volume_traj = simulate_npt_traj(coords, complex_box, lam, n_moves=2000)
        trajs.append(traj)
        volume_trajs.append(volume_traj)

    cmap = plt.get_cmap('viridis')
    colors = cmap.colors[::len(cmap.colors) // len(lambdas)][:len(lambdas)]

    for i, volume_traj in enumerate(volume_trajs):
        plt.plot(volume_traj, color=colors[i])
    plt.xlabel('# moves')
    plt.ylabel('volume (nm$^3$)')
    plt.savefig('volume_traj.png', dpi=300, bbox_inches='tight')
    plt.close()


    final_volumes = [np.median(volume_traj[-1000:]) for volume_traj in volume_trajs]
    plt.scatter(lambdas, final_volumes)
    plt.xlabel('$\lambda$')
    plt.ylabel('volume (nm$^3$)')
    plt.savefig('volume_vs_lambda.png', dpi=300, bbox_inches='tight')
    plt.close()

    for i, volume_traj in enumerate(volume_trajs):
        plt.hist(volume_traj[-1000:], c=colors[i], alpha=0.5, density=True)
    plt.xlabel('volume (nm$^3$)')
    plt.ylabel('probability density')
    plt.savefig('volume_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
