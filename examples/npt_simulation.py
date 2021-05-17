# Run a simulation at constant temperature and pressure

import numpy as np
from simtk import unit
from tqdm import tqdm

from testsystems.relative import hif2a_ligand_pair

from md.builders import build_water_system
from md.minimizer import minimize_host_4d

from fe.topology import SingleTopology
from fe.free_energy import RelativeFreeEnergy

from barostat.ensembles import PotentialEnergyModel, NPTEnsemble
from barostat.moves import MonteCarloBarostat, CoordsAndBox
from barostat.utils import get_group_indices, compute_box_volume

from timemachine.lib import custom_ops, LangevinIntegrator
from timemachine.constants import BOLTZ

if __name__ == '__main__':

    # thermodynamic parameters
    temperature = 300 * unit.kelvin
    pressure = 1.0 * unit.atmosphere

    # build a pair of alchemical ligands in a water box
    mol_a, mol_b, core, ff = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core, hif2a_ligand_pair.ff
    complex_system, complex_coords, complex_box, complex_top = build_water_system(4.0)

    min_complex_coords = minimize_host_4d([mol_a, mol_b], complex_system, complex_coords, ff, complex_box)

    single_topology = SingleTopology(mol_a, mol_b, core, ff)
    rfe = RelativeFreeEnergy(single_topology)

    unbound_potentials, sys_params, masses, coords = rfe.prepare_host_edge(
        ff.get_ordered_params(), complex_system, min_complex_coords
    )

    # define NPT ensemble
    potential_energy_model = PotentialEnergyModel(sys_params, unbound_potentials)
    ensemble = NPTEnsemble(potential_energy_model, temperature, pressure)

    lam = 0.63


    def reduced_potential_fxn(x, box):
        u, du_dx = ensemble.reduced_potential_and_gradient(x, box, lam)
        return u


    # define a barostat

    # (getting list of molecules is most easily done by looking at bond table)
    harmonic_bond_potential = unbound_potentials[0]
    group_indices = get_group_indices(harmonic_bond_potential)

    barostat = MonteCarloBarostat(reduced_potential_fxn, group_indices, max_delta_volume=0.1)

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


    def sample_velocities():
        """ TODO: move this into integrator or something? """
        v_unscaled = np.random.randn(len(masses), 3)

        # intended to be consistent with timemachine.integrator:langevin_coefficients
        temperature = ensemble.temperature.value_in_unit(unit.kelvin)
        sigma = np.sqrt(BOLTZ * temperature) * np.sqrt(1 / masses)

        return v_unscaled * np.expand_dims(sigma, axis=1)

        # return (sigma * v_unscaled.T).T


    def run_thermostatted_md(x: CoordsAndBox, n_steps=5) -> CoordsAndBox:

        # TODO: is there a way to set context coords, box, velocities without initializing a fresh Context?
        # TODO: is there a way to get velocities at the end?

        ctxt = custom_ops.Context(
            x.coords,
            sample_velocities(),
            x.box,
            integrator_impl,
            potential_energy_model.all_impls
        )

        # lambda schedule, du_dl_interval, x interval
        du_dls, xs = ctxt.multiple_steps(lam * np.ones(n_steps), 0, n_steps - 1)
        return CoordsAndBox(xs[-1], x.box)


    def simulate_npt_traj(coords, box, n_moves=1000):
        # alternate between thermostat moves and barostat moves
        traj = [CoordsAndBox(coords, box)]
        volume_traj = [compute_box_volume(traj[0].box)]

        trange = tqdm(range(1000))

        from time import time

        for _ in trange:
            t0 = time()
            after_nvt = run_thermostatted_md(traj[-1])
            t1 = time()
            after_npt = barostat.move(after_nvt)
            t2 = time()

            traj.append(after_npt)
            volume_traj.append(compute_box_volume(after_npt.box))

            trange.set_postfix(volume=f'{volume_traj[-1]:.3f}',
                               acceptance_fraction=f'{(barostat.n_accepted / barostat.n_proposed):.3f}',
                               md_proposal_time=f'{(t1 - t0):.3f}s',
                               barostat_proposal_time=f'{(t2 - t1):.3f}s',
                               )

        traj = np.array(traj)
        volume_traj = np.array(volume_traj)
        return traj, volume_traj


    import matplotlib.pyplot as plt

    initial_box_scales = [1.0, 1.05, 1.1, 1.15]
    trajs = []
    volume_trajs = []
    for scale in initial_box_scales:
        traj, volume_traj = simulate_npt_traj(coords, complex_box * scale, n_moves=1000)

        trajs.append(traj)
        volume_trajs.append(volume_traj)

    for volume_traj in volume_trajs:
        plt.plot(volume_traj)
    plt.xlabel('# moves')
    plt.ylabel('volume')
    plt.savefig('volume_traj.png', dpi=300, bbox_inches='tight')
    plt.close()
