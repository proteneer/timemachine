from md.thermostat.utils import sample_velocities
from md.thermostat.moves import UnadjustedLangevinMove


from md.builders import build_water_system
from md.minimizer import minimize_host_4d
from md.ensembles import PotentialEnergyModel, NVTEnsemble
from timemachine.lib import LangevinIntegrator
from fe.free_energy import AbsoluteFreeEnergy
from md.states import CoordsVelBox

from testsystems.relative import hif2a_ligand_pair

from simtk import unit
import numpy as np

temperature = 300 * unit.kelvin
initial_waterbox_width = 3.0 * unit.nanometer
timestep = 1.5 * unit.femtosecond
collision_rate = 1.0 / unit.picosecond
seed = 2021

mol_a, _, core, ff = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core, hif2a_ligand_pair.ff
complex_system, complex_coords, complex_box, complex_top = build_water_system(
    initial_waterbox_width.value_in_unit(unit.nanometer))

min_complex_coords = minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
afe = AbsoluteFreeEnergy(mol_a, ff)

unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
    ff.get_ordered_params(), complex_system, min_complex_coords
)

potential_energy_model = PotentialEnergyModel(sys_params, unbound_potentials, precision=np.float32)
integrator = LangevinIntegrator(
    temperature.value_in_unit(unit.kelvin),
    timestep.value_in_unit(unit.picosecond),
    collision_rate.value_in_unit(unit.picosecond ** -1),
    masses,
    seed
)
integrator_impl = integrator.impl()
bound_impls = potential_energy_model.all_impls

ensemble = NVTEnsemble(potential_energy_model, temperature)

def noneq_du_dl(x: CoordsVelBox, lambda_schedule: np.array) -> np.array:
    """Compute du_dl per step along nonequilibrium trajectory"""
    ctxt = custom_ops.Context(x.coords, x.velocities, x.box, integrator_impl, potential_energy_model.all_impls)

    # arguments: lambda_schedule, du_dl_interval, x_interval
    du_dl_traj, _ = ctxt.multiple_steps(lambda_schedule, 1, 0)

    return du_dl_traj




if __name__ == '__main__':

    # paths where we'll later save results
    work_increments_path = os.path.join(os.path.dirname(__file__), 'results/works_via_potential_increments.npy')
    optimized_lam_trajs_path = os.path.join(os.path.dirname(__file__), 'results/optimized_lam_trajs.npz')

    # equilibrium options
    n_equil_steps = 10000
    n_samples = 100

    # adaptation options
    n_md_steps_per_increment = 100  # number of MD steps run at fixed lambda, between lambda increments

    # load results from adaptive lambda spacing
    lambda_spacing_results = np.load(optimized_lam_trajs_path)
    incremental_stddev_thresholds = lambda_spacing_results['incremental_stddev_thresholds']

    # generate end-state samples
    n_equil_steps = 10000
    n_samples = 100

    v_0 = sample_velocities(masses * unit.amu, temperature)
    initial_state = CoordsVelBox(coords, v_0, complex_box)
    print('equilibrating...')
    thermostat_0 = UnadjustedLangevinMove(
        integrator_impl, potential_energy_model.all_impls,
        lam=0.0, n_steps=n_equil_steps
    )
    equilibrated_0 = thermostat_0.move(initial_state)

    print(f'collecting {n_samples} samples from lam=0...')
    samples_0 = sample_at_equilibrium(equilibrated_0, lam=0.0, n_samples=n_samples)

    # will run at the same total number of MD steps as each corresponding protocol optimization
    keys = list(lambda_spacing_results.keys())
    keys.remove('incremental_stddev_thresholds')
    available_thresholds = [incremental_stddev_thresholds[int(i)] for i in keys]
    print('running noneq MD at protocols computed with the following incremental_stddev_thresholds')
    print(available_thresholds)

    # construct interpolated versions of the adapted schedule, rather than doing cycles of
    #   lambda increment <-> MD propagation

    # dicts, for later dumping into .npz files since they'll have different shapes depending on
    #   total_md_steps
    du_dl_trajs_default = dict()
    du_dl_trajs_optimized = dict()

    # lists are fine for these, since we can stack them into flat arrays...
    works_default = []
    works_optimized = []

    total_md_step_range = []

    for key in keys:
        lam_traj = lambda_spacing_results[key]
        ind = int(key)
        threshold = incremental_stddev_thresholds[ind]
        total_md_steps = len(lam_traj) * n_md_steps_per_increment
        total_md_step_range.append(total_md_steps)

        print(f'using the protocol optimized with an incremental stddev threshold of {threshold:.3f}')
        print(f'running default and optimized protocols at # MD steps = {total_md_steps}...')
        # TODO: reduce the 2x code duplications here...
        default_schedule = construct_lambda_schedule(total_md_steps)
        optimized_schedule = interpolate_lambda_schedule(lam_traj, total_md_steps)

        default_noneq_mover = partial(noneq_du_dl, lambda_schedule=default_schedule)
        optimized_noneq_mover = partial(noneq_du_dl, lambda_schedule=optimized_schedule)

        du_dl_default = ensemble.reduce(np.array([default_noneq_mover(x) for x in tqdm(samples_0)]))
        du_dl_optimized = ensemble.reduce(np.array([optimized_noneq_mover(x) for x in tqdm(samples_0)]))

        key = str(total_md_steps)
        du_dl_trajs_default[key] = du_dl_default
        du_dl_trajs_optimized[key] = du_dl_optimized

        works_default.append(np.trapz(du_dl_default, default_schedule, axis=1))
        works_optimized.append(np.trapz(du_dl_optimized, optimized_schedule, axis=1))

        def describe(works):
            print(f'\tmean(w_f): {np.mean(works):.3f} kBT')
            print(f'\tstddev(w_f): {np.std(works):.3f} kBT')
            print(f'\tmin(w_f): {np.min(works):.3f} kBT')
            print(f'\tmax(w_f): {np.max(works):.3f} kBT')
            print(f'\tEXP(w_f): {EXP(works)[0]:.3f} kBT')

        print(f'default:')
        describe(works_default[-1])
        print('optimized:')
        describe(works_optimized[-1])

        # overwrite results at each iteration
        results = dict(
            total_md_step_range=total_md_step_range,
            works_default=np.array(works_default),
            works_optimized=np.array(works_optimized),
        )
        works_path = os.path.join(os.path.dirname(__file__), 'results/works_via_du_dl.npz')

        print(f'saving results to {works_path}')
        np.savez(works_path, **results)

        du_dl_trajs_default_path = os.path.join(os.path.dirname(__file__), 'results/du_dl_trajs_default.npz')
        du_dl_trajs_optimized_path = os.path.join(os.path.dirname(__file__), 'results/du_dl_trajs_optimized.npz')

        print(f'saving results to {du_dl_trajs_default_path}')
        np.savez(du_dl_trajs_default_path, **du_dl_trajs_default)

        print(f'saving results to {du_dl_trajs_optimized_path}')
        np.savez(du_dl_trajs_optimized_path, **du_dl_trajs_optimized)


    # collect endstate samples
    v_0 = sample_velocities(masses * unit.amu, temperature)
    initial_state = CoordsVelBox(coords, v_0, complex_box)
    print('equilibrating...')
    thermostat = UnadjustedLangevinMove(
        integrator_impl, potential_energy_model.all_impls, lam=0.0, n_steps=n_equil_steps)
    equilibrated = thermostat.move(initial_state)

    print(f'collecting {n_samples} samples from lam=0...')
    samples_0 = sample_at_equilibrium(equilibrated, lam=0.0, n_samples=n_samples)

    incremental_stddev_thresholds = np.logspace(0, -1, 5)
    results = dict()
    results['incremental_stddev_thresholds'] = incremental_stddev_thresholds

    for i, incremental_stddev_threshold in enumerate(incremental_stddev_thresholds):
        # run switching with adaptive lambda steps
        print(f'running adaptive noneq switching with {n_samples} trajectories and a threshold of {incremental_stddev_threshold}')
        sample_traj, lam_traj = adaptive_noneq(
            samples_0,
            n_md_steps_per_increment=n_md_steps_per_increment,
            incremental_stddev_threshold=incremental_stddev_threshold,
        )
        results[str(i)] = lam_traj

        print(f'saving optimized lambda schedules to {optimized_lam_trajs_path}')
        np.savez(optimized_lam_trajs_path, **results)
