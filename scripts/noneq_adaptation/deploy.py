from md.thermostat.utils import sample_velocities
from md.thermostat.moves import UnadjustedLangevinMove

from md.states import CoordsVelBox
from fe.free_energy import construct_lambda_schedule

from simtk import unit
import numpy as np
np.random.seed(0)

from timemachine.lib import custom_ops
from tqdm import tqdm

from functools import partial

import os

# from same script folder...
from testsystem import (
    temperature, coords, masses, complex_box,
    integrator_impl, ensemble, potential_energy_model,
)
from adapt_noneq import optimized_lam_trajs_path, sample_at_equilibrium, n_md_steps_per_increment

from pymbar import EXP


def interpolate_lambda_schedule(lambda_schedule, num_md_steps):
    """given a lambda schedule, with n windows, turn it into a lambda
    schedule with num_md_steps windows by interpolation"""
    n = len(lambda_schedule)
    xp, fp = np.linspace(0, 1, n), np.array(lambda_schedule)
    md_steps = np.linspace(0, 1, num_md_steps)
    interpolated_schedule = np.interp(md_steps, xp, fp)

    return interpolated_schedule


def noneq_du_dl(x: CoordsVelBox, lambda_schedule: np.array) -> np.array:
    """Compute du_dl per step along nonequilibrium trajectory"""
    ctxt = custom_ops.Context(x.coords, x.velocities, x.box, integrator_impl, potential_energy_model.all_impls)

    # arguments: lambda_schedule, du_dl_interval, x_interval
    du_dl_traj, _ = ctxt.multiple_steps(lambda_schedule, 1, 0)

    return du_dl_traj


if __name__ == '__main__':

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
