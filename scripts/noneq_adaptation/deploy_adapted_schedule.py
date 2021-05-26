from md.thermostat.utils import sample_velocities
from md.thermostat.moves import UnadjustedLangevinMove
from md.states import CoordsVelBox
import os
from simtk import unit
import numpy as np
from scipy.optimize import root_scalar

from tqdm import tqdm
from typing import List
from pymbar import EXP

from md.states import CoordsVelBox
from fe.free_energy import construct_lambda_schedule

from simtk import unit
import numpy as np
from scipy.optimize import root_scalar

from timemachine.lib import LangevinIntegrator, custom_ops
from tqdm import tqdm

from typing import List
from functools import partial

from pymbar import EXP

import os

# from same script folder...
from testsystem import (
    temperature, coords, masses, complex_box,
    integrator_impl, ensemble, potential_energy_model,
)
from adapt_noneq import optimized_lam_traj_path, sample_at_equilibrium


def interpolate_lambda_schedule(lambda_schedule, num_md_steps):
    """given a lambda schedule, with n windows, turn it into a lambda
    schedule with num_md_steps windows by interpolation"""
    n = len(lambda_schedule)
    xp, fp = np.linspace(0, 1, n), np.array(lambda_schedule)
    md_steps = np.linspace(0, 1, num_md_steps)
    interpolated_schedule = np.interp(md_steps, xp, fp)

    return interpolated_schedule


def noneq_move(x: CoordsVelBox, lambda_schedule: np.array) -> CoordsVelBox:
    """Run a nonequilibrium trajectory, storing final state"""
    ctxt = custom_ops.Context(x.coords, x.velocities, x.box, integrator_impl, potential_energy_model.all_impls)

    # arguments: lambda_schedule, du_dl_interval, x_interval
    _, _ = ctxt.multiple_steps(lambda_schedule, 0, 0)

    return CoordsVelBox(ctxt.get_x_t(), ctxt.get_v_t(), x.box.copy())


def noneq_du_dl(x: CoordsVelBox, lambda_schedule: np.array) -> np.array:
    """Compute du_dl per step along nonequilibrium trajectory"""
    ctxt = custom_ops.Context(x.coords, x.velocities, x.box, integrator_impl, potential_energy_model.all_impls)

    # arguments: lambda_schedule, du_dl_interval, x_interval
    du_dl_traj, _ = ctxt.multiple_steps(lambda_schedule, 1, 0)

    return du_dl_traj


if __name__ == '__main__':

    lam_traj = np.load(optimized_lam_traj_path)

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

    print('switching from lam=0 -> lam=1 to initialize lam=1 equilibrium sampling...')
    approx_equilibrated_1 = noneq_move(samples_0[-1], interpolate_lambda_schedule(lam_traj, n_equil_steps))
    thermostat_1 = UnadjustedLangevinMove(
        integrator_impl, potential_energy_model.all_impls,
        lam=1.0, n_steps=n_equil_steps
    )
    equilibrated_1 = thermostat_1.move(approx_equilibrated_1)

    print(f'collecting {n_samples} samples from lam=1...')
    samples_1 = sample_at_equilibrium(equilibrated_1, lam=1.0)

    # construct interpolated version of the adapted schedule, rather than doing cycles of
    #   lambda increment <-> MD propagation
    total_md_steps = len(lam_traj) * n_md_steps_per_increment
    md_steps = np.linspace(0, 1, total_md_steps)
    xp, fp = np.linspace(0, 1, len(lam_traj)), np.array(lam_traj)
    optimized_schedule = np.interp(md_steps, xp, fp)

    default_schedule = construct_lambda_schedule(total_md_steps)



    # now run timemachine noneq moves, computing work via du_dl increments,
    # default vs. optimized protocol, forward vs. reverse
    protocol_names = ['default', 'optimized']
    directions = ['forward', 'reverse']
    forward_schedules = dict(default=default_schedule, optimized=optimized_schedule)
    reverse_schedules = dict(default=default_schedule[::-1], optimized=optimized_schedule[::-1])
    schedules = dict(forward=forward_schedules, reverse=reverse_schedules)
    initial_ensembles = dict(forward=samples_0, reverse=samples_1)

    # save lambda schedule before doing anything
    schedule_path = os.path.join(os.path.dirname(__file__), 'results/schedules.npz')
    print(f'saving results to {schedule_path}')
    np.savez(schedule_path, **forward_schedules)

    du_dl_trajs, works = dict(), dict()
    for name in protocol_names:
        for direction in directions:
            schedule = schedules[direction][name]

            key = f'{name}_{direction}'
            noneq_mover = partial(noneq_du_dl, lambda_schedule=schedule)
            du_dl_trajs[key] = np.array([noneq_mover(x) for x in tqdm(initial_ensembles[direction])])
            works[key] = ensemble.reduce(np.trapz(du_dl_trajs, schedule, axis=1))

    du_dl_path = os.path.join(os.path.dirname(__file__), 'results/du_dl_trajs.npz')
    work_path = os.path.join(os.path.dirname(__file__), 'results/works.npz')

    print(f'saving results to {du_dl_path}')
    np.savez(du_dl_path, **du_dl_trajs)

    print(f'saving results to {work_path}')
    np.savez(work_path, **works)

    for key in works:
        print(f'stddev(w) in {key} condition: {np.std(works[key]):.3f} kBT')
