from md.builders import build_water_system
from md.minimizer import minimize_host_4d
from md.ensembles import PotentialEnergyModel, NVTEnsemble
from md.thermostat.utils import sample_velocities
from md.thermostat.moves import UnadjustedLangevinMove
from md.states import CoordsVelBox
from fe.free_energy import AbsoluteFreeEnergy, construct_lambda_schedule

from testsystems.relative import hif2a_ligand_pair

from simtk import unit
import numpy as np
from scipy.optimize import root_scalar

from timemachine.lib import LangevinIntegrator, custom_ops
from tqdm import tqdm

from typing import List
from functools import partial

from pymbar import EXP

import os

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


def u(state: CoordsVelBox, lam: float) -> float:
    """compute reduced potential"""
    energy, gradient = ensemble.reduced_potential_and_gradient(state.coords, state.box, lam)
    return energy


def u_vec(states: List[CoordsVelBox], lam: float) -> np.array:
    """compute reduced potential on list of states"""
    return np.array([u(state, lam) for state in states])


def sample_at_equilibrium(initial_state: CoordsVelBox, lam: float = 0.0, thinning: int = 1000, n_samples: int = 100) -> \
        List[CoordsVelBox]:
    """run MD """

    thermostat = UnadjustedLangevinMove(integrator_impl, potential_energy_model.all_impls, lam, n_steps=thinning)

    samples = [initial_state]
    for _ in tqdm(range(n_samples)):
        samples.append(thermostat.move(samples[-1]))

    return samples


def propagate(states: List[CoordsVelBox], lam: float = 0.0, n_steps: float = 500) -> List[CoordsVelBox]:
    thermostat = UnadjustedLangevinMove(integrator_impl, potential_energy_model.all_impls, lam, n_steps=n_steps)

    print(f'propagating {len(states)} systems by {n_steps * timestep.value_in_unit(unit.picosecond)}ps each...')
    updated_states = []
    for state in tqdm(states):  # TODO: loop could be paralllelized (e.g. on CUDAPoolClient)
        updated_states.append(thermostat.move(state))

    return updated_states


def find_next_increment(
        samples: List[CoordsVelBox], lam_initial: float,
        max_increment_size: float = 0.1, incremental_stddev_threshold: float = 0.1, xtol: float = 1e-4
) -> float:
    u_s = u_vec(samples, lam_initial)

    def work_increment_stddev(lam_increment: float) -> float:
        """stddev(u(samples, lam + lam_increment) - u(samples, lam))"""
        lam = lam_initial + lam_increment
        u_trial = u_vec(samples, lam)
        return np.std(u_trial - u_s)

    def f(lam_increment: float) -> float:
        """find the zero of this function to get a lambda increment
        that controls the stddev of work accumulated this step"""
        return work_increment_stddev(lam_increment) - incremental_stddev_threshold

    # try-except to catch rootfinding ValueError: f(a) and f(b) must have different signs
    #   which occurs when jumping all the way to lam=1.0 is still less than threshold
    try:
        result = root_scalar(f, bracket=(0, max_increment_size), xtol=xtol)
        lam_increment = result.root
    except ValueError as e:
        print(f'root finding error: {e}')
        lam_increment = max_increment_size

    return lam_increment


def noneq_du_dl(x: CoordsVelBox, lambda_schedule: np.array) -> np.array:
    """Compute du_dl per step along nonequilibrium trajectory"""
    ctxt = custom_ops.Context(x.coords, x.velocities, x.box, integrator_impl, potential_energy_model.all_impls)

    # arguments: lambda_schedule, du_dl_interval, x_interval
    du_dl_traj, _ = ctxt.multiple_steps(lambda_schedule, 1, 0)

    return du_dl_traj


def adaptive_noneq(samples_0: List[CoordsVelBox], n_md_steps_per_increment=100, incremental_stddev_threshold=0.5):
    """Generate lam=0 -> lam=1 trajectories by a scheme that makes adaptively sized lambda increments.

        Alternates between the following two steps:
        * Select the next lambda increment by finding the root of
            f(increment) = stddev(u(samples, lam + increment) - u(samples, lam)) - incremental_stddev_threshold
        * Propagate all samples for n_md_steps_per_increment
            (n_md_steps_per_increment can be << equilibration time)

    Notes
    -----
    * TODO: be able to run this also in reverse -- currently hard-codes lam=0 -> lam=1

    References
    ----------
    * Based on description of an adaptive SMC approach that appeared in
        Section 2.4.2. of https://arxiv.org/abs/1612.06468,
        which references Del Moral et al., 2012 and Zhou et al., 2015
        introducing and refining the approach.
        * OpenMM implementation with optional resampling https://gist.github.com/maxentile/be328e929abf4a92bee7d26967277f54
            with the threshold defined using a different criterion ("conditional effective sample size") vs. stddev(w)
        * More sophisticated implementation of adaptive SMC in perses
            https://github.com/choderalab/perses/blob/18ec8b9d69afeb6128b251cf1d1b89ac7801ed68/perses/app/relative_setup.py#L1378-L1838
    * A closely related approach "thermodynamic trailblazing" is developed in Andrea Rizzi's thesis
        https://search.proquest.com/openview/0f0bda7dc135aad7216b6acecb815d3c/1.pdf?pq-origsite=gscholar&cbl=18750&diss=y
        and implemented in Yank
        https://github.com/choderalab/yank/blob/59fc6313b3b7d82966afc539604c36f4db9b952c/Yank/pipeline.py#L1983-L2648
        One substantive difference compared with trailblazing is that here the samples are not in equilibrium after
        step 0, and here optimization only uses information in one direction.
    """

    sample_traj = [samples_0]
    lam_traj = [0.0]

    while lam_traj[-1] < 1.0:
        samples, lam = sample_traj[-1], lam_traj[-1]

        options = dict(max_increment_size=1.0 - lam, incremental_stddev_threshold=incremental_stddev_threshold)
        updated_lam = lam + find_next_increment(samples, lam, **options)
        print(f'next lambda={updated_lam:.4f}')

        updated_samples = propagate(samples, updated_lam, n_steps=n_md_steps_per_increment)

        sample_traj.append(updated_samples)
        lam_traj.append(updated_lam)

    return sample_traj, np.array(lam_traj)


if __name__ == '__main__':

    # collect endstate samples
    n_equil_steps = 10000
    v_0 = sample_velocities(masses * unit.amu, temperature)
    initial_state = CoordsVelBox(coords, v_0, complex_box)
    print('equilibrating...')
    thermostat = UnadjustedLangevinMove(integrator_impl, bound_impls, lam=0.0, n_steps=n_equil_steps)
    equilibrated = thermostat.move(initial_state)

    print('collecting samples from lam=0...')
    samples_0 = sample_at_equilibrium(equilibrated, lam=0.0)

    adaptation_options = dict(
        n_md_steps_per_increment=100,  # number of MD steps run at fixed lambda, between lambda increments
        incremental_stddev_threshold=0.25,  # tolerable stddev(w) in k_BT per lambda increment
    )

    sample_traj, lam_traj = adaptive_noneq(samples_0, **adaptation_options)

    print('collecting samples from lam=1...')
    samples_1 = sample_at_equilibrium(sample_traj[-1][-1], lam=1.0)

    # compute work via u(x, lam[t+1]) - u(x, lam[t]) increments
    work_increments = []
    for (X, lam_init, lam_final) in zip(sample_traj[:-1], lam_traj[:-1], lam_traj[1:]):
        work_increments.append(u_vec(X, lam_final) - u_vec(X, lam_init))
    work_increments = np.array(work_increments)
    works = np.sum(work_increments, 0)
    print(f'EXP(w_f): {EXP(works)}\n(via w = sum_t u(x_t, lam[t+1]) - u(x_t, lam[t])')

    # construct interpolated version of this schedule, rather than doing cycles of
    #   lambda increment <-> MD propagation
    total_md_steps = len(lam_traj) * 100
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
