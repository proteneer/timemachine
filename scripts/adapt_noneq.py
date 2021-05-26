from md.builders import build_water_system
from md.minimizer import minimize_host_4d
from md.ensembles import PotentialEnergyModel
from md.thermostat.utils import sample_velocities
from md.thermostat.moves import UnadjustedLangevinMove
from md.states import CoordsVelBox
from fe.free_energy import AbsoluteFreeEnergy, construct_lambda_schedule

from testsystems.relative import hif2a_ligand_pair

from simtk import unit
import numpy as np
from scipy.optimize import root_scalar

from timemachine.lib import LangevinIntegrator, custom_ops
from timemachine.constants import kB
from tqdm import tqdm

from typing import List
from functools import partial

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


def u(state: CoordsVelBox, lam: float):
    """evaluate reduced potential """
    energy, gradient = potential_energy_model.energy_and_gradient(state.coords, state.box, lam)
    return energy * unit.kilojoule_per_mole / (kB * temperature)


def u_vec(states: List[CoordsVelBox], lam: float):
    return np.array([u(state, lam) for state in states])


def sample_endstate(initial_state, lam=0.0, thinning=1000, n_samples=100):
    thermostat = UnadjustedLangevinMove(integrator_impl, potential_energy_model.all_impls, lam, n_steps=thinning)

    samples = [initial_state]
    for _ in tqdm(range(n_samples)):
        samples.append(thermostat.move(samples[-1]))

    return samples


def propagate(states, lam=0.0, n_steps=500):
    thermostat = UnadjustedLangevinMove(integrator_impl, potential_energy_model.all_impls, lam, n_steps=n_steps)

    updated_states = []
    for state in tqdm(states):  # TODO: loop could be paralllelized (e.g. on CUDAPoolClient)
        updated_states.append(thermostat.move(state))

    return updated_states


def find_next_increment(samples, lam_initial, max_increment_size=0.1, incremental_stddev_threshold=0.1, xtol=1e-4):
    u_s = u_vec(samples, lam_initial)

    def work_increment_stddev(lam_increment):
        lam = lam_initial + lam_increment
        u_trial = u_vec(samples, lam)
        return np.std(u_trial - u_s)

    def f(lam):
        return work_increment_stddev(lam) - incremental_stddev_threshold

    # try-except to catch rootfinding ValueError: f(a) and f(b) must have different signs
    #   which occurs when jumping all the way to lam=1.0 is still less than threshold
    try:
        result = root_scalar(f, bracket=(0, max_increment_size), xtol=xtol)
        lam_increment = result.root
    except ValueError as e:
        print(f'root finding error: {e}')
        lam_increment = max_increment_size

    return lam_increment


def noneq_move(x, lambda_schedule):
    ctxt = custom_ops.Context(x.coords, x.velocities, x.box, integrator_impl, potential_energy_model.all_impls)

    # arguments: lambda_schedule, du_dl_interval, x_interval
    du_dl_traj, _ = ctxt.multiple_steps(lambda_schedule, 1, 0)
    after = CoordsVelBox(ctxt.get_x_t(), ctxt.get_v_t(), x.box.copy())

    return after, du_dl_traj


def adaptive_noneq(samples_0, n_md_steps_per_increment=100, incremental_stddev_threshold=0.5):
    """ generate lam=0 -> lam=1 trajectory """

    sample_traj = [samples_0]
    lam_traj = [0.0]

    while lam_traj[-1] < 1.0:
        samples, lam = sample_traj[-1], lam_traj[-1]

        print('finding next lambda increment...')
        options = dict(max_increment_size=1.0 - lam, incremental_stddev_threshold=incremental_stddev_threshold)
        next_increment = find_next_increment(samples, lam, **options)
        updated_lam = lam + next_increment
        print(f'found! new lambda={updated_lam:.4f}')

        print('propagating...')
        updated_samples = propagate(samples, updated_lam, n_steps=n_md_steps_per_increment)
        print('done!')

        sample_traj.append(updated_samples)
        lam_traj.append(updated_lam)

    return sample_traj, np.array(lam_traj)


def reduced_work_from_du_dl_trajs(lambda_schedule, du_dl_trajs):
    Works = np.trapz(du_dl_trajs, lambda_schedule, axis=1) * unit.kilojoule_per_mole
    works = Works / (kB * temperature)

    return works


if __name__ == '__main__':

    # collect endstate samples
    v_0 = sample_velocities(masses * unit.amu, temperature)
    initial_state = CoordsVelBox(coords, v_0, complex_box)
    print('equilibrating...')
    thermostat = UnadjustedLangevinMove(integrator_impl, bound_impls, lam=0.0, n_steps=10000)
    equilibrated = thermostat.move(initial_state)

    print('collecting samples from lam=0...')
    samples_0 = sample_endstate(equilibrated, lam=0.0, thinning=1000, n_samples=100)

    adaptation_options = dict(
        n_md_steps_per_increment=100,  # number of MD steps run at fixed lambda, between lambda increments
        incremental_stddev_threshold=0.5,  # tolerable stddev(w) in k_BT per lambda increment
    )

    sample_traj, lam_traj = adaptive_noneq(samples_0, **adaptation_options)

    # compute work via u(x, lam[t+1]) - u(x, lam[t]) increments
    work_increments = []
    for (X, lam_init, lam_final) in zip(sample_traj[:-1], lam_traj[:-1], lam_traj[1:]):
        work_increments.append(u_vec(X, lam_final) - u_vec(X, lam_init))
    work_increments = np.array(work_increments)

    # construct interpolated version of this schedule, rather than doing cycles of
    #   lambda increment <-> MD propagation
    total_md_steps = len(lam_traj) * 100
    md_steps = np.linspace(0, 1, total_md_steps)
    xp = np.linspace(0, 1, len(lam_traj))
    fp = np.array(lam_traj)
    optimized_schedule = np.interp(md_steps, xp, fp)

    default_schedule = construct_lambda_schedule(total_md_steps)

    # now run timemachine noneq moves, computing work via du_dl increments
    default_noneq_move = partial(noneq_move, lambda_schedule=default_schedule)
    optimized_noneq_move = partial(noneq_move, lambda_schedule=optimized_schedule)

    default_du_dl_trajs = np.array([default_noneq_move(x0) for x0 in tqdm(samples_0)])
    optimized_du_dl_trajs = np.array([optimized_noneq_move(x0) for x0 in tqdm(samples_0)])

    default_w = reduced_work_from_du_dl_trajs(default_schedule, default_du_dl_trajs)
    optimized_w = reduced_work_from_du_dl_trajs(optimized_schedule, optimized_du_dl_trajs)

    # TODO: plots and analysis
