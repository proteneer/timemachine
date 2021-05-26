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

from testsystem import (
    temperature, timestep,
    integrator_impl, ensemble, potential_energy_model,
    coords, masses, complex_box,
)

# paths where we'll later save results
work_increments_path = os.path.join(os.path.dirname(__file__), 'results/works_via_potential_increments.npy')
optimized_lam_traj_path = os.path.join(os.path.dirname(__file__), 'results/optimized_lam_traj.npy')

# equilibrium options
n_equil_steps = 10000
n_samples = 100

# adaptation options
n_md_steps_per_increment = 100  # number of MD steps run at fixed lambda, between lambda increments
incremental_stddev_threshold = 0.25  # tolerable stddev(w) in k_BT per lambda increment


def u(state: CoordsVelBox, lam: float) -> float:
    """compute reduced potential"""
    energy, gradient = ensemble.reduced_potential_and_gradient(state.coords, state.box, lam)
    return energy


def u_vec(states: List[CoordsVelBox], lam: float) -> np.array:
    """compute reduced potential on list of states"""
    return np.array([u(state, lam) for state in states])


def sample_at_equilibrium(initial_state: CoordsVelBox, lam: float = 0.0, thinning: int = 1000, n_samples: int = 100) -> \
        List[CoordsVelBox]:
    """run MD"""

    thermostat = UnadjustedLangevinMove(integrator_impl, potential_energy_model.all_impls, lam, n_steps=thinning)

    samples = [initial_state]
    for _ in tqdm(range(n_samples)):
        samples.append(thermostat.move(samples[-1]))

    return samples[1:]


def propagate(states: List[CoordsVelBox], lam: float = 0.0, n_steps: float = 500) -> List[CoordsVelBox]:
    thermostat = UnadjustedLangevinMove(integrator_impl, potential_energy_model.all_impls, lam, n_steps=n_steps)

    print(f'propagating {len(states)} systems by {n_steps * timestep.value_in_unit(unit.picosecond)}ps each...')
    updated_states = []
    for state in tqdm(states):  # TODO: loop could be paralllelized (e.g. on CUDAPoolClient)
        updated_states.append(thermostat.move(state))

    return updated_states


def find_next_increment(
        samples: List[CoordsVelBox], lam_initial: float,
        max_increment_size: float = 0.1, incremental_stddev_threshold: float = 0.1, xtol: float = 1e-5
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
        Differences compared with trailblazing include that here the samples are not in equilibrium after
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
    v_0 = sample_velocities(masses * unit.amu, temperature)
    initial_state = CoordsVelBox(coords, v_0, complex_box)
    print('equilibrating...')
    thermostat = UnadjustedLangevinMove(integrator_impl, potential_energy_model.all_impls, lam=0.0,
                                        n_steps=n_equil_steps)
    equilibrated = thermostat.move(initial_state)

    print(f'collecting {n_samples} samples from lam=0...')
    samples_0 = sample_at_equilibrium(equilibrated, lam=0.0, n_samples=n_samples)

    # run switching with adaptive lambda steps
    print(f'running adaptive noneq switching with {n_samples} trajectories')
    sample_traj, lam_traj = adaptive_noneq(
        samples_0,
        n_md_steps_per_increment=n_md_steps_per_increment,
        incremental_stddev_threshold=incremental_stddev_threshold,
    )

    print(f'saving optimized lambda schedule to {optimized_lam_traj_path}')
    np.save(optimized_lam_traj_path, lam_traj)

    # compute work via sum of u(x, lam[t+1]) - u(x, lam[t]) increments
    work_increments = []
    for (X, lam_init, lam_final) in zip(sample_traj[:-1], lam_traj[:-1], lam_traj[1:]):
        work_increments.append(u_vec(X, lam_final) - u_vec(X, lam_init))
    work_increments = np.array(work_increments)
    works = np.sum(work_increments, 0)
    print(f'stddev(w_f): {np.std(works):.3f} kBT')
    print(f'EXP(w_f): {EXP(works)[0]:.3f} kBT')
    print('(with work computed via w = sum_t u(x_t, lam[t+1]) - u(x_t, lam[t])')

    print(f'saving works to {work_increments_path}')
    np.save(work_increments_path, work_increments)
