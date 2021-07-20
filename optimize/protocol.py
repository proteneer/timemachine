from scipy.optimize import root_scalar
from md.thermostat.moves import UnadjustedLangevinMove

from simtk import unit
import numpy as np
from tqdm import tqdm

from pymbar import EXP

import numpy as np
from typing import Callable, List, Tuple
from md.states import CoordsVelBox

Lambda = float
Energies = np.array
VectorizedPotentialEnergy = Callable[[List[CoordsVelBox], Lambda], Energies]
VectorizedPropagator = Callable[[List[CoordsVelBox], Lambda], List[CoordsVelBox]]
Samples = List[CoordsVelBox]


def construct_potential_wrappers(ensemble) -> Tuple[Callable, Callable]:
    """construct functions u(state, lam), u_vec(stats, lam)"""

    def u(state: CoordsVelBox, lam: float) -> float:
        """compute reduced potential"""
        energy, gradient = ensemble.reduced_potential_and_gradient(state.coords, state.box, lam)
        return energy

    def u_vec(states: List[CoordsVelBox], lam: float) -> np.array:
        """compute reduced potential on list of states"""
        return np.array([u(state, lam) for state in states])

    return u, u_vec



def construct_md_wrappers(ensemble, integrator_impl, timestep=2.0 * unit.femtosecond):
    """construct functions sample_at_equilibrium(x0, lam), propagate(xs, lam)"""

    def sample_at_equilibrium(
            initial_state: CoordsVelBox,
            lam: float = 0.0, thinning: int = 1000, n_samples: int = 100) -> List[CoordsVelBox]:
        """run MD"""

        thermostat = UnadjustedLangevinMove(integrator_impl, ensemble.potential_energy.all_impls, lam, n_steps=thinning)

        samples = [initial_state]
        for _ in tqdm(range(n_samples)):
            samples.append(thermostat.move(samples[-1]))

        return samples[1:]

    def propagate(states: List[CoordsVelBox], lam: float = 0.0, n_steps: float = 500) -> List[CoordsVelBox]:
        thermostat = UnadjustedLangevinMove(integrator_impl, ensemble.potential_energy.all_impls, lam, n_steps=n_steps)

        print(f'propagating {len(states)} systems by {n_steps * timestep.value_in_unit(unit.picosecond)}ps each...')
        updated_states = []
        for state in tqdm(states):  # TODO: loop could be paralllelized (e.g. on CUDAPoolClient)
            updated_states.append(thermostat.move(state))

        return updated_states

    return sample_at_equilibrium, propagate


def find_next_increment(
        samples: List[CoordsVelBox],
        u_vec: VectorizedPotentialEnergy,
        lam_initial: float,
        max_increment_size: float = 0.1,
        incremental_stddev_threshold: float = 0.1,
        xtol: float = 1e-5
) -> float:

    u_s = u_vec(samples, lam_initial)

    def work_increment_stddev(lam_increment: float) -> float:
        """stddev(u(samples, lam + lam_increment) - u(samples, lam))"""
        lam = lam_initial + lam_increment
        u_trial = u_vec(samples, lam)
        stddev = np.std(u_trial - u_s)

        # root-finder needs to check sign of stddev
        return np.nan_to_num(stddev, nan=+np.inf)

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


def adaptive_noneq(
        samples_0: List[CoordsVelBox],
        u_vec: VectorizedPotentialEnergy,
        propagate: VectorizedPropagator,
        incremental_stddev_threshold=0.5
):
    """Generate lam=0 -> lam=1 trajectories by a scheme that makes adaptively sized lambda increments.

        Alternates between the following two steps:
        * Select the next lambda increment by finding the root of
            f(increment) = stddev(u(samples, lam + increment) - u(samples, lam)) - incremental_stddev_threshold
        * Propagate all samples for some small number of MD/MCMC steps

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
        updated_lam = lam + find_next_increment(samples, u_vec, lam, **options)
        lam_traj.append(updated_lam)
        print(f'next lambda={updated_lam:.6f}')

        if updated_lam < 1.0:
            updated_samples = propagate(samples, updated_lam)
            sample_traj.append(updated_samples)

    return sample_traj, np.array(lam_traj)


def compute_work_increments(
        u_vec: VectorizedPotentialEnergy,
        sample_traj: List[Samples],
        lam_traj: List[Lambda]) -> np.array:
    """For computing work via sum of u(x, lam[t+1]) - u(x, lam[t]) increments"""

    work_increments = []
    for (X, lam_init, lam_final) in zip(sample_traj, lam_traj[:-1], lam_traj[1:]):
        work_increments.append(u_vec(X, lam_final) - u_vec(X, lam_init))
    work_increments = np.array(work_increments)
    works = np.sum(work_increments, 0)
    print(f'stddev(w_f): {np.std(works):.3f} kBT')
    print(f'EXP(w_f): {EXP(works)[0]:.3f} kBT')
    print('(with work computed via w = sum_t u(x_t, lam[t+1]) - u(x_t, lam[t])')

    return work_increments


def interpolate_lambda_schedule(lambda_schedule, num_md_steps):
    """given a lambda schedule, with n windows, turn it into a lambda
    schedule with num_md_steps windows by interpolation"""
    n = len(lambda_schedule)
    xp, fp = np.linspace(0, 1, n), np.array(lambda_schedule)
    md_steps = np.linspace(0, 1, num_md_steps)
    interpolated_schedule = np.interp(md_steps, xp, fp)

    return interpolated_schedule
