# gradually turn off a single water, displace, then turn on

import numpy as np
from jax import jit
from jax import numpy as jnp
from scipy.special import logsumexp
from tqdm import tqdm

from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.fe.lambda_schedule import construct_pre_optimized_absolute_lambda_schedule_solvent
from timemachine.ff.handlers import openmm_deserializer
from timemachine.integrator import LangevinIntegrator
from timemachine.md.barker import BarkerProposal
from timemachine.md.builders import build_water_system
from timemachine.md.thermostat.utils import sample_velocities
from timemachine.potentials import SummedPotential

cutoff = 1.2


def make_water_lam_potential(box_width=2.0):
    """make TIP3P potential with lam dependence (lam = 1 -> decouple zeroth water)

    Returns
    -------
    U(conf, box lam) -> R
    dU_dx(conf, box, lam) -> R^{Nx3}
    coords : R^{Nx3}
    box : R^{3x3}
    masses : R^N
    """

    system, coords, box, topology = build_water_system(box_width, "tip3p")
    num_atoms = len(coords)
    # num_waters = num_atoms // 3

    bps, masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    ubps = [bp.potential for bp in bps]
    params_0 = [bp.params for bp in bps]
    constant_masses = np.ones_like(masses) * np.mean(masses)

    # get mask to know where to apply w_offset params
    param_mask = [jnp.zeros_like(p, dtype=bool) for p in params_0[:-1]]
    param_mask.append(jnp.zeros_like(params_0[-1], dtype=bool).at[:, 3].set(True))

    # make summed potential
    sp = SummedPotential(ubps, params_0)
    sp_gpu = sp.to_gpu(np.float32)

    param_mask_flat = np.hstack([p.flatten() for p in param_mask])
    flat_params_0 = jnp.hstack([p.flatten() for p in params_0])

    def decouple_zeroth_water(lam):
        w_offsets = jnp.zeros(num_atoms)
        w_offsets = w_offsets.at[:3].set(cutoff * lam)
        return w_offsets

    def apply_w_offsets_flat(flat_params_0, w_offsets):
        return flat_params_0.at[param_mask_flat].set(w_offsets)

    @jit
    def flat_params_from_lam(lam):
        w_offsets = decouple_zeroth_water(lam)
        flat_params = apply_w_offsets_flat(flat_params_0, w_offsets)
        return flat_params

    def U(conf, box, lam):
        params = flat_params_from_lam(lam)
        return sp_gpu(conf, params, box)

    def dU_dx(conf, box, lam):
        return sp_gpu.unbound_impl.execute_du_dx(conf, flat_params_from_lam(lam), box)

    return U, dU_dx, coords, box, constant_masses


def make_langevin(dU_dx, box, lam, masses, temperature=DEFAULT_TEMP):
    def force_fxn(x):
        return -dU_dx(x, box, lam)

    langevin = LangevinIntegrator(force_fxn, masses=masses, temperature=temperature, dt=2.0e-3, friction=1.0)

    return langevin


# def barker_propagate(barker, x0, n_steps=1000):


def langevin_propagate(langevin, x0, v0, n_steps=1000, seed=None):
    """run Langevin for n_steps"""
    x, v = np.array(x0), np.array(v0)

    if seed is None:
        seed = np.random.randint(10000)
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        x, v = langevin.step(x, v, rng)

    return x, v


def nvt_propagate(dU_dx, x0, box, lam, masses, temperature=DEFAULT_TEMP, n_samples=100, thinning=1000):
    """Repeatedly call Langevin propagate"""
    x = np.array(x0)
    v = sample_velocities(masses, temperature)

    langevin = make_langevin(dU_dx, box, lam, masses, temperature)

    samples = []
    for _ in tqdm(range(n_samples)):
        x, v = langevin_propagate(langevin, x, v, thinning)
        samples.append(np.array(x))

    return np.array(samples)


def make_symmetric_lam_sched(n_windows_per_direction=100):
    """n_windows = 2 * n_windows_per_direction"""
    lam_fwd = construct_pre_optimized_absolute_lambda_schedule_solvent(n_windows_per_direction)
    return np.hstack([lam_fwd, lam_fwd[::-1]])


def propose_langevin_ncmc(
    U_fxn, dU_dx, x0, v0, box0, lam_sched, masses, n_propagation_steps=100, temperature=DEFAULT_TEMP
):
    """note: applying common but unsafe optimization, using inertial Langevin (not Metropolis-adjusted) as propagation kernel"""

    # TODO: make other MC moves at or near lam = 1 -- currently hard-coding a random displacement move

    x, v = np.array(x0), np.array(v0)

    assert lam_sched[0] == lam_sched[-1] == 0.0

    kBT = BOLTZ * temperature

    def u_fxn(x, lam):
        return U_fxn(x, box0, lam) / kBT

    def compute_incremental_reduced_work(x, lam, lam_next):
        return u_fxn(x, lam_next) - u_fxn(x, lam)

    def propagate(x, v, lam):
        langevin = make_langevin(dU_dx, box0, lam, masses)
        # force_norm = np.max(np.linalg.norm(dU_dx(x, box0, lam), axis=1))
        # if force_norm > MAX_FORCE_NORM:
        # 	print(f"warning: initial conditions at lam = {lam} are likely unstable: max force norm = {force_norm}")
        return langevin_propagate(langevin, x, v, n_propagation_steps)

    incremental_reduced_work_traj = []
    for (lam, lam_next) in zip(tqdm(lam_sched[:-1]), lam_sched[1:]):

        # compute work
        incremental_reduced_work_traj.append(compute_incremental_reduced_work(x, lam, lam_next))

        # propagate
        x, v = propagate(x, v, lam_next)

        # special case lam_next == 1.0
        # TODO: clean this up, use WaterDisplacementMove
        if lam_next == 1.0:
            x[:3] += np.random.randn(3)  # random displacement

    return x, np.array(incremental_reduced_work_traj)


# TODO: apply random permutations, so that x[:3] can reference uniform random waters, rather than always the same water


def propose_instantaneous_displacement(U_fxn, x0, box, lam, temperature=DEFAULT_TEMP):
    def log_q(x):
        return -U_fxn(x, box, lam) / (BOLTZ * temperature)

    log_q_0 = log_q(x0)
    x_prop = np.array(x0)
    x_prop[:3] += np.random.randn(
        3
    )  # random displacement -- TODO: consolidate with repeated definition above using WaterDisplacementMove
    log_q_prop = log_q(x_prop)

    log_accept_prob = np.minimum(0.0, log_q_prop - log_q_0)

    accepted = np.random.rand() < np.exp(log_accept_prob)
    aux = dict(log_accept_prob=log_accept_prob)

    if accepted:
        x_next = x_prop
    else:
        x_next = np.array(x0)

    return x_next, aux


if __name__ == "__main__":
    box_width = 2.0  # side observation: surprised at lack of error / warning message if I call potential with box much smaller than cutoff
    U, dU_dx, coords, box, masses = make_water_lam_potential(box_width=box_width)
    temperature = DEFAULT_TEMP
    kBT = BOLTZ * temperature

    # cache ze samples
    eq_sample_cache_fname = f"waterbox_samples_width={box_width}.npy"
    try:
        xs = np.load(eq_sample_cache_fname)
        print(f"found {len(xs)} cached samples at {eq_sample_cache_fname}")
    except FileNotFoundError:
        print(f"generating eq samples, then saving to {eq_sample_cache_fname}")

        # equilibrate
        def grad_log_q_0(x):
            return -dU_dx(x, box, 0.0) / kBT

        barker_0 = BarkerProposal(grad_log_q_0, seed=1234)

        x_equil = coords
        f0 = np.linalg.norm(dU_dx(x_equil, box, 0.0), axis=1).max()
        print("force norm before equilibration", f0)
        for _ in range(1000):
            x_equil = barker_0.sample(x_equil)
        f1 = np.linalg.norm(dU_dx(x_equil, box, 0.0), axis=1).max()
        print("force norm after equilibration", f1)

        xs = nvt_propagate(dU_dx, x_equil, box, 0.0, masses, n_samples=150, temperature=temperature)
        np.save(eq_sample_cache_fname, xs)

    # plot potential energy traj
    U_traj = np.array([U(x, box, 0.0) for x in xs])
    import matplotlib.pyplot as plt

    plt.plot(U_traj)
    plt.savefig("U_traj.png")
    plt.close()

    eq_samples = xs[50:]  # discard some initial bits due to burn-in?

    # get baseline number to plot as horizontal line
    n_trials_ncmc = 50
    n_trials_instantaneous = 10_000
    instantaneous_log_accept_probs = []

    for _ in tqdm(range(n_trials_instantaneous)):
        x0 = xs[np.random.choice(len(xs))]

        _, aux = propose_instantaneous_displacement(U, x0, box, 0.0, temperature)
        instantaneous_log_accept_probs.append(aux["log_accept_prob"])

    instantaneous_log_accept_prob = logsumexp(np.array(instantaneous_log_accept_probs)) - np.log(n_trials_instantaneous)
    print(f"instantaneous_log_accept_prob = {instantaneous_log_accept_prob}")

    def ncmc_propose(x, n_windows_per_direction=100, n_propagation_steps=100):
        v0 = sample_velocities(masses, temperature)
        lam_sched = make_symmetric_lam_sched(n_windows_per_direction)
        x_prop, incremental_work_traj = propose_langevin_ncmc(
            U, dU_dx, x, v0, box, lam_sched, masses, n_propagation_steps, temperature
        )

        return x_prop, incremental_work_traj

    # TODO: does this function work for corner case of 0 steps?
    n_windows_grid = [2, 5, 10, 20, 50, 100]
    n_propagation_steps_grid = [1, 10, 100]

    results = dict()
    for n_propagation_steps in n_propagation_steps_grid:
        for n_windows in n_windows_grid:
            accept_probs = []
            incremental_work_trajs = []

            print(f"n windows per direction = {n_windows}; n propagation steps = {n_propagation_steps}")
            print("work (kBT) | accept prob | mean accept prob | n attempts")
            for _ in range(n_trials_ncmc):
                x0 = xs[np.random.choice(len(xs))]
                x_prop, incremental_work_traj = ncmc_propose(x0, n_windows, n_propagation_steps)

                log_accept_prob = np.minimum(0.0, -np.sum(incremental_work_traj))
                accept_prob = np.exp(log_accept_prob)
                accept_probs.append(accept_prob)

                work = np.sum(incremental_work_traj)
                print(f"{work} | {accept_prob} | {np.mean(accept_probs)} | {len(accept_probs)}")

                incremental_work_trajs.append(incremental_work_traj)

            results[(n_propagation_steps, n_windows)] = (
                np.array(accept_probs),
                np.array(incremental_work_trajs),
                np.mean(accept_probs),
            )

    import pickle

    with open("langevin_ncmc_info.pkl", "wb") as f:
        pickle.dump(results, f)

    # TODO: maybe pick a different x-axis quantity, like # total MD steps, rather than n_windows?
    plt.hlines(
        np.exp(instantaneous_log_accept_prob),
        min(n_windows_grid),
        max(n_windows_grid),
        colors="grey",
        linestyles="--",
        label="instantaneous",
    )
    for n_propagation_steps in n_propagation_steps_grid:
        avg_accept_probs = [results[(n_propagation_steps, n)][-1] for n in n_windows_grid]

        plt.plot(n_windows_grid, avg_accept_probs, label=f"{n_propagation_steps}")

    scale = "log"
    plt.yscale(scale)
    plt.xlabel("# windows")
    plt.ylabel("avg accept prob")
    plt.legend(title="# propagation steps per window")
    plt.title("langevin + water-displacement NCMC in bulk water")

    plt.savefig(f"avg_accept_probs_figure_{scale}.png")
    plt.close()
