import argparse
from datetime import datetime
from functools import partial
from pickle import dump

import numpy as np

from timemachine.fe.free_energy_rabfe import construct_pre_optimized_absolute_lambda_schedule_solvent
from timemachine.md.smc import simple_smc, conditional_multinomial_resample
from timemachine.testsystems.biphenyl import construct_biphenyl_test_system


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_walkers", type=int, help="number of walkers", default=100)
    parser.add_argument("--n_windows", type=int, help="number of lambda windows", default=100)
    parser.add_argument("--n_md_steps", type=int, help="number of MD steps per move", default=100)
    parser.add_argument("--resample_thresh", type=float, help="resample when fractional ESS < thresh", default=0.6)
    parser.add_argument("--debug_mode", type=bool, help="save full trajectories", default=False)

    cmd_args = parser.parse_args()

    if cmd_args.debug_mode:
        n_frames = cmd_args.n_walkers * cmd_args.n_windows
        print(f"Warning! This will take a lot of disk space! ({n_frames} simulation frames)")

    return cmd_args


def set_up_biphenyl_system_for_smc(n_walkers, n_windows, n_md_steps, resample_thresh):
    """define initial samples, lambdas schedule, propagate fxn, log_prob fxn, resample fxn"""
    reduced_potential, mover, initial_samples = construct_biphenyl_test_system(n_steps=n_md_steps)
    sample_inds = np.random.choice(np.arange(len(initial_samples)), size=n_walkers)
    samples = [initial_samples[i] for i in sample_inds]

    lambdas = construct_pre_optimized_absolute_lambda_schedule_solvent(n_windows)[::-1]

    def propagate(xs, lam):
        mover.lamb = lam
        xs_next = [mover.move(x) for x in xs]
        return xs_next

    def log_prob(xs, lam):
        u_s = np.array([reduced_potential(x, lam) for x in xs])
        return -u_s

    resample = partial(conditional_multinomial_resample, thresh=resample_thresh)

    return samples, lambdas, propagate, log_prob, resample


def save_smc_result(smc_result, save_full_trajectories=False):
    uid = f"{datetime.now()}"

    traj, log_weights_traj, ancestry_traj, incremental_log_weights_traj = smc_result

    # by default, just save the final weighted samples, incremental log weight trajectory (and cmd_args)
    summary = dict(
        final_samples=traj[-1],
        final_log_weights=log_weights_traj[-1],
        ancestry_traj=ancestry_traj,
        incremental_log_weights_traj=incremental_log_weights_traj,
    )
    with open(f"summary_smc_result_{uid}.pkl", "wb") as f:
        dump((summary, cmd_args), f)

    # optionally save trajectories
    if save_full_trajectories:
        with open(f"full_smc_traj_{uid}.pkl", "wb") as f:
            dump((traj, cmd_args), f)


if __name__ == "__main__":
    cmd_args = parse_options()

    # prepare initial samples and lambda schedule, define functions for propagating, evaluating log_prob, and resampling
    samples, lambdas, propagate, log_prob, resample = set_up_biphenyl_system_for_smc(
        cmd_args.n_walkers, cmd_args.n_windows, cmd_args.n_md_steps, cmd_args.resample_thresh
    )

    # run simulation
    smc_result = simple_smc(samples, lambdas, propagate, log_prob, resample)

    # save summary
    save_smc_result(smc_result, save_full_trajectories=cmd_args.debug_mode)
