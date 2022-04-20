import argparse
from datetime import datetime
from pickle import dump

from timemachine.fe.absolute_hydration import set_up_ahfe_system_for_smc
from timemachine.md.smc import sequential_monte_carlo
from timemachine.testsystems.ligands import get_biphenyl


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


def save_smc_result(smc_result, save_full_trajectories=False):
    uid = f"{datetime.now().isoformat(timespec='seconds')}"

    # by default, just save the final weighted samples, incremental log weight trajectory (and cmd_args)
    summary = dict(
        final_samples=smc_result["traj"][-1],
        final_log_weights=smc_result["log_weights_traj"][-1],
        ancestry_traj=smc_result["ancestry_traj"],
        incremental_log_weights_traj=smc_result["incremental_log_weights_traj"],
    )
    with open(f"summary_smc_result_{uid}.pkl", "wb") as f:
        dump((summary, cmd_args), f)

    # optionally save trajectories
    if save_full_trajectories:
        with open(f"full_smc_traj_{uid}.pkl", "wb") as f:
            dump((smc_result, cmd_args), f)


if __name__ == "__main__":
    cmd_args = parse_options()

    # prepare initial samples and lambda schedule, define functions for propagating, evaluating log_prob, and resampling
    mol, _ = get_biphenyl()
    samples, lambdas, propagate, log_prob, resample = set_up_ahfe_system_for_smc(
        mol, cmd_args.n_walkers, cmd_args.n_windows, cmd_args.n_md_steps, cmd_args.resample_thresh
    )

    # run simulation
    smc_result = sequential_monte_carlo(samples, lambdas, propagate, log_prob, resample)

    # save summary
    save_smc_result(smc_result, save_full_trajectories=cmd_args.debug_mode)
