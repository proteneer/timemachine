import argparse
from datetime import datetime
from pickle import dump

import numpy as np
from scipy.special import logsumexp

from timemachine.constants import BOLTZ
from timemachine.datasets import fetch_freesolv
from timemachine.fe.absolute_hydration import set_up_ahfe_system_for_smc
from timemachine.md.smc import sequential_monte_carlo

temperature = 300


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_walkers", type=int, help="number of walkers", default=100)
    parser.add_argument("--n_windows", type=int, help="number of lambda windows", default=100)
    parser.add_argument("--n_md_steps", type=int, help="number of MD steps per move", default=100)
    parser.add_argument("--resample_thresh", type=float, help="resample when fractional ESS < thresh", default=0.6)
    parser.add_argument("--debug_mode", type=bool, help="save full trajectories", default=False)
    parser.add_argument("--n_mols", type=int, help="how many freesolv molecules to run on", default=10)

    cmd_args = parser.parse_args()

    if cmd_args.debug_mode:
        n_frames = cmd_args.n_walkers * cmd_args.n_windows
        print(f"Warning! This will take a lot of disk space! ({n_frames} simulation frames)")

    return cmd_args


def save_smc_result(uid, smc_result, save_full_trajectories=False):

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


def run_on_freesolv_mol(mol):
    name = mol.GetProp("_Name")
    props = mol.GetPropsAsDict()
    print(f"running on molecule {name}, dG={props['dG']} kcal/mol")

    # prepare initial samples and lambda schedule, define functions for propagating, evaluating log_prob, and resampling
    samples, lambdas, propagate, log_prob, resample = set_up_ahfe_system_for_smc(
        mol, cmd_args.n_walkers, cmd_args.n_windows, cmd_args.n_md_steps, cmd_args.resample_thresh
    )
    # run simulation
    smc_result = sequential_monte_carlo(samples, lambdas, propagate, log_prob, resample)
    log_weights = smc_result["log_weights_traj"][-1]
    reduced_dG = -logsumexp(log_weights - np.log(len(log_weights)))
    dG_kJmol = reduced_dG * (BOLTZ * temperature)
    dG = dG_kJmol / 4.184

    print(f"predicted: {dG:.3f} kcal/mol")
    print(f"experimental: {props['dG']:.3f} kcal/mol")

    # save summary
    uid = f"mol={name}_time={datetime.now().isoformat(timespec='seconds')}"
    save_smc_result(uid, smc_result, save_full_trajectories=cmd_args.debug_mode)


if __name__ == "__main__":
    cmd_args = parse_options()

    mols = fetch_freesolv()
    select_mols = mols[: cmd_args.n_mols]

    for mol in select_mols:
        run_on_freesolv_mol(mol)
