import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdkit import Chem
from scipy.special import logsumexp

from timemachine.constants import BOLTZ, DEFAULT_FF
from timemachine.datasets import fetch_freesolv
from timemachine.fe.absolute_hydration import set_up_ahfe_system_for_smc
from timemachine.fe.utils import get_mol_name
from timemachine.ff import Forcefield
from timemachine.md.smc import get_endstate_samples_from_smc_result, sequential_monte_carlo
from timemachine.parallel.client import CUDAPoolClient, SerialClient
from timemachine.parallel.utils import get_gpu_count

temperature = 300

# This is needed for pickled mols to have their properties preserved
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_walkers", type=int, help="number of walkers", default=100)
    parser.add_argument("--n_windows", type=int, help="number of lambda windows", default=100)
    parser.add_argument("--n_md_steps", type=int, help="number of MD steps per move", default=100)
    parser.add_argument("--resample_thresh", type=float, help="resample when fractional ESS < thresh", default=0.6)
    parser.add_argument("--debug_mode", type=bool, help="save full trajectories", default=False)
    parser.add_argument("--n_mols", type=int, help="how many freesolv molecules to run on")
    parser.add_argument("--seed", type=int, help="random seed used for np.random and MD mover", default=2022)
    parser.add_argument("--result_path", type=str, help="path with smc results", default=".")
    parser.add_argument("--filter_mols", type=str, help="filter molecules", nargs="+", default=[])
    parser.add_argument("--n_gpus", type=int, help="number of gpus", default=get_gpu_count())
    parser.add_argument(
        "--n_cpus",
        type=int,
        help="number of cpus to use for each SMC worker. None means use the default which varies based on the client.",
        default=None,
    )
    parser.add_argument(
        "--ff", type=str, help="path to forcefield file or use default SMIRNOFF ff if not set", default=DEFAULT_FF
    )

    cmd_args = parser.parse_args()

    if cmd_args.debug_mode:
        n_frames = cmd_args.n_walkers * cmd_args.n_windows
        print(f"Warning! This will take a lot of disk space! ({n_frames} simulation frames)")

    return cmd_args


def get_ff(ff_path=DEFAULT_FF) -> Forcefield:
    """
    Load the forcefield given the path to the ff py file.
    """
    return Forcefield.load_from_file(ff_path)


def get_result_path(path: Path, mol_id: str) -> Path:
    """
    Return the path to the smc results for a particular molecule.
    """
    return path / f"summary_smc_result_{mol_id}.pkl"


def get_full_traj_path(path: Path, mol_id: str) -> Path:
    return path / f"full_smc_traj_{mol_id}.pkl"


def save_smc_result(path: Path, mol: int, smc_result: Dict, cmd_args: argparse.Namespace, save_full_trajectories=False):
    """
    Save the smc results as a pkl'd dictionary.

    Parameters
    ----------
    path:
        Path to the smc results pkl cache

    mol: ROMol
        Molecule sampled using smc

    smc_result:
        'initial_samples_refined, 'final_samples_refined' correspond to
        the refined samples from the end states.
        'initial_log_weights', 'final_log_weights' correspond to the
        log weights from both states.
        'traj' is a list if trajectories, one for each lambda.

    cmd_args:
        Command line arguments, stored for reproducibility.

    save_full_trajectories:
        Set to True to store the full smc trajectories

    """
    mol_name = get_mol_name(mol)
    summary = dict(
        mol=mol,
        cmd_args=cmd_args,
        initial_samples_refined=smc_result["initial_samples_refined"],
        initial_log_weights=smc_result["log_weights_traj"][0],
        final_samples_refined=smc_result["final_samples_refined"],
        final_log_weights=smc_result["log_weights_traj"][-1],
        ancestry_traj=smc_result["ancestry_traj"],
        incremental_log_weights_traj=smc_result["incremental_log_weights_traj"],
    )
    if save_full_trajectories:
        summary["initial_samples"] = smc_result["traj"][0]
        summary["final_samples"] = smc_result["traj"][-1]

    with open(get_result_path(path, mol_name), "wb") as f:
        pickle.dump(summary, f)

    # optionally save trajectories
    if save_full_trajectories:
        with open(get_full_traj_path(path, mol_name), "wb") as f:
            pickle.dump(smc_result, f)


def run_on_freesolv_mol(mol: Chem.rdchem.Mol, cmd_args: argparse.Namespace):
    name = get_mol_name(mol)
    props = mol.GetPropsAsDict()
    result_path = Path(cmd_args.result_path)
    print(f"running on molecule {name}, dG={props['dG']} kcal/mol")

    # prepare initial samples and lambda schedule, define functions for propagating, evaluating log_prob, and resampling
    samples, lambdas, propagate, log_prob, resample = set_up_ahfe_system_for_smc(
        mol,
        cmd_args.n_walkers,
        cmd_args.n_windows,
        cmd_args.n_md_steps,
        cmd_args.resample_thresh,
        seed=cmd_args.seed,
        ff=get_ff(cmd_args.ff),
        num_workers=cmd_args.n_cpus,
    )
    # run simulation
    smc_result = sequential_monte_carlo(samples, lambdas, propagate, log_prob, resample)
    log_weights = smc_result["log_weights_traj"][-1]
    reduced_dG = -logsumexp(log_weights - np.log(len(log_weights)))
    dG_kJmol = reduced_dG * (BOLTZ * temperature)
    dG = dG_kJmol / 4.184

    print(f"predicted: {dG:.3f} kcal/mol")
    print(f"experimental: {props['dG']:.3f} kcal/mol")

    initial_samples_refined, final_samples_refined = get_endstate_samples_from_smc_result(
        smc_result, propagate, lambdas
    )
    smc_result["initial_samples_refined"] = initial_samples_refined
    smc_result["final_samples_refined"] = final_samples_refined

    # save summary
    save_smc_result(result_path, mol, smc_result, cmd_args, save_full_trajectories=cmd_args.debug_mode)


def run_on_mols(mols: List[Chem.rdchem.Mol], cmd_args: argparse.Namespace):
    for mol in mols:
        run_on_freesolv_mol(mol, cmd_args)


def main():
    cmd_args = parse_options()
    mols = fetch_freesolv(n_mols=cmd_args.n_mols, filter_mols=cmd_args.filter_mols)

    # Create result folder
    result_path = Path(cmd_args.result_path)
    result_path.mkdir(exist_ok=True, parents=True)

    # Set up client
    num_gpus = cmd_args.n_gpus or 1
    cmd_args.n_cpus = cmd_args.n_cpus or os.cpu_count()
    cmd_args.n_cpus = cmd_args.n_cpus // num_gpus

    if num_gpus > 1:
        client = CUDAPoolClient(max_workers=num_gpus)
    else:
        client = SerialClient()
    client.verify()
    print(f"using {num_gpus} gpus with {cmd_args.n_cpus} cpus per gpu")

    # Batch mols
    batch_mols = defaultdict(list)
    for i in range(len(mols)):
        batch_mols[i % num_gpus].append(mols[i])

    futures = []
    for mol_subset in batch_mols.values():
        futures.append(client.submit(run_on_mols, mol_subset, cmd_args))

    # Wait for jobs to complete
    _ = [fut.result() for fut in futures]


if __name__ == "__main__":
    main()
