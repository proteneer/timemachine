import argparse
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.special import logsumexp

from timemachine.constants import BOLTZ, DEFAULT_FF, KCAL_TO_KJ
from timemachine.fe.absolute_hydration import set_up_ahfe_system_for_smc
from timemachine.fe.utils import get_mol_name
from timemachine.ff import Forcefield
from timemachine.md.smc import get_endstate_samples_from_smc_result, sequential_monte_carlo
from timemachine.parallel.client import AbstractFileClient, CUDAPoolClient, FileClient, save_results
from timemachine.parallel.utils import batch_list, get_gpu_count
from timemachine.testsystems import fetch_freesolv

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
    parser.add_argument(
        "--exclude_mols", type=str, help="exclude the given molecules from the run", nargs="+", default=[]
    )
    parser.add_argument("--n_gpus", type=int, help="number of gpus", default=get_gpu_count())
    parser.add_argument(
        "--n_cpus",
        type=int,
        help="number of cpus to use for each SMC worker. None means use the default of all cpus on the worker.",
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


def get_result_path(mol_id: str) -> str:
    """
    Return the path to the smc results for a particular molecule.
    """
    return f"summary_smc_result_{mol_id}.pkl"


def get_full_traj_path(mol_id: str) -> str:
    return f"full_smc_traj_{mol_id}.pkl"


def save_smc_result(
    file_client: AbstractFileClient,
    mol: Chem.rdchem.Mol,
    smc_result: dict,
    cmd_args: argparse.Namespace,
    save_full_trajectories=False,
):
    """
    Save the smc results as a pkl'd dictionary.

    Parameters
    ----------
    file_client:
        Client used to store the result files.

    mol: ROMol
        Molecule sampled using smc

    smc_result:
        'initial_samples_refined, 'final_samples_refined' correspond to
        the refined samples from the end states.
        'initial_log_weights', 'final_log_weights' correspond to the
        log weights from both states.
        'traj' list of samples for lambdas, except for the last one

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

    result_path = get_result_path(mol_name)
    pkl_contents = pickle.dumps(summary)
    file_client.store(result_path, pkl_contents)

    # optionally save trajectories
    if save_full_trajectories:
        full_traj_path = get_full_traj_path(mol_name)
        pkl_contents = pickle.dumps(smc_result)
        file_client.store(full_traj_path, pkl_contents)


def run_on_freesolv_mol(
    file_client: AbstractFileClient, mol: Chem.rdchem.Mol, ff: Forcefield, cmd_args: argparse.Namespace
) -> str:
    """
    Returns
    -------
    str:
        Relative path to the result pkl.
    """
    name = get_mol_name(mol)
    props = mol.GetPropsAsDict()
    print(f"running on molecule {name}, dG={props['dG']} kcal/mol")

    # prepare initial samples and lambda schedule, define functions for propagating, evaluating log_prob, and resampling
    samples, lambdas, propagate, log_prob, resample = set_up_ahfe_system_for_smc(
        mol,
        cmd_args.n_walkers,
        cmd_args.n_windows,
        cmd_args.n_md_steps,
        cmd_args.resample_thresh,
        seed=cmd_args.seed,
        ff=ff,
        num_workers=cmd_args.n_cpus,
    )
    # run simulation
    smc_result = sequential_monte_carlo(samples, lambdas, propagate, log_prob, resample)
    log_weights = smc_result["log_weights_traj"][-1]
    reduced_dG = -logsumexp(log_weights - np.log(len(log_weights)))
    dG_kJmol = reduced_dG * (BOLTZ * temperature)
    dG = dG_kJmol / KCAL_TO_KJ

    print(f"predicted: {dG:.3f} kcal/mol")
    print(f"experimental: {props['dG']:.3f} kcal/mol")

    initial_samples_refined, final_samples_refined = get_endstate_samples_from_smc_result(
        smc_result, propagate, lambdas
    )
    smc_result["initial_samples_refined"] = initial_samples_refined
    smc_result["final_samples_refined"] = final_samples_refined

    # save summary
    save_smc_result(file_client, mol, smc_result, cmd_args, save_full_trajectories=cmd_args.debug_mode)
    return get_result_path(get_mol_name(mol))


def run_on_mols(
    file_client: AbstractFileClient, mols: list[Chem.rdchem.Mol], ff: Forcefield, cmd_args: argparse.Namespace
) -> list[str]:
    results = []
    for mol in mols:
        results.append(run_on_freesolv_mol(file_client, mol, ff, cmd_args))
    return results


def main():
    cmd_args = parse_options()
    mols = fetch_freesolv(n_mols=cmd_args.n_mols, exclude_mols=cmd_args.exclude_mols)

    # Create result folder
    result_path = Path(cmd_args.result_path)
    result_path.mkdir(exist_ok=True, parents=True)

    # Set up client
    num_gpus = cmd_args.n_gpus or 1
    client = CUDAPoolClient(max_workers=num_gpus)
    client.verify()
    file_client = FileClient(base=Path(cmd_args.result_path))
    print(f"using {num_gpus} gpus with {cmd_args.n_cpus or 'default'} cpus per gpu")

    # Batch mols
    ff = get_ff(cmd_args.ff)
    batch_mols = batch_list(mols, num_gpus)
    futures = []
    for mol_subset in batch_mols:
        futures.append(client.submit(run_on_mols, file_client, mol_subset, ff, cmd_args))

    # Wait for jobs to complete
    batched_results = [fut.result() for fut in futures]
    results = [result for batch in batched_results for result in batch]

    # Copy data
    local_file_client = FileClient(base=Path(cmd_args.result_path))
    save_results(results, local_file_client, file_client)


if __name__ == "__main__":
    main()
