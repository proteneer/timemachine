import os
import pickle
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
from common import get_hif2a_ligands_as_sdf_file, temporary_working_dir
from rdkit import Chem
from scipy.special import logsumexp

from timemachine.constants import BOLTZ
from timemachine.datasets import fetch_freesolv
from timemachine.fe.free_energy import RABFEResult
from timemachine.fe.utils import get_mol_name

# All examples are to be tested nightly
pytestmark = [pytest.mark.nightly]


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def run_example(
    example_name: str, cli_args: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None
) -> subprocess.CompletedProcess:
    """
    Runs an example script

    Parameters
    ----------

    example_name: Name of the example
            The name of a file within the examples/ directory

    cli_args: List of command line arguments to pass

    env: Dictionary to override environment variables

    cwd: Directory to run in, defaults in current directory

    Returns
    -------

    Returns the completed subprocess
    """
    example_path = EXAMPLES_DIR / example_name
    assert example_path.is_file(), f"No such example {example_path}"
    subprocess_env = os.environ.copy()
    if env is not None:
        subprocess_env.update(env)
    subprocess_args = [sys.executable, str(example_path), *cli_args]
    proc = subprocess.run(
        subprocess_args,
        env=subprocess_env,
        check=True,
        cwd=cwd,
    )
    return proc


def test_relative_binding():
    """
    Test validate_relative_binding.py to ensure that it functions correctly
    """
    protein_path = str(Path(__file__).resolve().parent / "data" / "hif2a_nowater_min.pdb")
    temp_ligands = get_hif2a_ligands_as_sdf_file(1)

    # Any fewer windows and the lambda schedule validation will fail
    windows = str(5)
    steps = str(2000)
    seed = str(2022)

    output_path = "rabfe_results.sdf"
    cli_args = [
        "--blocker_name",
        "338",
        "--ligand_sdf",
        temp_ligands.name,
        "--protein_pdb",
        protein_path,
        "--num_complex_conv_windows",
        windows,
        "--num_complex_windows",
        windows,
        "--num_solvent_conv_windows",
        windows,
        "--num_solvent_windows",
        windows,
        "--num_solvent_prod_steps",
        steps,
        "--num_complex_prod_steps",
        steps,
        "--output_path",
        output_path,
        "--seed",
        seed,
    ]
    with temporary_working_dir() as temp_dir:
        _ = run_example("validate_relative_binding.py", cli_args, cwd=temp_dir)
        sdf_output = Path(temp_dir) / output_path
        assert sdf_output.is_file()
        for mol in Chem.SDMolSupplier(str(sdf_output), removeHs=False):
            result = RABFEResult.from_mol(mol)
            assert isinstance(result.dG_bind, float)
            assert isinstance(result.dG_bind_err, float)


def test_smc_freesolv():
    """run_smc_on_freesolv.py with reasonable settings on a small subset of FreeSolv, and expect
    * output in summary_smc_result_*.pkl
    * no NaNs in accumulated log weights
    * predictions within 2 kcal/mol of experiment
    """
    # n_mols=2 -> ~10 minutes per mol on a single T4 GPU
    config = dict(n_walkers=100, n_windows=100, n_md_steps=100, n_mols=2, seed=2022)
    cli_args = [f"--{key}={val}" for (key, val) in config.items()]
    temperature = 300

    experimental_dGs = {get_mol_name(mol): float(mol.GetProp("dG")) for mol in fetch_freesolv()}

    def get_predicted_dG(fname):
        """in kcal/mol"""
        with open(fname, "rb") as f:
            smc_result = pickle.load(f)

        # expect no NaNs in incremental log weights
        incremental_log_weights_traj = smc_result["incremental_log_weights_traj"]
        assert np.isfinite(incremental_log_weights_traj).all()

        # compute dG in kcal/mol from final log weights
        log_weights = smc_result["final_log_weights"]
        reduced_dG = -logsumexp(log_weights - np.log(len(log_weights)))
        dG_in_kJmol = reduced_dG * (BOLTZ * temperature)
        dG_in_kcalmol = dG_in_kJmol / 4.184

        return dG_in_kcalmol

    def get_mol_name_from_pkl(fname):
        """in kcal/mol"""
        with open(fname, "rb") as f:
            smc_result = pickle.load(f)
        # Make sure dG properties are preserved
        _ = smc_result["mol"].GetProp("dG")
        return get_mol_name(smc_result["mol"])

    with temporary_working_dir() as temp_dir:
        # expect running this script to write summary_result_result_{mol_name}_*.pkl files
        output_path = str(Path(temp_dir) / "summary_smc_result_*.pkl")
        assert len(glob(output_path)) == 0
        _ = run_example("run_smc_on_freesolv.py", cli_args, cwd=temp_dir)
        smc_result_fnames = glob(output_path)
        assert len(smc_result_fnames) == config["n_mols"]

        # load predictions and experimental values
        dG_preds = []
        dG_expts = []
        for fname in smc_result_fnames:
            dG_preds.append(get_predicted_dG(fname))
            dG_expts.append(experimental_dGs[get_mol_name_from_pkl(fname)])
        dG_preds = np.array(dG_preds)
        dG_expts = np.array(dG_expts)

        # compute error summaries
        mean_abs_err_kcalmol = np.mean(np.abs(dG_preds - dG_expts))

        # expect small error
        # * MAE of ~1.5 kcal/mol: run with these settings on 10 molecules from FreeSolv
        # * MAE of ~1.5 kcal/mol: run with these settings on ~500 molecules from FreeSolv
        # * MAE of ~1.1 kcal/mol: FreeSolv reference calculations
        #   https://www.biorxiv.org/content/10.1101/104281v1.full
        assert mean_abs_err_kcalmol <= 2
