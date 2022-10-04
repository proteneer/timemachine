import csv
import os
import pickle
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from common import temporary_working_dir
from numpy.typing import NDArray as Array
from scipy.special import logsumexp

from timemachine.constants import DEFAULT_KT, KCAL_TO_KJ
from timemachine.datasets import fetch_freesolv
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
    print("Running with args:", "".join(subprocess_args))
    proc = subprocess.run(
        subprocess_args,
        env=subprocess_env,
        check=True,
        cwd=cwd,
    )
    return proc


def get_cli_args(config: Dict) -> List[str]:
    return [(f"--{key}={val}" if val is not None else f"--{key}") for (key, val) in config.items()]


@pytest.fixture(scope="module")
def smc_free_solv_path():
    # n_mols=2 -> ~10 minutes per mol on a single T4 GPU
    config = dict(n_walkers=100, n_windows=100, n_md_steps=100, n_mols=2, seed=2022, n_gpus=1)
    with temporary_working_dir() as temp_dir:
        # expect running this script to write summary_result_result_{mol_name}_*.pkl files
        output_path = str(Path(temp_dir) / "summary_smc_result_*.pkl")
        assert len(glob(output_path)) == 0
        _ = run_example("run_smc_on_freesolv.py", get_cli_args(config), cwd=temp_dir)
        yield temp_dir


def get_smc_free_solv_results(result_path: str) -> Tuple[Array, Array]:
    # return the dG_preds, dG_expts for the given free solv run
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
        dG_in_kJmol = reduced_dG * DEFAULT_KT
        dG_in_kcalmol = dG_in_kJmol / KCAL_TO_KJ

        return dG_in_kcalmol

    def get_mol_name_from_pkl(fname):
        """in kcal/mol"""
        with open(fname, "rb") as f:
            smc_result = pickle.load(f)
        # Make sure dG properties are preserved
        _ = smc_result["mol"].GetProp("dG")
        return get_mol_name(smc_result["mol"])

    output_path = str(Path(result_path) / "summary_smc_result_*.pkl")
    smc_result_fnames = glob(output_path)
    assert len(smc_result_fnames) == 2

    # load predictions and experimental values
    dG_preds = []
    dG_expts = []
    for fname in smc_result_fnames:
        dG_preds.append(get_predicted_dG(fname))
        dG_expts.append(experimental_dGs[get_mol_name_from_pkl(fname)])
    dG_preds = np.array(dG_preds)
    dG_expts = np.array(dG_expts)
    return dG_preds, dG_expts


def test_smc_freesolv(smc_free_solv_path):
    """run_smc_on_freesolv.py with reasonable settings on a small subset of FreeSolv, and expect
    * output in summary_smc_result_*.pkl
    * no NaNs in accumulated log weights
    * predictions within 2 kcal/mol of experiment
    """
    dG_preds, dG_expts = get_smc_free_solv_results(smc_free_solv_path)
    # compute error summaries
    mean_abs_err_kcalmol = np.mean(np.abs(dG_preds - dG_expts))
    print(dG_preds, dG_expts, mean_abs_err_kcalmol)

    # expect small error
    # * MAE of ~1.5 kcal/mol: run with these settings on 10 molecules from FreeSolv
    # * MAE of ~1.5 kcal/mol: run with these settings on ~500 molecules from FreeSolv
    # * MAE of ~1.1 kcal/mol: FreeSolv reference calculations
    #   https://www.biorxiv.org/content/10.1101/104281v1.full
    assert mean_abs_err_kcalmol <= 2


def test_smc_freesolv_fit(smc_free_solv_path):
    """
    refit_freesolv given the free solv results on two molecules.
    Expect the fit MAE to be low, since we are fitting on all of the data.
    Also test the loss-only mode which should give similar results.
    """
    dG_preds, dG_expts = get_smc_free_solv_results(smc_free_solv_path)
    mean_abs_err_kcalmol_orig = np.mean(np.abs(dG_preds - dG_expts))

    def read_dgs():
        dG_preds = []
        dG_expts = []
        with open("fit_pred_dg.csv", "r") as f:
            for row in csv.DictReader(f):
                # Convert to kcal/mol to be consistent with above test
                dG_preds.append(float(row["pred_dg (kJ/mol)"]) / KCAL_TO_KJ)
                dG_expts.append(float(row["exp_dg (kJ/mol)"]) / KCAL_TO_KJ)
        dG_preds = np.array(dG_preds)
        dG_expts = np.array(dG_expts)
        return dG_preds, dG_expts

    with temporary_working_dir() as temp_dir:
        config = dict(result_path=smc_free_solv_path, n_mols=2, n_gpus=1)
        run_example("refit_freesolv.py", get_cli_args(config), cwd=temp_dir)
        dG_preds, dG_expts = read_dgs()

        fit_ff_fname = "fit_ffld_all_final.py"
        assert Path(fit_ff_fname).exists()

        config = dict(result_path=smc_free_solv_path, loss_only=None, ff_refit=fit_ff_fname, n_mols=2, n_gpus=1)
        run_example("refit_freesolv.py", get_cli_args(config), cwd=temp_dir)
        dG_preds_lo, dG_expts_lo = read_dgs()

    mean_abs_err_kcalmol_fit = np.mean(np.abs(dG_preds - dG_expts))
    assert mean_abs_err_kcalmol_fit < mean_abs_err_kcalmol_orig
    assert mean_abs_err_kcalmol_fit < 0.1

    mean_abs_err_kcalmol_lo = np.mean(np.abs(dG_preds_lo - dG_expts_lo))
    assert mean_abs_err_kcalmol_fit == pytest.approx(mean_abs_err_kcalmol_lo, abs=1e-1)
