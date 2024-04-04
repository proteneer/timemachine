import os
import pickle
import subprocess
import sys
from contextlib import contextmanager
from glob import glob
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from common import temporary_working_dir
from numpy.typing import NDArray as Array
from scipy.special import logsumexp

from timemachine.constants import DEFAULT_FF, DEFAULT_KT, KCAL_TO_KJ
from timemachine.datasets import fetch_freesolv
from timemachine.fe import rbfe
from timemachine.fe.free_energy import PairBarResult, SimulationResult
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
    print("Running with args:", " ".join(subprocess_args))
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
    _dG_preds = []
    _dG_expts = []
    for fname in smc_result_fnames:
        _dG_preds.append(get_predicted_dG(fname))
        _dG_expts.append(experimental_dGs[get_mol_name_from_pkl(fname)])
    dG_preds = np.array(_dG_preds)
    dG_expts = np.array(_dG_expts)
    return dG_preds, dG_expts


@pytest.mark.skip("needs update since removal of lambda dependence in nonbonded potentials")
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


@contextmanager
def get_rbfe_edge_list_hif2a_path(seed):
    with resources.as_file(resources.files("timemachine.datasets.fep_benchmark.hif2a")) as hif2a_data:
        base_config = dict(
            n_frames=2,
            ligands=hif2a_data / "ligands.sdf",
            forcefield=DEFAULT_FF,
            protein=hif2a_data / "5tbm_prepared.pdb",
            n_gpus=1,
            seed=seed,
            n_eq_steps=2,
            n_windows=3,
        )

        def run(results_csv, temp_dir):
            output_path = str(Path(temp_dir) / rbfe.get_success_result_path("*", "*"))
            assert len(glob(output_path)) == 0
            config = dict(results_csv=results_csv, **base_config)
            _ = run_example("rbfe_edge_list.py", get_cli_args(config), cwd=temp_dir)
            return Path(temp_dir)

        with open(hif2a_data / "results_edges_5ns.csv", "r") as fp:
            edges_rows = fp.readlines()

        edges_rows_sample = edges_rows[:3]  # keep header and first 2 edges (338 -> 165, 338 -> 215)
        edges = [("338", "165"), ("338", "215")]

        with temporary_working_dir() as temp_dir:
            with (Path(temp_dir) / "edges.csv").open("w") as fp:
                fp.writelines(edges_rows_sample)
                fp.flush()
                path = run(fp.name, temp_dir)
            yield path, base_config, edges


DEFAULT_SEED = 2023


@pytest.fixture(scope="module")
def rbfe_edge_list_hif2a_path():
    with get_rbfe_edge_list_hif2a_path(DEFAULT_SEED) as r:
        yield r


def load_simulation_results(path: Path) -> Tuple[SimulationResult, SimulationResult]:
    with path.open("rb") as fp:
        results = pickle.load(fp)

    (
        _,  # mol_a
        _,  # mol_b
        _,  # edge_metadata
        _,  # core,
        solvent_res,
        _,  # solvent_top,
        complex_res,
        _,  # complex_top,
    ) = results

    return solvent_res, complex_res


def test_rbfe_edge_list_hif2a(rbfe_edge_list_hif2a_path):
    path, config, edges = rbfe_edge_list_hif2a_path
    n_windows = config["n_windows"]

    def check_results(results_path):
        # Just check that results are present and have the expected shape
        # (we don't do enough sampling here for statistical checks)

        # NOTE: We're mainly interested in checking that simulation frames have been serialized properly; we already
        # have more exhaustive checks in test_relative_free_energy.py

        assert results_path.exists()
        solvent_res, complex_res = load_simulation_results(results_path)

        for result in solvent_res, complex_res:
            L = len(result.final_result.initial_states)
            assert L == n_windows

            assert isinstance(result, SimulationResult)
            assert isinstance(result.frames, list)
            assert len(result.frames) == n_windows  # frames from first and last windows
            for frames in result.frames:
                assert len(frames) == config["n_frames"]

            N, _ = result.frames[0][0].shape
            assert N > 0

            for frames in result.frames:
                for frame in frames:
                    assert frame.ndim == 2
                    assert frame.shape == (N, 3)

    for mol_a_name, mol_b_name in edges:
        check_results(path / rbfe.get_success_result_path(mol_a_name, mol_b_name))


def assert_simulation_results_equal(r1: SimulationResult, r2: SimulationResult):
    def assert_pair_bar_results_equal(p1: PairBarResult, p2: PairBarResult):
        np.testing.assert_array_equal(p1.dGs, p2.dGs)
        np.testing.assert_array_equal(p1.dG_errs, p2.dG_errs)
        np.testing.assert_array_equal(p1.dG_err_by_component_by_lambda, p2.dG_err_by_component_by_lambda)
        np.testing.assert_array_equal(p1.overlaps, p2.overlaps)
        np.testing.assert_array_equal(p1.overlap_by_component_by_lambda, p2.overlap_by_component_by_lambda)
        np.testing.assert_array_equal(p1.u_kln_by_component_by_lambda, p2.u_kln_by_component_by_lambda)

    assert_pair_bar_results_equal(r1.final_result, r2.final_result)

    for p1, p2 in zip(r1.intermediate_results, r2.intermediate_results):
        assert_pair_bar_results_equal(p1, p2)


def test_rbfe_edge_list_reproducible(rbfe_edge_list_hif2a_path):
    path1, _, edges = rbfe_edge_list_hif2a_path

    with get_rbfe_edge_list_hif2a_path(DEFAULT_SEED) as (path2, _, _):
        with get_rbfe_edge_list_hif2a_path(DEFAULT_SEED + 1) as (path3, _, _):
            for mol_a_name, mol_b_name in edges:

                def load_results(dir):
                    path = dir / rbfe.get_success_result_path(mol_a_name, mol_b_name)
                    assert path.exists()
                    return load_simulation_results(path)

                solvent_res_1, complex_res_1 = load_results(path1)
                solvent_res_2, complex_res_2 = load_results(path2)
                solvent_res_3, complex_res_3 = load_results(path3)

                # results at path2 should be bitwise equivalent to those at path1
                assert_simulation_results_equal(solvent_res_1, solvent_res_2)
                assert_simulation_results_equal(complex_res_1, complex_res_2)

                # results at path3 should differ from those at path1 and path2
                with pytest.raises(AssertionError):
                    assert_simulation_results_equal(solvent_res_1, solvent_res_3)
                with pytest.raises(AssertionError):
                    assert_simulation_results_equal(complex_res_1, complex_res_3)


@pytest.mark.parametrize("insertion_type", ["untargeted"])
def test_water_sampling_mc_bulk_water(insertion_type):
    reference_data_path = EXAMPLES_DIR.parent / "tests" / "data" / f"reference_bulk_water_{insertion_type}.npz"
    assert reference_data_path.is_file()
    reference_data = np.load(reference_data_path)
    with resources.as_file(resources.files("timemachine.datasets.water_exchange")) as water_exchange:
        config = dict(
            out_cif="bulk.cif",
            water_pdb=water_exchange / "bb_0_waters.pdb",
            iterations=5,
            md_steps_per_batch=1000,
            mc_steps_per_batch=1000,
            equilibration_steps=5000,
            insertion_type=insertion_type,
            use_hmr=1,
            save_last_frame="comp_frame.npz",
        )

    with temporary_working_dir() as temp_dir:
        # expect running this script to write summary_result_result_{mol_name}_*.pkl files
        proc = run_example("water_sampling_mc.py", get_cli_args(config), cwd=temp_dir)
        assert proc.returncode == 0
        assert (Path(temp_dir) / config["out_cif"]).is_file()
        last_frame = Path(temp_dir) / config["save_last_frame"]
        assert last_frame.is_file()

        test_data = np.load(last_frame)
        assert test_data.files == reference_data.files
        for key in reference_data.files:
            np.testing.assert_array_equal(test_data[key], reference_data[key])


@pytest.mark.parametrize("batch_size", [1, 250, 512, 1000])
@pytest.mark.parametrize("insertion_type", ["targeted", "untargeted"])
def test_water_sampling_mc_buckyball(batch_size, insertion_type):
    reference_data_path = EXAMPLES_DIR.parent / "tests" / "data" / f"reference_6_water_{insertion_type}.npz"
    assert reference_data_path.is_file()
    reference_data = np.load(reference_data_path)
    with resources.as_file(resources.files("timemachine.datasets.water_exchange")) as water_exchange:
        config = dict(
            out_cif="bulk.cif",
            water_pdb=water_exchange / "bb_6_waters.pdb",
            ligand_sdf=water_exchange / "bb_centered_espaloma.sdf",
            iterations=50,
            md_steps_per_batch=1000,
            mc_steps_per_batch=5000,
            equilibration_steps=5000,
            insertion_type=insertion_type,
            use_hmr=1,
            batch_size=batch_size,
            save_last_frame="comp_frame.npz",
        )
    with temporary_working_dir() as temp_dir:
        # expect running this script to write summary_result_result_{mol_name}_*.pkl files
        proc = run_example("water_sampling_mc.py", get_cli_args(config), cwd=temp_dir)
        assert proc.returncode == 0
        assert (Path(temp_dir) / config["out_cif"]).is_file()
        last_frame = Path(temp_dir) / config["save_last_frame"]
        assert last_frame.is_file()

        test_data = np.load(last_frame)
        assert test_data.files == reference_data.files
        for key in reference_data.files:
            np.testing.assert_array_equal(test_data[key], reference_data[key])
