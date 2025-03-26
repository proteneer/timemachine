import hashlib
import os
import pickle
import subprocess
import sys
from glob import glob
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from common import ARTIFACT_DIR_NAME, temporary_working_dir
from numpy.typing import NDArray as Array
from scipy.special import logsumexp

from timemachine.constants import DEFAULT_FF, DEFAULT_KT, KCAL_TO_KJ
from timemachine.fe.free_energy import assert_deep_eq
from timemachine.fe.utils import get_mol_name, read_sdf_mols_by_name
from timemachine.ff import Forcefield
from timemachine.testsystems import fetch_freesolv

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def hash_file(path: Path, chunk_size: int = 2048) -> str:
    assert path.is_file(), f"{path!s} doesn't exist"
    m = hashlib.sha256()
    with open(path, "rb") as ifs:
        chunk = ifs.read(chunk_size)
        while len(chunk) > 0:
            m.update(chunk)
            chunk = ifs.read(chunk_size)
    return m.hexdigest()


def run_example(
    example_name: str, cli_args: list[str], env: Optional[dict[str, str]] = None, cwd: Optional[str] = None
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


def get_cli_args(config: dict) -> list[str]:
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


def get_smc_free_solv_results(result_path: str) -> tuple[Array, Array]:
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


@pytest.mark.nightly
@pytest.mark.parametrize("insertion_type", ["untargeted"])
def test_water_sampling_mc_bulk_water(insertion_type):
    reference_data_path = EXAMPLES_DIR.parent / "tests" / "data" / f"reference_bulk_water_{insertion_type}.npz"
    assert reference_data_path.is_file()
    reference_data = np.load(reference_data_path)
    with resources.as_file(resources.files("timemachine.testsystems.water_exchange")) as water_exchange:
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
            # save_last_frame=reference_data_path, # uncomment me to manually update the data folders.
        )

    with temporary_working_dir() as temp_dir:
        # expect running this script to write summary_result_result_{mol_name}_*.pkl files
        proc = run_example("water_sampling_mc.py", get_cli_args(config), cwd=temp_dir)
        assert proc.returncode == 0
        assert (Path(temp_dir) / str(config["out_cif"])).is_file()
        last_frame = Path(temp_dir) / str(config["save_last_frame"])
        assert last_frame.is_file()

        test_data = np.load(last_frame)
        assert test_data.files == reference_data.files
        for key in reference_data.files:
            np.testing.assert_array_equal(test_data[key], reference_data[key])


@pytest.mark.nightly
@pytest.mark.parametrize("batch_size", [1, 250, 512, 1000])
@pytest.mark.parametrize("insertion_type", ["targeted", "untargeted"])
def test_water_sampling_mc_buckyball(batch_size, insertion_type):
    # Expectations of the test:
    # 1) Different batch_sizes produces identical final frames
    # 2) Different insertion_types produces different final frames, but bitwise identical to a reference final frame.

    reference_data_path = EXAMPLES_DIR.parent / "tests" / "data" / f"reference_6_water_{insertion_type}.npz"
    assert reference_data_path.is_file()
    reference_data = np.load(reference_data_path)

    # setup cli kwargs for the run_example_script
    with resources.as_file(resources.files("timemachine.testsystems.water_exchange")) as water_exchange:
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
            # save_last_frame=reference_data_path, # uncomment me to manually update the data folders.
        )

    with temporary_working_dir() as temp_dir:
        proc = run_example("water_sampling_mc.py", get_cli_args(config), cwd=temp_dir)
        assert proc.returncode == 0
        assert (Path(temp_dir) / str(config["out_cif"])).is_file()
        last_frame = Path(temp_dir) / str(config["save_last_frame"])
        assert last_frame.is_file()
        test_data = np.load(last_frame)
        assert test_data.files == reference_data.files

        for key in reference_data.files:
            np.testing.assert_array_equal(test_data[key], reference_data[key])


@pytest.mark.parametrize(
    "leg, n_windows, n_frames, n_eq_steps",
    [
        ("vacuum", 6, 50, 1000),
        pytest.param("solvent", 5, 50, 1000, marks=pytest.mark.nightly),
        pytest.param("complex", 5, 50, 1000, marks=pytest.mark.nightly),
    ],
)
@pytest.mark.parametrize("mol_a, mol_b", [("15", "30")])
@pytest.mark.parametrize("seed", [2025])
def test_run_rbfe_legs(
    leg,
    n_windows,
    n_frames,
    n_eq_steps,
    mol_a,
    mol_b,
    seed,
):
    # To update the leg result hashes, refer to the hashes generated from CI runs.
    # The CI jobs produce an artifact for the results stored at ARTIFACT_DIR_NAME
    # which can be used to investigate the results that generated the hashes.
    # Hashes are of results.npz, lambda0_traj.npz and lambda1_traj.npz respectively.
    leg_results_hashes = {
        "vacuum": (
            "ca784fd5ebfbc562f301e2153f108ad0e8b7b765f0fc305cb0e2b13004929519",
            "551f363c55f439a1003d7170a6fb7a6f04a032a3185ef332ad00e34c8b8526a8",
            "315f295bdb702e837b8ac8b28d346de9c6b74c13a4bbbb178de0ea8ec8fca775",
        ),
        "solvent": (
            "9ccc4cb16ea43c0eed4941a3485a591055ca9e68f30bdc6ae49113ca247ddbd4",
            "5c0f5fea16e7fad695fe48c4d7e9400de1ee081616e66d81c13cff975275785d",
            "18478e891fe7e5448df89c10e8eedba32f86bdf572a01749c63957da0dc315f6",
        ),
        "complex": (
            "8f79bac368d05fb58fbd3a2f0fa0a0acb60b63e65351b3f94447de3b025865e6",
            "9e66f5219a9d27331e5dbe38d458b6092dcdf15d1e9d11ff1595b40a46daa91a",
            "a30c101c16373e17a10c062c3ab63fe6ee29d5d21e58c7f445361fc8be520848",
        ),
    }
    with resources.as_file(resources.files("timemachine.testsystems.fep_benchmark.hif2a")) as hif2a_dir:
        config = dict(
            mol_a=mol_a,
            mol_b=mol_b,
            sdf_path=hif2a_dir / "ligands.sdf",
            pdb_path=hif2a_dir / "5tbm_prepared.pdb",
            seed=seed,
            legs=leg,
            n_eq_steps=n_eq_steps,
            n_frames=n_frames,
            n_windows=n_windows,
            forcefield=DEFAULT_FF,
            output_dir=f"{ARTIFACT_DIR_NAME}/rbfe_{mol_a}_{mol_b}_{leg}_{seed}",
        )

        def verify_run(output_dir: Path):
            assert output_dir.is_dir()
            mols_by_name = read_sdf_mols_by_name(output_dir / "mols.sdf")
            assert len(mols_by_name) == 2
            assert mol_a in mols_by_name
            assert mol_b in mols_by_name
            assert (output_dir / "md_params.pkl").is_file()
            assert (output_dir / "atom_mapping.svg").is_file()
            assert (output_dir / "core.pkl").is_file()
            assert (output_dir / "ff.py").is_file()

            assert Forcefield.load_from_file(output_dir / "ff.py") is not None

            leg_dir = output_dir / leg
            assert leg_dir.is_dir()
            assert (leg_dir / "results.npz").is_file()
            assert (leg_dir / "lambda0_traj.npz").is_file()
            assert (leg_dir / "lambda1_traj.npz").is_file()

            assert (leg_dir / "simulation_result.pkl").is_file()
            if leg in ["solvent", "complex"]:
                assert (leg_dir / "host_config.pkl").is_file()
            else:
                assert not (leg_dir / "host_config.pkl").is_file()
            assert (leg_dir / "hrex_transition_matrix.png").is_file()
            assert (leg_dir / "hrex_swap_acceptance_rates_convergence.png").is_file()
            assert (leg_dir / "hrex_replica_state_distribution_heatmap.png").is_file()
            if leg == "complex":
                assert (leg_dir / "water_sampling_acceptances.png").is_file()

            results = np.load(str(leg_dir / "results.npz"))
            assert results["pred_dg"].size == 1
            assert results["pred_dg"].dtype == np.float64
            assert results["pred_dg"] != 0.0

            assert results["pred_dg_err"].size == 1
            assert results["pred_dg_err"].dtype == np.float64
            assert results["pred_dg_err"] != 0.0

            assert results["n_windows"].size == 1
            assert results["n_windows"].dtype == np.intp
            assert 2 <= results["n_windows"] <= config["n_windows"]
            assert isinstance(results["overlaps"], np.ndarray)
            assert all(isinstance(overlap, float) for overlap in results["overlaps"])

            for lamb in [0, 1]:
                traj_data = np.load(str(leg_dir / f"lambda{lamb:d}_traj.npz"))
                assert len(traj_data["coords"]) == n_frames
                assert len(traj_data["boxes"]) == n_frames

        def verify_leg_results_hashes(output_dir: Path):
            leg_dir = output_dir / leg
            results_hash = hash_file(leg_dir / "results.npz")
            endstate_0_hash = hash_file(leg_dir / "lambda0_traj.npz")
            endstate_1_hash = hash_file(leg_dir / "lambda1_traj.npz")
            assert (results_hash, endstate_0_hash, endstate_1_hash) == leg_results_hashes[leg]

        config_a = config.copy()
        config_a["output_dir"] = config["output_dir"] + "_a"
        proc = run_example("run_rbfe_legs.py", get_cli_args(config_a))
        assert proc.returncode == 0
        verify_run(Path(config_a["output_dir"]))
        verify_leg_results_hashes(Path(config_a["output_dir"]))

        config_b = config.copy()
        config_b["output_dir"] = config["output_dir"] + "_b"
        assert config_b["output_dir"] != config_a["output_dir"], "Runs are writing to the same output directory"
        proc = run_example("run_rbfe_legs.py", get_cli_args(config_b))
        assert proc.returncode == 0
        verify_run(Path(config_b["output_dir"]))

        def verify_simulations_match(ref_dir: Path, comp_dir: Path):
            with open(ref_dir / "md_params.pkl", "rb") as ifs:
                ref_md_params = pickle.load(ifs)
            with open(comp_dir / "md_params.pkl", "rb") as ifs:
                comp_md_params = pickle.load(ifs)
            assert ref_md_params == comp_md_params, "MD Parameters don't match"

            with open(ref_dir / "core.pkl", "rb") as ifs:
                ref_core = pickle.load(ifs)
            with open(comp_dir / "core.pkl", "rb") as ifs:
                comp_core = pickle.load(ifs)
            assert np.all(ref_core == comp_core), "Atom mappings don't match"

            ref_results = np.load(str(ref_dir / leg / "results.npz"))
            comp_results = np.load(str(comp_dir / leg / "results.npz"))
            np.testing.assert_equal(ref_results["pred_dg"], comp_results["pred_dg"])
            np.testing.assert_equal(ref_results["pred_dg_err"], comp_results["pred_dg_err"])
            np.testing.assert_array_equal(ref_results["overlaps"], comp_results["overlaps"])
            np.testing.assert_equal(ref_results["n_windows"], comp_results["n_windows"])

            with open(ref_dir / leg / "simulation_result.pkl", "rb") as ifs:
                ref_res = pickle.load(ifs)
            with open(comp_dir / leg / "simulation_result.pkl", "rb") as ifs:
                comp_res = pickle.load(ifs)
            assert len(ref_res.final_result.initial_states) == ref_results["n_windows"]
            assert len(ref_res.final_result.initial_states) == len(comp_res.final_result.initial_states)

            for ref_state, comp_state in zip(ref_res.final_result.initial_states, comp_res.final_result.initial_states):
                np.testing.assert_array_equal(ref_state.x0, comp_state.x0)
                np.testing.assert_array_equal(ref_state.v0, comp_state.v0)
                np.testing.assert_array_equal(ref_state.box0, comp_state.box0)
                np.testing.assert_array_equal(ref_state.ligand_idxs, comp_state.ligand_idxs)
                np.testing.assert_array_equal(ref_state.protein_idxs, comp_state.protein_idxs)
                assert_deep_eq(ref_state.potentials, comp_state.potentials)

            for lamb in [0, 1]:
                ref_traj = np.load(str(ref_dir / leg / f"lambda{lamb}_traj.npz"))
                comp_traj = np.load(str(comp_dir / leg / f"lambda{lamb}_traj.npz"))
                np.testing.assert_array_equal(ref_traj["coords"], comp_traj["coords"])
                np.testing.assert_array_equal(ref_traj["boxes"], comp_traj["boxes"])

        verify_simulations_match(Path(config_a["output_dir"]), Path(config_b["output_dir"]))
