import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from common import get_hif2a_ligands_as_sdf_file, temporary_working_dir
from rdkit import Chem

from timemachine.fe.free_energy import RABFEResult

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
