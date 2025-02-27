from typing import Optional

from rdkit import Chem

from timemachine.fe.utils import get_mol_name, read_sdf
from timemachine.utils import path_to_internal_file


def fetch_freesolv(n_mols: Optional[int] = None, exclude_mols: Optional[set[str]] = None) -> list[Chem.Mol]:
    """
    Return the (potentially truncated) FreeSolv data set.

    Parameters
    ----------
    n_mols:
        Limit to this number of mols.
        Default of None means to keep all of the molecules.

    exclude_mols:
        Exclude molecules in the given set.

    """
    with path_to_internal_file("timemachine.testsystems.freesolv", "freesolv.sdf") as freesolv_path:
        mols = read_sdf(str(freesolv_path))

    # filter and truncate
    exclude_mols = exclude_mols or set()
    filtered_mols = [mol for mol in mols if get_mol_name(mol) not in exclude_mols]
    first_n_filtered_mols = filtered_mols[:n_mols]

    return first_n_filtered_mols
