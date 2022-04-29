from importlib import resources
from typing import List, Optional, Set

from rdkit import Chem

from timemachine.fe.utils import get_mol_name


def fetch_freesolv(n_mols: Optional[int] = None, exclude_mols: Optional[Set[str]] = None) -> List[Chem.Mol]:
    """
    Return the (potentially filtered) version of the free solv data set.

    Parameters
    ----------
    n_mols:
        Limit to this number of mols. Default of None
        means to keep all of the molecules.

    exclude_mols:
        Filter molecules in the given set.

    """
    exclude_mols = exclude_mols or set()
    with resources.path("timemachine.datasets.freesolv", "freesolv.sdf") as freesolv_path:
        supplier = Chem.SDMolSupplier(str(freesolv_path), removeHs=False)
    return [mol for mol in supplier if get_mol_name(mol) not in exclude_mols][:n_mols]
