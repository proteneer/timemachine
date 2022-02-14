from importlib import resources
from typing import List

from rdkit import Chem


def fetch_freesolv() -> List[Chem.Mol]:
    freesolv_path = resources.files(__package__) / "freesolv" / "freesolv.sdf"
    supplier = Chem.SDMolSupplier(str(freesolv_path), removeHs=False)
    return [mol for mol in supplier]
