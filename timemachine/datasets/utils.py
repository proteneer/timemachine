from importlib import resources
from typing import List

from rdkit import Chem


def fetch_freesolv() -> List[Chem.Mol]:
    with resources.path("timemachine.datasets.freesolv", "freesolv.sdf") as freesolv_path:
        supplier = Chem.SDMolSupplier(str(freesolv_path), removeHs=False)
    return [mol for mol in supplier]
