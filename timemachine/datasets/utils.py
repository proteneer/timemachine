from importlib import resources
from typing import List

from rdkit import Chem


def fetch_freesolv() -> List[Chem.Mol]:
    with resources.path("timemachine", "datasets") as datasets_path:
        freesolv_path = str(datasets_path / "freesolv" / "freesolv.sdf")
        supplier = Chem.SDMolSupplier(freesolv_path, removeHs=False)
        return [mol for mol in supplier]
