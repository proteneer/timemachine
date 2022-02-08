from pathlib import Path
from typing import List

from rdkit import Chem

import timemachine

PATH_TO_FREESOLV = str(Path(timemachine.__file__).parent / "datasets/freesolv/freesolv.sdf")


def fetch_freesolv() -> List[Chem.Mol]:
    supplier = Chem.SDMolSupplier(PATH_TO_FREESOLV, removeHs=False)
    return [mol for mol in supplier]
