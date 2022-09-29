# construct a relative transformation

from importlib import resources

import numpy as np
from rdkit import Chem


def get_hif2a_ligand_pair_single_topology():
    """Return two ligands from hif2a and the manually specified atom mapping"""

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    core = np.array(
        [
            [0, 0],
            [2, 2],
            [1, 1],
            [6, 6],
            [5, 5],
            [4, 4],
            [3, 3],
            [15, 16],
            [16, 17],
            [17, 18],
            [18, 19],
            [19, 20],
            [20, 21],
            [32, 30],
            [26, 25],
            [27, 26],
            [7, 7],
            [8, 8],
            [9, 9],
            [10, 10],
            [29, 11],
            [11, 12],
            [12, 13],
            [14, 15],
            [31, 29],
            [13, 14],
            [23, 24],
            [30, 28],
            [28, 27],
            [21, 22],
        ]
    )
    return mol_a, mol_b, core
