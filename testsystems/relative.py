# construct a relative transformation

import numpy as np
from rdkit import Chem

from fe import free_energy, topology
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from pathlib import Path

root = Path(__file__).parent.parent

path_to_protein = str(root.joinpath('tests/data/hif2a_nowater_min.pdb'))


def _setup_hif2a_ligand_pair(ff='ff/params/smirnoff_1_1_0_ccc.py'):
    """Manually constructed atom map

    TODO: replace this with a testsystem class similar to those used in openmmtools
    """
    path_to_ligand = str(root.joinpath('tests/data/ligands_40.sdf'))
    path_to_ff = str(root.joinpath(ff))

    with open(path_to_ff) as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)

    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)
    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    core = np.array([[0, 0],
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
                     [21, 22]]
                    )

    single_topology = topology.SingleTopology(mol_a, mol_b, core, forcefield)
    rfe = free_energy.RelativeFreeEnergy(single_topology)

    return rfe

hif2a_ligand_pair = _setup_hif2a_ligand_pair()
