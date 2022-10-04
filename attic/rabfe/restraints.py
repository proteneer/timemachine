import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from timemachine.fe.utils import get_romol_conf

def setup_relative_restraints_by_distance(
    mol_a: Chem.Mol, mol_b: Chem.Mol, cutoff: float = 0.1, terminal: bool = False
):
    """
    Setup restraints between atoms in two molecules using
    a cutoff distance between atoms

    Parameters
    ----------
    mol_a: Chem.Mol
        First molecule

    mol_b: Chem.Mol
        Second molecule

    cutoff: float=0.1
        Distance between atoms to consider as a match

    terminal: bool=false
        Map terminal atoms

    Returns
    -------
    np.array (N, 2)
        Atom mapping between atoms in mol_a to atoms in mol_b.
    """

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)
    core_idxs_a = []
    core_idxs_b = []

    for idx, a in enumerate(mol_a.GetAtoms()):
        if not terminal and a.GetDegree() == 1:
            continue
        for b_idx, b in enumerate(mol_b.GetAtoms()):
            if not terminal and b.GetDegree() == 1:
                continue
            if np.linalg.norm(ligand_coords_a[idx] - ligand_coords_b[b_idx]) < cutoff:
                core_idxs_a.append(idx)
                core_idxs_b.append(b_idx)
    assert len(core_idxs_a) == len(core_idxs_b), "Core sizes were inconsistent"

    rij = cdist(ligand_coords_a[core_idxs_a], ligand_coords_b[core_idxs_b])

    row_idxs, col_idxs = linear_sum_assignment(rij)

    core_idxs = np.array(
        [(core_idxs_a[core_a], core_idxs_b[core_b]) for core_a, core_b in zip(row_idxs, col_idxs)],
        dtype=np.int32,
    )

    return core_idxs