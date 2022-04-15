from jax.config import config

config.update("jax_enable_x64", True)

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


def setup_relative_restraints_using_smarts(mol_a, mol_b, smarts):
    """
    Setup restraints between atoms in two molecules using
    a pre-defined SMARTS pattern.

    Parameters
    ----------
    mol_a: Chem.Mol
        First molecule

    mol_b: Chem.Mol
        Second molecule

    smarts: string
        Smarts pattern defining the common core.

    Returns
    -------
    np.array (N, 2)
        Atom mapping between atoms in mol_a to atoms in mol_b.

    """

    # check to ensure the core is connected
    # technically allow for this but we need to do more validation before
    # we can be fully comfortable
    assert "." not in smarts

    core = Chem.MolFromSmarts(smarts)

    # we want *all* possible combinations.
    limit = 1000
    all_core_idxs_a = np.array(mol_a.GetSubstructMatches(core, uniquify=False, maxMatches=limit))
    all_core_idxs_b = np.array(mol_b.GetSubstructMatches(core, uniquify=False, maxMatches=limit))

    assert len(all_core_idxs_a) < limit
    assert len(all_core_idxs_b) < limit

    best_rmsd = np.inf
    best_core_idxs_a = None
    best_core_idxs_b = None

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)

    # setup relative orientational restraints
    # rough sketch of algorithm:
    # find core atoms in mol_a
    # find core atoms in mol_b
    # for all matches in mol_a
    #    for all matches in mol_b
    #       use the hungarian algorithm to assign matching
    #       if sum is smaller than best, then store.

    for core_idxs_a in all_core_idxs_a:
        for core_idxs_b in all_core_idxs_b:

            ri = np.expand_dims(ligand_coords_a[core_idxs_a], 1)
            rj = np.expand_dims(ligand_coords_b[core_idxs_b], 0)
            rij = np.sqrt(np.sum(np.power(ri - rj, 2), axis=-1))

            row_idxs, col_idxs = linear_sum_assignment(rij)

            rmsd = np.linalg.norm(ligand_coords_a[core_idxs_a[row_idxs]] - ligand_coords_b[core_idxs_b[col_idxs]])

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_core_idxs_a = core_idxs_a
                best_core_idxs_b = core_idxs_b

    core_idxs = np.stack([best_core_idxs_a, best_core_idxs_b], axis=1).astype(np.int32)
    print("core_idxs", core_idxs, "rmsd", best_rmsd)

    return core_idxs
