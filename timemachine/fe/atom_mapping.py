from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

from timemachine.fe.topology import AtomMappingError


class CompareDistNonterminal(rdFMCS.MCSAtomCompare):
    """
    Custom comparator used in the FMCS code.
    This allows two atoms to match if:
        1. Neither atom is a terminal atom (H, F, Cl, Halogens etc.)
        2. They are within 1 angstrom of each other.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compare(self, p, mol1, atom1, mol2, atom2):

        if mol1.GetAtomWithIdx(atom1).GetDegree() == 1:
            return False
        if mol2.GetAtomWithIdx(atom2).GetDegree() == 1:
            return False

        x_i = mol1.GetConformer(0).GetPositions()[atom1]
        x_j = mol2.GetConformer(0).GetPositions()[atom2]

        threshold = 1.0  # angstroms
        return np.linalg.norm(x_i - x_j) <= threshold


def mcs_map(a, b, threshold: float = 2.0, timeout: int = 5, smarts: Optional[str] = None):
    """Find the MCS map of going from A to B"""
    params = rdFMCS.MCSParameters()
    params.BondCompareParameters.CompleteRingsOnly = 1
    params.BondCompareParameters.RingMatchesRingOnly = 1
    params.BondTyper = rdFMCS.BondCompare.CompareAny

    params.AtomCompareParameters.CompleteRingsOnly = 1
    params.AtomCompareParameters.RingMatchesRingOnly = 1
    params.AtomCompareParameters.matchValences = 0
    params.AtomCompareParameters.MaxDistance = threshold
    params.AtomTyper = rdFMCS.AtomCompare.CompareAny

    params.Timeout = timeout
    if smarts is not None:
        params.InitialSeed = smarts
    return rdFMCS.FindMCS([a, b], params)


def mcs_map_graph_only_complete_rings(a, b, timeout: int = 3600, smarts: Optional[str] = None):
    """Find the MCS map of going from A to B, disregarding conformer information. This also ensures
    that core-core bonds are not broken."""
    return rdFMCS.FindMCS(
        [a, b],
        maximizeBonds=True,
        timeout=timeout,
        matchValences=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        matchChiralTag=False,
        atomCompare=Chem.rdFMCS.AtomCompare.CompareAny,
        bondCompare=Chem.rdFMCS.BondCompare.CompareAny,
    )


def _check_core_map_distances(mol_a, mol_b, core, threshold=0.5) -> bool:
    """compute vector of distances[i] = distance(conf_a[core_a[i]], conf_b[core_b[i]]),
    check whether distances[i] <= threshold for all i"""

    a, b = core[:, 0], core[:, 1]
    conf_a = mol_a.GetConformer(0).GetPositions()
    conf_b = mol_b.GetConformer(0).GetPositions()
    distances = np.linalg.norm(conf_a[a] - conf_b[b], axis=1)
    return (distances <= threshold).all()


def get_core_by_mcs(mol_a, mol_b, query, threshold=0.5):
    """Return np integer array that can be passed to RelativeFreeEnergy constructor

    Parameters
    ----------
    mol_a, mol_b, query : RDKit molecules
    threshold : float, in angstroms

    Returns
    -------
    core : np.ndarray of ints, shape (n_MCS, 2)

    Notes
    -----
    * Warning! Some atoms that intuitively should be mapped together are not,
        when threshold=0.5 Å in custom atom comparison, because conformers aren't
        quite aligned enough.
    * Warning! Because of the intermediate representation of a substructure query,
        the core indices can get flipped around,
        for example if the substructure match hits only part of an aromatic ring.

        In some cases, this means that pairs of atoms that do not satisfy the
        atom comparison function can be mapped together.
    """

    # fetch conformer, assumed aligned
    conf_a = mol_a.GetConformer(0).GetPositions()
    conf_b = mol_b.GetConformer(0).GetPositions()

    # note that >1 match possible here -- must pick minimum-cost match
    matches_a = mol_a.GetSubstructMatches(query, uniquify=False)
    matches_b = mol_b.GetSubstructMatches(query, uniquify=False)

    cost = np.zeros((len(matches_a), len(matches_b)))
    for i, a in enumerate(matches_a):
        for j, b in enumerate(matches_b):
            # if a single pair is outside of threshold, we set the cost to inf
            dij = np.linalg.norm(conf_a[np.array(a)] - conf_b[np.array(b)], axis=1)
            cost[i, j] = dij.sum() if np.all(dij < threshold) else np.inf

    # find (i,j) = argmin cost
    min_i, min_j = np.unravel_index(np.argmin(cost, axis=None), cost.shape)

    # concatenate into (n_atoms, 2) array
    inds_a, inds_b = matches_a[min_i], matches_b[min_j]
    core = np.array([inds_a, inds_b]).T

    if not _check_core_map_distances(mol_a, mol_b, core, threshold):
        raise AtomMappingError(f"not all mapped atoms are within {threshold:.3f}Å of each other")

    return core
