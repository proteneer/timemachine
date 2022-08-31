from copy import deepcopy
from functools import partial
from typing import Any, Callable, Optional

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
        return bool(np.linalg.norm(x_i - x_j) <= threshold)


SmartsString = str
MCSResult = Any  # Chem.rdFMCS.MCSResult
MCSFxn = Callable[[Chem.Mol, Chem.Mol, Optional[SmartsString]], MCSResult]


def possibly_fallback_to_heavy_atom_mcs(
    mcs_fxn: MCSFxn, a: Chem.Mol, b: Chem.Mol, smarts: Optional[str] = None
) -> MCSResult:
    """if mcs_fxn(a, b, smarts) fails, try again with smarts from easier mcs_fxn(RemoveHs(a), RemoveHs(b))"""

    result = mcs_fxn(a, b, smarts)

    def unacceptable(result):
        return result.numBonds < 2

    if unacceptable(result) and smarts is None:
        # try again, but seed with MCS computed without explicit hydrogens
        a_without_hs = Chem.RemoveHs(deepcopy(a))
        b_without_hs = Chem.RemoveHs(deepcopy(b))

        heavy_atom_result = mcs_fxn(a_without_hs, b_without_hs, None)

        result = mcs_fxn(a, b, heavy_atom_result.smartsString)

    if unacceptable(result):
        message = f"""MCS result unacceptable!
            timed out: {result.canceled}
            # atoms in MCS: {result.numAtoms}
            # bonds in MCS: {result.numBonds}
        """
        raise AtomMappingError(message)

    return result


def _mcs_conformer_aware(a, b, threshold: float = 2.0, timeout: int = 5, smarts: Optional[str] = None):
    """Find maximum common substructure between mols a and b
    using reasonable settings for single topology:
    * only match atoms within distance threshold
        (assumes conformers are aligned)
    * disallow partial ring matches
    * disregard element identity and valence
    """
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


def mcs_conformer_aware(a, b, threshold: float = 2.0, timeout: int = 5, smarts: Optional[str] = None):
    """Compute maximum common substructure between mols a and b,
    possibly reseeding with result of easier MCS(RemoveHs(a), RemoveHs(b))
    """
    mcs_fxn = partial(_mcs_conformer_aware, threshold=threshold, timeout=timeout)
    return possibly_fallback_to_heavy_atom_mcs(mcs_fxn, a, b, smarts)


def _mcs_graph_only_complete_rings(a, b, timeout: int = 60, smarts: Optional[str] = None):
    """Find the MCS of A and B, disregarding conformer information. This also ensures
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
        seedSmarts=smarts if type(smarts) == str else "",
    )


def mcs_graph_only_complete_rings(a, b, timeout: int = 60, smarts: Optional[str] = None):
    """Find the MCS of A and B, disregarding conformer information. This also ensures
    that core-core bonds are not broken.
    Possibly reseeding with result of easier MCS(RemoveHs(a), RemoveHs(b))."""
    mcs_fxn = partial(_mcs_graph_only_complete_rings, timeout=timeout)
    return possibly_fallback_to_heavy_atom_mcs(mcs_fxn, a, b, smarts)


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
            cost[i, j] = np.sum(np.where(dij < threshold, dij, +np.inf))

    # find (i,j) = argmin cost
    min_i, min_j = np.unravel_index(np.argmin(cost, axis=None), cost.shape)

    # concatenate into (n_atoms, 2) array
    inds_a, inds_b = matches_a[min_i], matches_b[min_j]
    core = np.array([inds_a, inds_b]).T

    if np.isinf(cost[min_i, min_j]):
        raise AtomMappingError(f"not all mapped atoms are within {threshold:.3f}Å of each other")

    return core
