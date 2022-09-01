from copy import deepcopy
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

from timemachine.fe.topology import AtomMappingError


def mcs(a, b, threshold: float = 2.0, timeout: int = 5, smarts: Optional[str] = None, conformer_aware=True, retry=True):
    """Find maximum common substructure between mols a and b
    using reasonable settings for single topology:
    * disallow partial ring matches
    * disregard element identity and valence

    if conformer_aware=True, only match atoms within distance threshold
        (assumes conformers are aligned)

    if retry=True, then reseed with result of easier MCS(RemoveHs(a), RemoveHs(b)) in case of failure
    """
    params = rdFMCS.MCSParameters()

    # bonds
    params.BondCompareParameters.CompleteRingsOnly = 1
    params.BondCompareParameters.RingMatchesRingOnly = 1
    params.BondTyper = rdFMCS.BondCompare.CompareAny

    # atoms
    params.AtomCompareParameters.CompleteRingsOnly = 1
    params.AtomCompareParameters.RingMatchesRingOnly = 1
    params.AtomCompareParameters.matchValences = 0
    params.AtomCompareParameters.MatchChiralTag = 0
    if conformer_aware:
        params.AtomCompareParameters.MaxDistance = threshold
    params.AtomTyper = rdFMCS.AtomCompare.CompareAny

    # globals
    params.Timeout = timeout
    if smarts is not None:
        params.InitialSeed = smarts

    # try on given mols
    result = rdFMCS.FindMCS([a, b], params)

    # optional fallback
    def is_trivial(mcs_result) -> bool:
        return mcs_result.numBonds < 2

    if retry and is_trivial(result) and smarts is None:
        # try again, but seed with MCS computed without explicit hydrogens
        a_without_hs = Chem.RemoveHs(deepcopy(a))
        b_without_hs = Chem.RemoveHs(deepcopy(b))

        heavy_atom_result = rdFMCS.FindMCS([a_without_hs, b_without_hs], params)
        params.InitialSeed = heavy_atom_result.smartsString

        result = rdFMCS.FindMCS([a, b], params)

    if is_trivial(result):
        message = f"""MCS result trivial!
            timed out: {result.canceled}
            # atoms in MCS: {result.numAtoms}
            # bonds in MCS: {result.numBonds}
        """
        raise AtomMappingError(message)

    return result


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
