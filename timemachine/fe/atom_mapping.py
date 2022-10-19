from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdmolops
from scipy.spatial.distance import cdist

from timemachine.constants import DEFAULT_FF
from timemachine.fe.topology import AtomMappingError
from timemachine.fe.utils import set_romol_conf
from timemachine.ff import Forcefield
from timemachine.md.align import align_mols_by_core


def mcs(
    a,
    b,
    threshold: float = 2.0,
    timeout: int = 5,
    smarts: Optional[str] = None,
    conformer_aware: bool = True,
    retry: bool = True,
    match_hydrogens: bool = True,
    ring_options: bool = True,
):
    """Find maximum common substructure between mols a and b
    using reasonable settings for single topology:
    * disallow partial ring matches
    * disregard element identity and valence

    if conformer_aware=True, only match atoms within distance threshold
        (assumes conformers are aligned)

    if retry=True, then reseed with result of easier MCS(RemoveHs(a), RemoveHs(b)) in case of failure

    if match_hydrogens=False, then do not match using hydrogens. Will not retry

    if ring_options = False, do not set complete rings only and ring matches ring only.
        Note: If True, this may expose RDKIT bugs which can cause this function to hang.
    """
    params = rdFMCS.MCSParameters()

    # bonds
    params.BondCompareParameters.CompleteRingsOnly = int(ring_options)
    params.BondCompareParameters.RingMatchesRingOnly = int(ring_options)
    params.BondTyper = rdFMCS.BondCompare.CompareAny

    # atoms
    params.AtomCompareParameters.CompleteRingsOnly = int(ring_options)
    params.AtomCompareParameters.RingMatchesRingOnly = int(ring_options)
    params.AtomCompareParameters.MatchValences = 0
    params.AtomCompareParameters.MatchChiralTag = 0
    if conformer_aware:
        params.AtomCompareParameters.MaxDistance = threshold
    params.AtomTyper = rdFMCS.AtomCompare.CompareAny
    # globals
    params.Timeout = timeout
    if smarts is not None:
        if match_hydrogens:
            params.InitialSeed = smarts
        else:
            # need to remove Hs from the input smarts
            params.InitialSeed = Chem.MolToSmarts(Chem.RemoveHs(Chem.MolFromSmarts(smarts)))

    def strip_hydrogens(mol):
        """Strip hydrogens with deepcopy to be extra safe"""
        return Chem.RemoveHs(deepcopy(mol))

    if not match_hydrogens:
        # Setting CompareAnyHeavyAtom doesn't handle this correctly, strip hydrogens explicitly
        a = strip_hydrogens(a)
        b = strip_hydrogens(b)
        # Disable retrying, as it will compare original a and b
        retry = False

    # try on given mols
    result = rdFMCS.FindMCS([a, b], params)

    # optional fallback
    def is_trivial(mcs_result) -> bool:
        return mcs_result.numBonds < 2

    if retry and is_trivial(result) and smarts is None:
        # try again, but seed with MCS computed without explicit hydrogens
        a_without_hs = strip_hydrogens(a)
        b_without_hs = strip_hydrogens(b)

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

        In some cases, this can fail to find a mapping that satisfies the distance
        threshold, raising an AtomMappingError.
    """

    # fetch conformer, assumed aligned
    conf_a = mol_a.GetConformer(0).GetPositions()
    conf_b = mol_b.GetConformer(0).GetPositions()

    # note that >1 match possible here -- must pick minimum-cost match
    # TODO: possibly break this into two stages
    #  following https://github.com/proteneer/timemachine/pull/819#discussion_r966130215
    max_matches = 10_000
    matches_a = mol_a.GetSubstructMatches(query, uniquify=False, maxMatches=max_matches)
    matches_b = mol_b.GetSubstructMatches(query, uniquify=False, maxMatches=max_matches)

    # warn if this search won't be exhaustive
    if len(matches_a) == max_matches or len(matches_b) == max_matches:
        print("Warning: max_matches exceeded -- cannot guarantee to find a feasible core")

    if len(matches_a) == 0 or len(matches_b) == 0:
        raise AtomMappingError(f"No matches {matches_a} {matches_b}")

    # once rather than in subsequent double for-loop
    all_distances = cdist(conf_a, conf_b)
    gt_threshold = all_distances > threshold

    matches_a = [np.array(a) for a in matches_a]
    matches_b = [np.array(b) for b in matches_b]

    cost = np.zeros((len(matches_a), len(matches_b)))

    for i, a in enumerate(matches_a):
        for j, b in enumerate(matches_b):
            if np.any(gt_threshold[a, b]):
                cost[i, j] = +np.inf
            else:
                dij = all_distances[a, b]
                cost[i, j] = np.sum(dij)

    # find (i,j) = argmin cost
    min_i, min_j = np.unravel_index(np.argmin(cost, axis=None), cost.shape)

    # concatenate into (n_atoms, 2) array
    inds_a, inds_b = matches_a[min_i], matches_b[min_j]
    core = np.array([inds_a, inds_b]).T

    if np.isinf(cost[min_i, min_j]):
        raise AtomMappingError(f"not all mapped atoms are within {threshold:.3f}Å of each other")

    return core


def get_core_with_alignment(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    threshold: float = 2.0,
    n_steps: int = 200,
    k: float = 10000,
    ff: Optional[Forcefield] = None,
    initial_smarts: Optional[str] = None,
) -> Tuple[NDArray, str]:
    """Selects a core between two molecules, by finding an initial core then aligning based on the core.

    Parameters
    ----------
    mol_a: RDKit Mol

    mol_b: RDKit Mol

    threshold: float
        Threshold between atoms in angstroms

    n_steps: float
        number of steps to run for alignment

    ff: Forcefield or None
        Forcefield to use for alignment, defaults to DEFAULT_FF forcefield if None

    initial_smarts: str or None
        If set uses smarts as the initial seed to MCS and as a fallback
        if mcs results in a trivial core.

    Returns
    -------
    core : np.ndarray of ints, shape (n_MCS, 2)
    smarts: SMARTS string used to find core

    Notes
    -----
    * Warning! The initial core can contain an incorrect mapping, in that case the
        core returned will be the same as running mcs followed by get_core_by_mcs.
    """
    # Copy mols so that when we change coordinates doesn't corrupt inputs
    a_copy = deepcopy(mol_a)
    b_copy = deepcopy(mol_b)

    if ff is None:
        ff = Forcefield.load_from_file(DEFAULT_FF)

    def setup_core(mol_a, mol_b, match_hydrogens, initial_smarts):
        result = mcs(
            mol_a,
            mol_b,
            threshold=threshold,
            match_hydrogens=match_hydrogens,
            smarts=initial_smarts,
            ring_options=False,
        )
        query_mol = Chem.MolFromSmarts(result.smartsString)
        print(result.smartsString)
        core = get_core_by_mcs(mol_a, mol_b, query_mol, threshold=threshold)
        return core, result.smartsString

    try:
        heavy_atom_core, _ = setup_core(a_copy, b_copy, False, initial_smarts)

        conf_a, conf_b = align_mols_by_core(mol_a, mol_b, heavy_atom_core, ff, n_steps=n_steps, k=k)
        set_romol_conf(a_copy, conf_a)
        set_romol_conf(b_copy, conf_b)

        core, smarts = setup_core(a_copy, b_copy, True, initial_smarts)
        core, smarts = filter_partial_rings(a_copy, b_copy, core, smarts)
        if len(core) <= 2:
            raise AtomMappingError("Result is trivial after filtering rings")
        return core, smarts
    except AtomMappingError as err:
        # Fall back to user provided smarts
        if initial_smarts is not None:
            print(f"WARNING: Could not get atom mapping: {err}, falling back to user defined smarts: {initial_smarts}")
            query_mol = Chem.MolFromSmarts(initial_smarts)
            core = get_core_by_mcs(mol_a, mol_b, query_mol, threshold=threshold)
            return core, initial_smarts
        raise err


def find_partial_matched_rings(mol_a: Chem.Mol, mol_b: Chem.Mol, core_a: NDArray, core_b: NDArray) -> List[int]:
    """
    Return a list of atom indicies that are partially mapped rings.
    Note that partially mapped fused rings are allowed, as long as
    the complete subring is mapped.

    Parameters
    ----------
    mol_a:
        Return a list of partially mapped ring atom idxs for this mol.
    mol_b:
        Pair molecule used to determine ring mapping.
    core_a:
        mapped atom indicies in mol_a
    core_b:
        mapped atom indicies in mol_b
    """
    partial_map_idxs_in_mol_a = set()
    ring_info_a = mol_a.GetRingInfo()
    ring_info_b = mol_b.GetRingInfo()
    for ring_atoms_a in ring_info_a.AtomRings():
        ring_atoms_a_set = set(ring_atoms_a)
        matches = {}
        keep_atoms_a = []
        for a_idx in ring_atoms_a_set:
            if a_idx in core_a:
                b_idx = core_b[list(core_a).index(a_idx)]
                ring_a_sizes = ring_info_a.AtomRingSizes(int(a_idx))
                ring_b_sizes = ring_info_b.AtomRingSizes(int(b_idx))
                if set(ring_a_sizes).intersection(set(ring_b_sizes)):
                    matches[a_idx] = b_idx
                    # if atom a belongs to a fused ring that is mapped,
                    # we want to keep these atoms, even if the rest of the
                    # fused ring is not mapped

                    #     ^  _        ^  _
                    #    | | _ >  -> | | _|
                    #     v           v

                    # Keep the shared atoms of the 6-membered ring
                    # even though the 5 -> 4 membered ring is not mapped
                    if len(ring_a_sizes) > 1 or len(ring_b_sizes) > 1:
                        keep_atoms_a.append(a_idx)
                else:
                    pass

        if len(matches) == len(ring_atoms_a):
            continue
        else:
            # complete rings only
            partial_map_idxs = [a_idx for a_idx in ring_atoms_a_set - set(keep_atoms_a) if a_idx in core_a]
            partial_map_idxs_in_mol_a.update(partial_map_idxs)
    return list(partial_map_idxs_in_mol_a)


def get_core_atom_idxs(mol_to_core: NDArray, mol_atom_idxs: List[int]) -> List[int]:
    """
    Given a mapping to the core and a list of mol atom idxs,
    return the corresponding core atom idxs.

    Parameters
    ----------
    mol_to_core: np.ndarray of ints, shape (n_MCS, 2)
        Maps from the molecule to the core indicies.
    mol_atom_idxs: np.ndarray of ints, shape (n_MCS,)
        List of atom idxs to convert.
    """
    core_atom_idxs = set()
    for a_idx in mol_atom_idxs:
        ordered = list(mol_to_core[:, 0])
        core_atom_idxs.add(mol_to_core[ordered.index(a_idx), 1])
    return list(core_atom_idxs)


def filter_partial_rings(mol_a: Chem.Mol, mol_b: Chem.Mol, core: NDArray, core_smarts: str):
    """
    Given a set of molecules and a MCS, remove partially mapped rings
    and incomplete rings from the core/smarts.
    """
    # Find a list of partially mapped rings atoms for the mapping in each direction
    a_partial_match = find_partial_matched_rings(mol_a, mol_b, core[:, 0], core[:, 1])
    b_partial_match = find_partial_matched_rings(mol_b, mol_a, core[:, 1], core[:, 0])

    # Get the corresponding core atoms for the partially mapped atoms
    mol_q_2d = Chem.MolFromSmarts(core_smarts)

    a_to_q = np.array([[int(x[1]), int(x[0])] for x in enumerate(core[:, 0])])
    b_to_q = np.array([[int(x[1]), int(x[0])] for x in enumerate(core[:, 1])])

    core_to_remove_idxs = []
    core_to_remove_idxs.extend(get_core_atom_idxs(a_to_q, a_partial_match))
    core_to_remove_idxs.extend(get_core_atom_idxs(b_to_q, b_partial_match))
    core_to_remove_idxs = list(set(core_to_remove_idxs))

    # remove atoms that are no long matched from the core
    cut_core = deepcopy(mol_q_2d)

    # store original idx
    for a in cut_core.GetAtoms():
        a.SetProp("orig_idx", str(a.GetIdx()))

    cut_core = Chem.EditableMol(cut_core)

    for a_idx in sorted(core_to_remove_idxs, reverse=True):
        cut_core.RemoveAtom(int(a_idx))

    cut_core = cut_core.GetMol()

    # Modified from RDKIT cookbook Index ID#: RDKitCB_31
    # Find the largest fragment and remove all other fragments
    # from the core. This takes care of non-ring atoms that are
    # no longer part of the MCS due to the removal of the
    # partially mapped ring atoms.
    mol_frags = sorted(rdmolops.GetMolFrags(cut_core, asMols=False))

    mol_frag_sizes = [len(frag) for frag in mol_frags]
    keep_idx = mol_frag_sizes.index(max(mol_frag_sizes))

    # keep only the largest fragment
    remove_core_atoms = []  # atom idxs in cut_core
    for i, mol_frag in enumerate(mol_frags):
        if i == keep_idx:
            continue
        remove_core_atoms.extend(mol_frag)

    # figure out which atoms have been removed in the original mol_q_2d core
    orig_removed_core_atoms = []  # atom idxs in the original core
    for a_idx in sorted(remove_core_atoms, reverse=True):
        a = cut_core.GetAtomWithIdx(int(a_idx))
        orig_removed_core_atoms.append(int(a.GetProp("orig_idx")))

    # actually remove the atoms from the core
    cut_core = Chem.EditableMol(cut_core)
    for a_idx in sorted(remove_core_atoms, reverse=True):
        cut_core.RemoveAtom(int(a_idx))

    # largest fragment is the new core
    largest_core_mol = cut_core.GetMol()
    largest_core_smarts = Chem.MolToSmarts(largest_core_mol)

    # remove any atoms in the original core that are no longer mapped
    largest_core = []
    for i_row, row in enumerate(core):
        if row[0] in a_partial_match or row[1] in b_partial_match:
            continue
        if i_row in orig_removed_core_atoms:
            continue
        largest_core.append(row)
    return np.array(largest_core), largest_core_smarts
