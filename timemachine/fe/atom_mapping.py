from copy import deepcopy
from typing import Optional, Tuple

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
        Note that this options seem to be buggy.
    """
    params = rdFMCS.MCSParameters()

    # bonds
    if ring_options:
        params.BondCompareParameters.CompleteRingsOnly = 1
        params.BondCompareParameters.RingMatchesRingOnly = 1
    params.BondTyper = rdFMCS.BondCompare.CompareAny

    # atoms
    if ring_options:
        # TODO:  Always set rmr to true then filter partially mapped
        # ring atoms.
        params.AtomCompareParameters.CompleteRingsOnly = 1
        params.AtomCompareParameters.RingMatchesRingOnly = 1
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

    def setup_core(mol_a, mol_b, match_hydrogens):
        result = mcs(mol_a, mol_b, threshold=threshold, match_hydrogens=match_hydrogens, smarts=initial_smarts, ring_options=False)
        query_mol = Chem.MolFromSmarts(result.smartsString)
        core = get_core_by_mcs(mol_a, mol_b, query_mol, threshold=threshold)
        return core, result.smartsString

    try:
        heavy_atom_core, _ = setup_core(a_copy, b_copy, False, initial_smarts)

        conf_a, conf_b = align_mols_by_core(mol_a, mol_b, heavy_atom_core, ff, n_steps=n_steps, k=k)
        set_romol_conf(a_copy, conf_a)
        set_romol_conf(b_copy, conf_b)

        core, smarts = setup_core(a_copy, b_copy, True, initial_smarts)
        core, smarts = filter_partial_rings(a_copy, b_copy, core, smarts)
        return core, smarts
    except AtomMappingError as err:
        # Fall back to user provided smarts
        if initial_smarts is not None:
            print(f"WARNING: Could not get atom mapping: {err}, falling back to user defined smarts: {initial_smarts}")
            query_mol = Chem.MolFromSmarts(initial_smarts)
            core = get_core_by_mcs(mol_a, mol_b, query_mol, threshold=threshold)
            return core, initial_smarts
        raise err


# def find_partial_matched_rings(mol, core_a, core_b):
#     partial_map_idxs_in_mol = set()
#     for ring_atoms in mol.GetRingInfo().AtomRings():
#         ring_atoms_a_set = set(ring_atoms)
#         matches = {}
#         for a_idx in ring_atoms_set:
#             if a_idx in core_a:
#                 b_idx = core_b[list(core_a).index(a_idx)]
#                 matches[a_idx] = b_idx
#         if len(matches) == len(ring_atoms):
#             print('all matched', matches, ring_atoms)
#             continue
#         else:
#             # TODO: This does not exclude rings of different sizes
#             # if all atoms in the ring are mapped
#             partial_map_idxs_in_ring = set(matches.keys())
#             print('not matched', matches, ring_atoms, partial_map_idxs_in_ring)
#             partial_map_idxs_in_mol.update(partial_map_idxs_in_ring)
#     return partial_map_idxs_in_mol


def find_partial_matched_rings(mol_a, mol_b, core_a, core_b):
    partial_map_idxs_in_mol = set()
    ring_info_a = mol_a.GetRingInfo()
    ring_info_b = mol_b.GetRingInfo()
    for ring_atoms_a in ring_info_a.AtomRings():
        ring_atoms_a_set = set(ring_atoms_a)
        matches = {}
        keep_atoms_a = []
        for a_idx in ring_atoms_a_set:
            if a_idx in core_a:
                b_idx = core_b[list(core_a).index(a_idx)]
                # ring matches ring
                print("rmr", a_idx, ring_info_a.AtomRingSizes(int(a_idx)), b_idx, ring_info_b.AtomRingSizes(int(b_idx)))
                if set(ring_info_a.AtomRingSizes(int(a_idx))).intersection(set(ring_info_b.AtomRingSizes(int(b_idx)))):
                    matches[a_idx] = b_idx
                    # if atom a belongs to a fused ring that is mapped,
                    # we want to keep these atoms, even if the rest of the
                    # fused ring is not mapped

                    #     ^  _        ^  _
                    #    | | _ >  -> | | _|
                    #     v           v

                    # Keep the shared atoms of the 6-membered ring
                    # even though the 5 -> 4 membered ring is not mapped
                    # TODO: If there is say a fused 5-6 ring that goes to 6-5
                    # sharing the same two atoms does this cause problems?
                    keep_atoms_a.append(a_idx)
                else:
                    pass
                    # print('rmr failed', set(ring_info_a.AtomRingSizes(int(a_idx))).intersection(set(ring_info_b.AtomRingSizes(int(b_idx)))))
        if len(matches) == len(ring_atoms_a):
            print("all matched", matches, ring_atoms_a)
            continue
        else:
            # complete rings only
            partial_map_idxs = [a_idx for a_idx in ring_atoms_a_set - set(keep_atoms_a) if a_idx in core_a]
            print("not matched", matches, ring_atoms_a, partial_map_idxs)
            partial_map_idxs_in_mol.update(partial_map_idxs)
    return partial_map_idxs_in_mol


def get_core_unmatched(mol_to_core, unmatched_idxs):
    core_unmatched = set()
    for a_idx in unmatched_idxs:
        ordered = list(mol_to_core[:, 0])
        core_unmatched.add(mol_to_core[ordered.index(a_idx), 1])
    return list(core_unmatched)


def filter_partial_rings(mol_a, mol_b, core, smarts):
    a_unmatched = find_partial_matched_rings(mol_a, mol_b, core[:, 0], core[:, 1])
    b_unmatched = find_partial_matched_rings(mol_b, mol_a, core[:, 1], core[:, 0])
    print("a_unmatched", a_unmatched, "b_unmatched", b_unmatched)
    mol_q_2d = Chem.MolFromSmarts(smarts)
    q_to_a = np.array([[int(x[0]), int(x[1])] for x in enumerate(core[:, 0])])
    q_to_b = np.array([[int(x[0]), int(x[1])] for x in enumerate(core[:, 1])])

    a_to_q = np.array([[int(x[1]), int(x[0])] for x in enumerate(core[:, 0])])
    b_to_q = np.array([[int(x[1]), int(x[0])] for x in enumerate(core[:, 1])])

    core_unmatched_idxs = []
    core_unmatched_idxs.extend(get_core_unmatched(a_to_q, a_unmatched))
    core_unmatched_idxs.extend(get_core_unmatched(b_to_q, b_unmatched))
    core_unmatched_idxs = list(set(core_unmatched_idxs))
    print("core_unmatched_idxs", core_unmatched_idxs)

    cut_core = deepcopy(mol_q_2d)

    for a in cut_core.GetAtoms():
        a.SetProp("orig_idx", str(a.GetIdx()))

    cut_core = Chem.EditableMol(cut_core)

    for a_idx in sorted(core_unmatched_idxs, reverse=True):
        cut_core.RemoveAtom(int(a_idx))

    cut_core = cut_core.GetMol()

    remove_atoms = deepcopy(core_unmatched_idxs)

    # Modified from RDKIT cookbook Index ID#: RDKitCB_31
    mol_frags = sorted(rdmolops.GetMolFrags(cut_core, asMols=False))

    mol_frag_sizes = [len(frag) for frag in mol_frags]
    keep_idx = mol_frag_sizes.index(max(mol_frag_sizes))

    remove_core_atoms = []  # atom idxs in cut_core
    for i, mol_frag in enumerate(mol_frags):
        if i == keep_idx:
            continue
        remove_core_atoms.extend(mol_frag)
    print("remove_core_atoms", remove_core_atoms)

    orig_removed_core_atoms = []  # atom idxs in the original core
    for a_idx in sorted(remove_core_atoms, reverse=True):
        a = cut_core.GetAtomWithIdx(int(a_idx))
        orig_removed_core_atoms.append(int(a.GetProp("orig_idx")))

    cut_core = Chem.EditableMol(cut_core)
    for a_idx in sorted(remove_core_atoms, reverse=True):
        cut_core.RemoveAtom(int(a_idx))
    largest_core_mol = cut_core.GetMol()

    largest_core_smarts = Chem.MolToSmarts(largest_core_mol)
    print("largest_core_smarts", largest_core_smarts, largest_core_mol.GetNumAtoms())

    largest_core = []
    for i_row, row in enumerate(core):
        if row[0] in a_unmatched or row[1] in b_unmatched:
            continue
        if i_row in orig_removed_core_atoms:
            continue
        largest_core.append(row)
    largest_core = np.array(largest_core)
    return largest_core, largest_core_smarts
>>>>>>> b23b7a31 (filter out rings match ring to work around rdkit bug)
