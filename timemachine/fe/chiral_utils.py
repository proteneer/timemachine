import itertools
from enum import Enum
from typing import List, Set, Tuple

import numpy as np
from rdkit import Chem

from timemachine.potentials.chiral_restraints import pyramidal_volume, torsion_volume

FourTuple = Tuple[int, int, int, int]

ChiralConflict = Tuple[FourTuple, FourTuple]


class ChiralCheckMode(Enum):
    FLIP = 1
    UNDEFINED = 2


def setup_chiral_atom_restraints(mol, conf, a_idx):
    """
    Setup chiral atom restraints for the molecule at a_idx by inspecting the
    given geometry.

    Parameters
    ----------
    mol: Chem.Mol
        input molecule

    conf: np.ndarray (N,3)
        conformation of the molecule

    a_idx: int
        Which atom to set up restraints for

    Returns
    -------
    list of 4-tuple
        Has length N choose 3, where N is the number of neighbors
        (c0, i0, j0, k0), (c0, i1, j1, k1), ...

    """

    nbs = mol.GetAtomWithIdx(a_idx).GetNeighbors()
    restr_idxs = []
    for a_i, a_j, a_k in itertools.combinations(nbs, 3):
        i, j, k = a_i.GetIdx(), a_j.GetIdx(), a_k.GetIdx()
        vol = pyramidal_volume(conf[a_idx], conf[i], conf[j], conf[k])
        # vol may be >0 or <0, our chiral restraint always enforces vol < 0.

        if vol < 0:
            restr_idxs.append((a_idx, i, j, k))
        else:
            restr_idxs.append((a_idx, j, i, k))

    return restr_idxs


def setup_chiral_bond_restraints(mol, conf, src_idx, dst_idx):
    """
    Setup chiral bond restraints for the molecule at a_idx by inspecting the
    given geometry.

    Parameters
    ----------
    mol: Chem.Mol
        input molecule

    conf: np.ndarray (N,3)
        conformation of the molecule

    src_idx: int
        Which starting atom of the bond to setup restraints for

    dst_idx: int
        Which ending atom of the bond to setup restraints for

    Returns
    -------
    List of 4-tuple
        Returns up to 4 chiral volumes based on the torsion of the form:
        (i_0, src_idx, dst_idx, l_0), (i_1, src_idx, dst_idx, l_1), ...

        Note that i_j may not be necessarily less than l_j
    """
    src_nbs = [a.GetIdx() for a in mol.GetAtomWithIdx(src_idx).GetNeighbors()]
    dst_nbs = [a.GetIdx() for a in mol.GetAtomWithIdx(dst_idx).GetNeighbors()]

    assert src_idx in dst_nbs
    assert dst_idx in src_nbs

    src_nbs.remove(dst_idx)
    dst_nbs.remove(src_idx)

    # build chiral restraints
    restr_idxs = []
    signs = []

    # set up torsions a,b,c,d
    b, c = src_idx, dst_idx
    for a in src_nbs:
        for d in dst_nbs:
            vol = torsion_volume(conf[a], conf[b], conf[c], conf[d])
            restr_idxs.append((a, b, c, d))
            if vol < 0:
                # (jkaus): the restraints are turned on when the volume is positive
                # so use the opposite sign here
                signs.append(1)
            else:
                signs.append(-1)

    return restr_idxs, signs


def find_chiral_atoms(mol):
    """
    Find chiral atoms in a molecule. Note that an atom is chiral if it has a non-invertible
    energy barrier. Even a center like methane is considered chiral.

    Parameters
    ----------
    mol: Chem.Mol
        input molecule

    Returns
    -------
    set of int
        Chiral atoms

    Notes
    -----
    May want to split this function into two definitions,
    one that says methane has a chiral center, and one that doesn't.
    """
    # these should be mutually exclusive, but if any pattern is hit then the results
    # are accumulated to a set
    chiral_patterns = [
        "[X4:1]",  # any tetrahedral atom
        "[#16X3,#15X3:1]",  # trivalent sulfur, phosphorous are assumed to be non-invertible
        # "[#7X3:1](~[R])(~[R])~[R]",  # nitrogen directly bonded to three ring atoms  # TODO: handle pyramidal nitrogen
    ]

    chiral_atoms = set()
    for patt in chiral_patterns:
        query_mol = Chem.MolFromSmarts(patt)
        assert query_mol is not None
        for match in mol.GetSubstructMatches(query_mol):
            chiral_atoms.add(match[0])

    return chiral_atoms


def setup_all_chiral_atom_restr_idxs(mol, conf) -> List[FourTuple]:
    """Apply setup_chiral_atom_restraints to all atoms found by find_chiral_atoms"""
    chiral_atom_set = find_chiral_atoms(mol)
    chiral_atom_restr_idxs = []
    for a_idx in chiral_atom_set:
        idxs = setup_chiral_atom_restraints(mol, conf, a_idx)
        for ii in idxs:
            assert ii not in chiral_atom_restr_idxs
        chiral_atom_restr_idxs.extend(idxs)
    return chiral_atom_restr_idxs


class ChiralRestrIdxSet:
    """Support fast checks of whether a trial 4-tuple is consistent with a set of chiral atom idxs"""

    def __init__(self, restr_idxs: List[FourTuple]):
        self.restr_idxs = [(int(c), int(i), int(j), int(k)) for (c, i, j, k) in restr_idxs]
        self.allowed_set, self.disallowed_set = self.expand_symmetries()

    @classmethod
    def from_mol(cls, mol, conf):
        restr_idxs = setup_all_chiral_atom_restr_idxs(mol, conf)
        return ChiralRestrIdxSet(restr_idxs)

    def expand_symmetries(self) -> Tuple[Set[FourTuple], Set[FourTuple]]:
        allowed_set = set()
        disallowed_set = set()

        for (center, i, j, k) in self.restr_idxs:
            # rotations
            allowed_set.add((center, i, j, k))
            allowed_set.add((center, j, k, i))
            allowed_set.add((center, k, i, j))

            # swaps
            disallowed_set.add((center, i, k, j))
            disallowed_set.add((center, j, i, k))
            disallowed_set.add((center, k, j, i))

        assert allowed_set.isdisjoint(disallowed_set)

        return allowed_set, disallowed_set

    def defines(self, trial_tuple: FourTuple) -> bool:
        return (trial_tuple in self.allowed_set) or (trial_tuple in self.disallowed_set)

    def disallows(self, trial_tuple: FourTuple) -> bool:
        return trial_tuple in self.disallowed_set


def _find_atom_map_chiral_conflicts_one_direction(
    core: np.ndarray,
    chiral_set_a: ChiralRestrIdxSet,
    chiral_set_b: ChiralRestrIdxSet,
    mode: ChiralCheckMode = ChiralCheckMode.FLIP,
) -> Set[ChiralConflict]:
    if mode == ChiralCheckMode.FLIP:
        conflict_condition_fxn = chiral_set_b.disallows
    elif mode == ChiralCheckMode.UNDEFINED:
        conflict_condition_fxn = lambda mapped_tuple_b: not chiral_set_b.defines(mapped_tuple_b)
    else:
        raise ValueError("invalid chiral check mode")

    # initialize convenient representations
    mapped_set_a = set(core[:, 0])
    conflicts = set()
    mapping_a_to_b = {int(a_i): int(b_i) for (a_i, b_i) in core}

    def apply_mapping(c, i, j, k):
        return mapping_a_to_b[c], mapping_a_to_b[i], mapping_a_to_b[j], mapping_a_to_b[k]

    # iterate over restraints defined in A, searching for possible conflicts
    for restr_tuple_a in chiral_set_a.restr_idxs:
        if set(restr_tuple_a).issubset(mapped_set_a):
            mapped_tuple_b = apply_mapping(*restr_tuple_a)

            if conflict_condition_fxn(mapped_tuple_b):
                conflicts.add((restr_tuple_a, mapped_tuple_b))

    return conflicts


def _has_chiral_atom_map_flips_one_direction(
    core: np.ndarray,
    chiral_set_a: ChiralRestrIdxSet,
    chiral_set_b: ChiralRestrIdxSet,
) -> bool:
    # _find_atom_map_chiral_conflicts_one_direction, except (1) return bool not set, (2) hard-code mode = FLIP

    conflict_condition_fxn = chiral_set_b.disallows

    # initialize convenient representations
    mapped_set_a = set(core[:, 0])
    mapping_a_to_b = {int(a_i): int(b_i) for (a_i, b_i) in core}

    def apply_mapping(c, i, j, k):
        return mapping_a_to_b[c], mapping_a_to_b[i], mapping_a_to_b[j], mapping_a_to_b[k]

    # iterate over restraints defined in A, searching for possible conflicts
    for restr_tuple_a in chiral_set_a.restr_idxs:
        if set(restr_tuple_a).issubset(mapped_set_a):
            mapped_tuple_b = apply_mapping(*restr_tuple_a)

            if conflict_condition_fxn(mapped_tuple_b):
                return True
    return False


def find_atom_map_chiral_conflicts(
    core: np.ndarray,
    chiral_set_a: ChiralRestrIdxSet,
    chiral_set_b: ChiralRestrIdxSet,
    mode: ChiralCheckMode = ChiralCheckMode.FLIP,
) -> Set[ChiralConflict]:
    """

    Parameters
    ----------
    core
        atom map, establishing correspondences
            mol_a[a_i] <-> mol_b[b_i]
        for (a_i, b_i) in core

    chiral_set_a, chiral_set_b
        chiral restraint sets for mols a and b

    mode : ChiralCheckMode
        FLIP : find cases where chiral atom restraints are defined
            for both mols a and b with opposite signs
        UNDEFINED: find cases where chiral atom restraints are defined
            for mol a (resp. b) but not mol b (resp. a)

    Returns
    -------
    conflicts
        set of conflicting pairs of 4-tuples
        ((a_c, a_i, a_j, a_k), (b_c, b_i, b_j, b_k))

    See Also
    --------
    * find_chiral_atoms -- definition of atom chirality used here -- notably: hydrogens are distinguishable
        (see additional motivation in https://github.com/proteneer/timemachine/pull/754 and related PR discussion)
    """
    conflicts_a2b = _find_atom_map_chiral_conflicts_one_direction(core, chiral_set_a, chiral_set_b, mode)
    conflicts_b2a = _find_atom_map_chiral_conflicts_one_direction(core[:, ::-1], chiral_set_b, chiral_set_a, mode)

    conflicts = conflicts_a2b.union(set((a, b) for (b, a) in conflicts_b2a))

    return conflicts


def has_chiral_atom_flips(core, chiral_set_a, chiral_set_b) -> bool:
    """find_atom_map_chiral_conflicts, except (1) return bool not set, (2) hard-code mode = FLIP"""
    # both directions
    if _has_chiral_atom_map_flips_one_direction(core, chiral_set_a, chiral_set_b):
        return True
    if _has_chiral_atom_map_flips_one_direction(core[:, ::-1], chiral_set_b, chiral_set_a):
        return True
    return False


def find_chiral_bonds(mol):
    """
    Find chiral bonds in a molecule. Current limited to double bonds and amides. Similarly,
    a bond is considered chiral if it has an extremely high rotational barrier that would
    be typically kinetically inaccessible.

    Parameters
    ----------
    mol: Chem.Mol
        input molecule

    Returns
    -------
    set of 2-tuple
        Chiral bonds

    """

    chiral_patterns = [
        "[X2,X3:1]=[X2,X3:2]",  # all double bonds with two or three neighbors,
        "[NX3,NX2:1][CX3:2](=[OX1])",  # amide bond
    ]

    chiral_bonds = set()
    for patt in chiral_patterns:
        query_mol = Chem.MolFromSmarts(patt)
        assert query_mol is not None
        for match in mol.GetSubstructMatches(query_mol):
            chiral_bonds.add(tuple(sorted([match[0], match[1]])))

    return chiral_bonds
