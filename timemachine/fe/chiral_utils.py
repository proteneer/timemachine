import itertools
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from functools import partial
from typing import Callable

import numpy as np
from jax import jit
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from timemachine.constants import DEFAULT_CHIRAL_ATOM_RESTRAINT_K
from timemachine.fe.dummy import canonicalize_bond
from timemachine.fe.utils import get_romol_conf
from timemachine.graph_utils import convert_to_nx, enumerate_simple_paths
from timemachine.potentials.chiral_restraints import U_chiral_atom_batch, pyramidal_volume, torsion_volume

FourTuple = tuple[int, int, int, int]

ChiralConflict = tuple[FourTuple, FourTuple]


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
        "[#7X3:1](~F)(~F)~F",  # specific non-invertible nitrogen, for software testing
        # TODO: handle invertible pyramidal nitrogen
    ]

    chiral_atoms = set()
    for patt in chiral_patterns:
        query_mol = Chem.MolFromSmarts(patt)
        assert query_mol is not None
        for match in mol.GetSubstructMatches(query_mol):
            chiral_atoms.add(match[0])

    return chiral_atoms


def setup_all_chiral_atom_restr_idxs(mol, conf) -> list[FourTuple]:
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

    def __init__(self, restr_idxs: list[FourTuple] | NDArray):
        self.restr_idxs = restr_idxs
        self.allowed_set, self.disallowed_set = self.expand_symmetries()

    @classmethod
    def from_mol(cls, mol, conf):
        restr_idxs = setup_all_chiral_atom_restr_idxs(mol, conf)
        return ChiralRestrIdxSet(restr_idxs)

    def expand_symmetries(self) -> tuple[set[FourTuple], set[FourTuple]]:
        allowed_set = set()
        disallowed_set = set()

        for center, i, j, k in self.restr_idxs:
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
) -> set[ChiralConflict]:
    conflict_condition_fxn: Callable[[FourTuple], bool]
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


def has_chiral_atom_flips(
    core: Sequence[int],
    chiral_set_a: ChiralRestrIdxSet,
    chiral_set_b: ChiralRestrIdxSet,
) -> bool:
    # _find_atom_map_chiral_conflicts_one_direction, except (1) return bool not set, (2) hard-code mode = FLIP

    mapping_a_to_b = core

    # iterate over restraints defined in A, searching for possible conflicts
    for c_a, i_a, j_a, k_a in chiral_set_a.restr_idxs:
        mapped_tuple_b = mapping_a_to_b[c_a], mapping_a_to_b[i_a], mapping_a_to_b[j_a], mapping_a_to_b[k_a]
        if chiral_set_b.disallows(mapped_tuple_b):
            return True
    return False


def find_atom_map_chiral_conflicts(
    core: np.ndarray,
    chiral_set_a: ChiralRestrIdxSet,
    chiral_set_b: ChiralRestrIdxSet,
    mode: ChiralCheckMode = ChiralCheckMode.FLIP,
) -> set[ChiralConflict]:
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


def find_canonical_amide_bonds(mol):
    query = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]")
    amide_bonds = {canonicalize_bond((i, j)) for i, j, _, _ in mol.GetSubstructMatches(query)}
    return amide_bonds


def _find_flipped_torsions(
    torsions_a: Mapping[FourTuple, float], torsions_b: Mapping[FourTuple, float], core: Sequence[int]
) -> Iterator[ChiralConflict]:
    for (ia, ja, ka, la), sign_a in torsions_a.items():
        idxs_b = core[ia], core[ja], core[ka], core[la]
        try:
            sign_b = torsions_b[idxs_b]
        except KeyError:
            continue
        if sign_a != sign_b:
            yield ((ia, ja, ka, la), idxs_b)


def setup_find_flipped_planar_torsions(
    mol_a: Chem.rdchem.Mol, mol_b: Chem.rdchem.Mol
) -> Callable[[Sequence[int]], Iterator[tuple[FourTuple, FourTuple]]]:
    """Returns a function that enumerates core planar torsions that would be flipped by the given mapping.

    A planar torsion is defined here to be a torsion whose central bond is one of

    - a double or aromatic bond
    - an amide bond, as defined by :py:func:`find_canonical_amide_bonds`

    Parameters
    ----------
    mol_a, mol_b : rdkit.Chem.rdchem.Mol
        Input mols. Each mol must have a conformer defined.

    Returns
    -------
    Function with signature ((core: sequence of int) -> iterator over pairs of four-tuples)
        In the returned pairs, the first (second) tuple corresponds to the indices of the flipped torsion in mol_a (mol_b).
    """

    def enumerate_planar_torsions(mol):
        conf = get_romol_conf(mol)
        graph = convert_to_nx(mol)
        idxs = {canonicalize_bond(tuple(idxs)) for idxs in enumerate_simple_paths(graph, 4)}
        amide_bonds = find_canonical_amide_bonds(mol)

        planar_torsions = dict()
        for i, j, k, l in idxs:
            if canonicalize_bond((j, k)) not in amide_bonds:
                bond_type = mol.GetBondBetweenAtoms(j, k).GetBondType()
                if bond_type != BondType.DOUBLE and bond_type != BondType.AROMATIC:
                    continue

            # (j, k) is double, aromatic, or amide
            volume = torsion_volume(conf[i], conf[j], conf[k], conf[l])
            planar_torsions[(i, j, k, l)] = np.sign(volume)

        return planar_torsions

    planar_torsions_a = enumerate_planar_torsions(mol_a)
    planar_torsions_b = enumerate_planar_torsions(mol_b)

    # add reversed tuples to avoid needing to canonicalize
    planar_torsions_b.update({(l, k, j, i): sign for (i, j, k, l), sign in planar_torsions_b.items()})

    find_flipped_planar_torsions = partial(_find_flipped_torsions, planar_torsions_a, planar_torsions_b)

    return find_flipped_planar_torsions


def make_chiral_restr_fxns(mol_a, mol_b, chiral_k: float = DEFAULT_CHIRAL_ATOM_RESTRAINT_K):
    restr_idxs_a = np.array(setup_all_chiral_atom_restr_idxs(mol_a, get_romol_conf(mol_a)))
    restr_idxs_b = np.array(setup_all_chiral_atom_restr_idxs(mol_b, get_romol_conf(mol_b)))

    @jit
    def U_a(x_a):
        return 0.0 if restr_idxs_a.size == 0 else U_chiral_atom_batch(x_a, restr_idxs_a, chiral_k).sum()

    @jit
    def U_b(x_b):
        return 0.0 if restr_idxs_b.size == 0 else U_chiral_atom_batch(x_b, restr_idxs_b, chiral_k).sum()

    return U_a, U_b


def xs_ab_from_xs(xs: NDArray, atom_map):
    """map convert_single_topology_mols over xs

    Parameters
    ----------
    xs: An array of coordinates
        Coordinates containing the alchemical molecule constructed for RBFE
    atom_map: timemachine.fe.single_topology.AtomMapMixin
        Contains the atom map between the two end state molecules

    Returns
    -------
    2-tuple
        Returns a tuple of the mol_a and mol_b frames.

    """
    # Import here to avoid circular, TBD Deboggle
    from timemachine.fe.cif_writer import convert_single_topology_mols

    n_a = atom_map.num_atoms_a
    xs_a_, xs_b_ = [], []
    for x in xs:
        combined = convert_single_topology_mols(x, atom_map)
        xs_a_.append(combined[:n_a])
        xs_b_.append(combined[n_a:])
    xs_a = np.array(xs_a_)
    xs_b = np.array(xs_b_)
    return xs_a, xs_b


def make_chiral_flip_heatmaps(simulation_result, atom_map, mol_a, mol_b):
    """Evaluate mol_a and mol_b chiral restraint energy in each frame of a simulation

    Parameters
    ----------
    simulation_result: timemachine.fe.free_energy.SimulationResult
        Containing all of the frames that make up a simulation_result, may contain intermediate windows

    atom_map: timemachine.fe.single_topology.AtomMapMixin
        Contains the atom map between the two end state molecules

    Returns
    -------
    2-tuple
        Returns a tuple of the chiral energies in endstate A (lamb=0.0) and endstate B (lamb=1.0)
        each with shape (num_states, frames_per_state). Chiral energies are zero when there is no inversion
    """
    mol_a_chiral_conflicts = []
    mol_b_chiral_conflicts = []
    U_a, U_b = make_chiral_restr_fxns(mol_a, mol_b)

    for traj in simulation_result.frames:
        # Truncate off just the ligands, to handle all type of simulations
        # Use list comprehension to avoid loading the complete frames into memory traj is a StoredArrays object
        xs = np.array([frame[-atom_map.get_num_atoms() :] for frame in traj])
        xs_a, xs_b = xs_ab_from_xs(xs, atom_map)
        # TODO: probably vmap
        mol_a_chiral_conflicts.append(np.array([U_a(x) for x in xs_a]))
        mol_b_chiral_conflicts.append(np.array([U_b(x) for x in xs_b]))

    mol_a_chiral_conflicts = np.array(mol_a_chiral_conflicts)
    mol_b_chiral_conflicts = np.array(mol_b_chiral_conflicts)

    assert mol_a_chiral_conflicts.shape == (len(simulation_result.frames), len(simulation_result.frames[0]))

    return mol_a_chiral_conflicts, mol_b_chiral_conflicts
