from functools import wraps
from typing import Set, Tuple

from rdkit import Chem

from timemachine.graph_utils import convert_to_nx

Angle = Tuple[int, int, int]


def canonicalize_bonded_ixn(arr):
    """
    Canonicalize a bonded interaction. If arr[0] < arr[-1] then arr is
    returned, else if arr[0] > arr[-1], then arr[::-1] is returned. If
    arr[0] == arr[-1] then an exception is thrown.

    Parameters
    ----------
    arr: list of int
        Bond indices.

    Returns
    -------
    arr: list of int
        Canonicalized bond indices.

    """
    container_type = type(arr)

    if len(arr) == 0:
        raise ValueError("zero sized array")
    elif len(arr) == 1:
        return arr
    elif arr[0] > arr[-1]:
        return container_type(reversed(arr))
    elif arr[0] == arr[-1]:
        raise ValueError("Invalid bond with first and last indices equal")
    else:
        return arr


@wraps(canonicalize_bonded_ixn)
def canonicalize_bond(bond):
    assert len(bond) == 2
    return canonicalize_bonded_ixn(bond)


@wraps(canonicalize_bonded_ixn)
def canonicalize_angle(angle):
    assert len(angle) == 3
    return canonicalize_bonded_ixn(angle)


@wraps(canonicalize_bonded_ixn)
def canonicalize_proper_torsion(torsion):
    assert len(torsion) == 4
    return canonicalize_bonded_ixn(torsion)


@wraps(canonicalize_bonded_ixn)
def canonicalize_improper_torsion(torsion):
    """(a, b, c, d) -> (a', b, c', d')
    where
    (a', c', d') = sorted((a, c, d))
    """

    # sort neighbors (used for FF parameter look up)
    container_type = type(torsion)
    assert len(set(torsion)) == len(torsion)

    center = torsion[1]
    neighbors = [torsion[0], torsion[2], torsion[3]]

    a, c, d = sorted(neighbors)
    b = center

    return container_type([a, b, c, d])


def get_improper_torsion_permutations(torsion):
    """Get all trefoil permutations"""

    # see also implementations in
    # * TM ImproperTorsionHandler https://github.com/proteneer/timemachine/blob/451803e01afe6231147a0e6a3ca019d4aa5069d8/timemachine/ff/handlers/bonded.py#L225-L230
    # * OpenFF toolkit https://github.com/openforcefield/openff-toolkit/blob/fade767977cda3c2d70399ac38644aa5428414fe/openff/toolkit/typing/engines/smirnoff/parameters.py#L3437-L3453

    torsion = canonicalize_improper_torsion(torsion)

    center = torsion[1]
    neighbors = [torsion[0], torsion[2], torsion[3]]
    neighbor_permutations = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    canonical_permutations = []
    for (i, j, k) in neighbor_permutations:
        canonical_permutations.append(canonicalize_bonded_ixn([center, neighbors[i], neighbors[j], neighbors[k]]))

    return canonical_permutations


def get_all_angles(mol) -> Set[Angle]:
    """Get all canonical tuples (a, b, c) such that (a bonded to b), (b bonded to c) and (a != c)"""

    g = convert_to_nx(mol)
    n = mol.GetNumAtoms()

    angles = set()
    for a in range(n):
        for b in range(n):
            b_neighbors = g.neighbors(b)
            for c in range(n):
                ab = a in b_neighbors
                bc = c in b_neighbors
                a_neq_c = a != c

                if ab and bc and a_neq_c:
                    angles.add(canonicalize_angle((a, b, c)))
    return angles


def match_smirks(mol, smirks):
    """
    Notes
    -----
    * See also implementations of match_smirks in
        * bootstrap_am1.py, which is identical
        * bcc_aromaticity.py, which uses OpenEye instead of RDKit
    """

    # Make a copy of the molecule
    rdmol = Chem.Mol(mol)
    # Use designated aromaticity model
    Chem.SanitizeMol(rdmol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
    Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)

    # Set up query.
    qmol = Chem.MolFromSmarts(smirks)  # cannot catch the error
    if qmol is None:
        raise ValueError('RDKit could not parse the SMIRKS string "{}"'.format(smirks))

    # Create atom mapping for query molecule
    idx_map = dict()
    for atom in qmol.GetAtoms():
        smirks_index = atom.GetAtomMapNum()
        if smirks_index != 0:
            idx_map[smirks_index - 1] = atom.GetIdx()
    map_list = [idx_map[x] for x in sorted(idx_map)]

    # Perform matching
    matches = list()
    for match in rdmol.GetSubstructMatches(qmol, uniquify=False):
        mas = [match[x] for x in map_list]
        matches.append(tuple(mas))

    return matches
