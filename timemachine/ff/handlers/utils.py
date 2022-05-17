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
    """
    WIP: extract from two definitions:
    1. https://github.com/proteneer/timemachine/blob/8d6bd25a143aa81e8b3b8c6a33e6e03afe272c56/timemachine/ff/handlers/bonded.py#L226-L229
    2. https://github.com/proteneer/timemachine/blob/8d6bd25a143aa81e8b3b8c6a33e6e03afe272c56/timemachine/ff/handlers/bonded.py#L206-L213
    """

    # approach 1: sort neighbors (used for FF parameter look up)
    container_type = type(torsion)
    assert len(set(torsion)) == len(torsion)
    _a, b, _c, _d = torsion
    a, c, d = sorted([_a, _c, _d])
    canonicalized_1 = container_type([a, b, c, d])

    # approach 2: take min((b, a, c, d), (d, c, a, b)) (used for applying trefoil convention)
    a, b, c, d = torsion
    canonicalized_2 = canonicalize_bonded_ixn([b, a, c, d])

    # TODO: which one?
    # return canonicalized
    raise NotImplementedError(f"can't decide between {canonicalized_1} and {canonicalized_2}!")


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
