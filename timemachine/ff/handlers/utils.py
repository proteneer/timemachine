from typing import Set, Tuple

from rdkit import Chem

from timemachine.graph_utils import convert_to_nx

Angle = Tuple[int, int, int]


def canonicalize_bond(arr):
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


def canonicalize_angle(angle) -> Tuple[int, int, int]:
    """Treat angle(a, b, c) as equivalent to angle(c, b, a) -- e.g. when assessing ff coverage"""
    a, b, c = angle
    return min((a, b, c), (c, b, a))


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
