import re

from rdkit import Chem


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


def check_bond_smarts_symmetric(bond_smarts: str) -> bool:
    """Match [<atom1>:1]*[<atom2>:2],
    and return whether atom1 and atom2 are identical strings

    Notes
    -----
    * The AM1CCC model contains symmetric patterns that must be assigned 0 parameters
        (Otherwise, behavior when symmetric bond matches in an arbitrary direction)
    * Only checks string equivalence!
        for example
        check_bond_smarts_symmetric("[#6,#7:1]~[#7,#6:2]")
        will be a false negative
    * Does not match all possible bond smarts
        for example
        "[#6,#7:1]~[#7,#6:2]~[#1]"
        will throw an error
    """

    pattern = re.compile(r"\[(?P<atom1>.*)\:1\].\[(?P<atom2>.*)\:2\]")
    match = pattern.match(bond_smarts)
    assert type(match) is re.Match, "unrecognized bond smarts"
    return match.group("atom1") == match.group("atom2")
