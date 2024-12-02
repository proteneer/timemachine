from typing import List, Optional, Tuple, TypeAlias

from rdkit import Chem

Mol: TypeAlias = Chem.rdchem.Mol

SMILES_BY_RES_NAME = {
    "ACE": "CC=O",
    "NME": "CN",
    # 'ARG': 'N[C@@H](CCCNC(N)=[NH2+])C(O)=O',
    # to preserve symmetry use symmetric resonance structure
    "ARG": "N[C@@H](CCC[NH+]=C(N)N)C(O)=O",
    "HID": "C1=C(NC=N1)C[C@@H](C(=O)O)N",
    "HIE": "N[C@@H](CC1=CNC=N1)C(O)=O",
    "HIP": "N[C@@H](CC1=CNC=[NH+]1)C(O)=O",
    "LYS": "N[C@@H](CCCC[NH3+])C(O)=O",
    "ASP": "N[C@@H](CC([O-])=O)C(O)=O",
    "ASH": "N[C@@H](CC(O)=O)C(O)=O",
    "GLU": "N[C@@H](CCC([O-])=O)C(O)=O",
    "GLH": "N[C@@H](CCC(O)=O)C(O)=O",
    "SER": "C([C@@H](C(=O)O)N)O",
    "THR": "C[C@H]([C@@H](C(=O)O)N)O",
    "ASN": "C([C@@H](C(=O)O)N)C(=O)N",
    "GLN": "C(CC(=O)N)[C@@H](C(=O)O)N",
    "CYS": "C([C@@H](C(=O)O)N)S",
    "CYM": "N[C@@H](C[S-])C(O)=O",
    "GLY": "C(C(=O)O)N",
    "PRO": "C1C[C@H](NC1)C(=O)O",
    "ALA": "O=C(O)C(N)C",
    "VAL": "CC(C)[C@@H](C(=O)O)N",
    "ILE": "CC[C@H](C)[C@@H](C(=O)O)N",
    "LEU": "CC(C)C[C@@H](C(=O)O)N",
    "MET": "CSCC[C@@H](C(=O)O)N",
    "PHE": "C1=CC=C(C=C1)C[C@@H](C(=O)O)N",
    "TYR": "C1=CC(=CC=C1C[C@@H](C(=O)O)N)O",
    "TRP": "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",
}


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


def get_query_mol(mol: Mol) -> Mol:
    """
    Return a query mol that has generic bonds.
    """
    query_params = Chem.AdjustQueryParameters.NoAdjustments()
    query_params.makeBondsGeneric = True
    query_generic_bonds = Chem.AdjustQueryProperties(mol, query_params)
    return query_generic_bonds


def make_residue_mol(name, atoms, bonds, name_list) -> Mol:
    # Generate an rdkit molecule given a list of atoms and a list of bonds
    # for a residue
    mw = Chem.RWMol()
    mw.BeginBatchEdit()
    for i, atom in enumerate(atoms):
        aa = Chem.Atom(atom)
        aa.SetProp("molAtomMapNumber", str(i))
        mw.AddAtom(aa)

    for src, dst in bonds:
        bond_tuple = name_list[src], name_list[dst]
        if bond_tuple in [("C", "O"), ("O", "C")]:
            mw.AddBond(src, dst, Chem.BondType.DOUBLE)
        else:
            mw.AddBond(src, dst, Chem.BondType.SINGLE)
    mw.CommitBatchEdit()
    mw = Chem.RemoveHs(mw, implicitOnly=True, sanitize=False)
    mw.SetProp("_Name", name)
    return mw


def make_residue_mol_from_template(template_name: str) -> Optional[Mol]:
    """
    Given a template residue name (e.g. HIP or NALA),
    return an rdkit molecule that has the proper
    bond types and charges set. Return None if the residue
    could not be generated from the template.
    """
    # Convert capped to regular residue name
    res_name, has_n_cap, _ = get_res_name(template_name)

    if res_name not in SMILES_BY_RES_NAME:
        return None

    # Create a residue mol with the proper bond types and charges
    proper_res_mol = Chem.AddHs(Chem.MolFromSmiles(SMILES_BY_RES_NAME[res_name]))
    proper_res_mol.SetProp("_Name", template_name + "_proper")

    # Set up the NH3+ cap, for NXYZ residues
    if has_n_cap:
        proper_res_mol = add_n_cap(proper_res_mol)

    return proper_res_mol


def update_mol_topology(omm_res_mol: Mol, proper_res_mol: Mol, name_list: List[str]):
    """
    Update omm_res_mol in place to copy the bond types and
    charges from the `proper_res_mol` generated using
    `make_residue_mol_from_template`.

    `name_list` is the list of atom names for the `omm_res_mol`.

    """
    match = proper_res_mol.GetSubstructMatch(get_query_mol(omm_res_mol))

    # Match maps the proper res mol (which has the proper bond types and charges)
    # back to the omm_res_mol (used for the smirks assignment)
    fwd_map = {i: v for i, v in enumerate(match)}  # from proper_res to omm
    proper_res_atoms = {i: atom for i, atom in enumerate(proper_res_mol.GetAtoms())}
    proper_res_mol.UpdatePropertyCache()

    for atom in omm_res_mol.GetAtoms():
        proper_res_idx = fwd_map[atom.GetIdx()]
        proper_res_atom = proper_res_atoms[proper_res_idx]

        new_charge = None
        if proper_res_atom.GetFormalCharge() != 0:
            new_charge = proper_res_atom.GetFormalCharge()

        # For capped residues, need to get the charge right for symmtery
        if name_list[atom.GetIdx()] == "OXT":
            new_charge = -1

        if new_charge is not None:
            atom.SetFormalCharge(new_charge)

    # Update the bonds/aromatic type based on the template
    proper_bonds = {}
    for bond in proper_res_mol.GetBonds():
        src_idx = bond.GetBeginAtomIdx()
        dst_idx = bond.GetEndAtomIdx()
        proper_bonds[(src_idx, dst_idx)] = bond
        proper_bonds[(dst_idx, src_idx)] = bond

    omm_res_mol.UpdatePropertyCache()
    for bond in omm_res_mol.GetBonds():
        src_idx = bond.GetBeginAtomIdx()
        dst_idx = bond.GetEndAtomIdx()

        bond_tuple = name_list[src_idx], name_list[dst_idx]
        # may not be matched (due to a partial molecule match)
        # and already handled in make_residue_mol
        if bond_tuple in [("C", "O"), ("O", "C"), ("C", "OXT"), ("OXT", "C")]:
            continue

        proper_src_idx = fwd_map[src_idx]
        proper_dst_idx = fwd_map[dst_idx]
        k = (proper_src_idx, proper_dst_idx)
        bond.SetBondType(proper_bonds[k].GetBondType())
        bond.SetIsAromatic(proper_bonds[k].GetIsAromatic())

    omm_res_mol.UpdatePropertyCache()


def get_res_name(res_name: str) -> Tuple[str, bool, bool]:
    """
    Return
    ------
    res_name: str
        3 letter residue name
    has_n_cap: bool
        True if this residue is capped at the N terminus.
    has_c_cap: bool
        True if this residue is capped at the C terminus.
    """
    has_c_cap = False
    has_n_cap = False
    if len(res_name) == 4 and res_name[0] == "C":
        res_name = res_name[1:]
        has_c_cap = True
    if len(res_name) == 4 and res_name[0] == "N":
        has_n_cap = True
        res_name = res_name[1:]

    return res_name, has_n_cap, has_c_cap


def add_n_cap(mol: Mol) -> Mol:
    """
    Add an N cap to the template residue mol.

    NOTE: This only works properly for residues constructed
    from the template, see `SMILES_BY_RES_NAME`.
    """
    mw = Chem.RWMol(mol)
    mw.BeginBatchEdit()
    # Need to adjust charge of the N to +1
    query_mol = get_query_mol(Chem.MolFromSmiles("NCC(O)=O"))
    matches = mol.GetSubstructMatches(query_mol)

    # find the N to add the cap
    for i, atom in enumerate(mw.GetAtoms()):
        if i not in matches[0]:
            continue
        if atom.GetSymbol() == "N":
            atom.SetFormalCharge(+1)
            atm_h = mw.AddAtom(Chem.Atom("H"))
            mw.AddBond(atm_h, atom.GetIdx())
            break
    mw.CommitBatchEdit()
    return mw
