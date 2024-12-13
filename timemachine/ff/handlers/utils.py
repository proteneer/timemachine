from typing import List, Optional, Tuple, TypeAlias

from rdkit import Chem

Mol: TypeAlias = Chem.rdchem.Mol

AMIDE_SMILES = "NCC(O)=O"

SMILES_BY_RES_NAME = {
    "ACE": "CC=O",
    "NME": "CN",
    "ARG": "N[C@@H](CCC[NH+]=C(N)N)C(O)=O",  # to preserve symmetry use symmetric resonance structure
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
    Return a query mol that has generic bonds for substructure matching.
    """
    query_params = Chem.AdjustQueryParameters.NoAdjustments()
    query_params.makeBondsGeneric = True
    query_generic_bonds = Chem.AdjustQueryProperties(mol, query_params)
    return query_generic_bonds


def make_residue_mol(name: str, elements: List[str], bonds: List[Tuple[int, int]]) -> Mol:
    """
    Generate an rdkit molecule given a list of elements and a list of bonds
    for a residue.

    Parameters
    ----------
    name:
        Name of the residue.
    elements:
        List of atomic elements.
    bonds:
        List of atom idxs to connect.

    """
    mol = Chem.RWMol()
    mol.BeginBatchEdit()
    for i, element in enumerate(elements):
        atom = Chem.Atom(element)
        atom.SetProp("molAtomMapNumber", str(i))
        mol.AddAtom(atom)

    for src, dst in bonds:
        mol.AddBond(src, dst, Chem.BondType.SINGLE)

    mol.CommitBatchEdit()
    mol = Chem.RemoveHs(mol, implicitOnly=True, sanitize=False)
    mol.SetProp("_Name", name)
    return mol


def make_residue_mol_from_template(template_name: str) -> Optional[Mol]:
    """
    Given a template residue name (e.g. HIP or NALA),
    return an RDKit molecule that has the proper
    bond types and charges set. Return None if the residue
    could not be generated from the template.

    Parameters
    ----------
    template_name:
        Name of the residue. Supported residues are in `SMILES_BY_RES_NAME`
        and the equivalent N- or C-capped versions.

    Return
    ------
        Mol or None if the molecule could not be generated.
    """
    # Convert capped to regular residue name
    res_name, has_n_cap, has_c_cap = get_res_name(template_name)

    if res_name not in SMILES_BY_RES_NAME:
        if res_name != "HOH":
            print(f"WARNING: Skipping unknown residue {res_name}")
        return None

    # Create a residue mol with the proper bond types and charges
    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES_BY_RES_NAME[res_name]))
    mol.SetProp("_Name", template_name + "_template")

    # Set up the NH3+ cap, for NXYZ residues
    if has_n_cap:
        mol = add_n_cap(mol)

    # Set up the COO- cap, for CXYZ residues
    if has_c_cap:
        mol = add_c_cap(mol)

    return mol


def update_mol_topology(topology_res_mol: Mol, template_res_mol: Mol):
    """
    Update the topology_res_mol to copy the bond types and
    charges from the template_res_mol.

    Parameters
    ----------
    topology_res_mol:
        Generated using `make_residue_mol` to generate an RDKit mol
        from an OpenMM residue. This molecule is updated in place
        with the bond types and formal charges from `template_res_mol`.
    template_res_mol:
        Template molecule generated using `make_residue_mol_from_template`.
    """
    match = template_res_mol.GetSubstructMatch(get_query_mol(topology_res_mol))

    # Match maps the topology_res_mol to template_res_mol,
    # which has the proper bond types and charges
    fwd_map = {i: v for i, v in enumerate(match)}
    template_res_atoms = {i: atom for i, atom in enumerate(template_res_mol.GetAtoms())}
    template_res_mol.UpdatePropertyCache()

    for atom in topology_res_mol.GetAtoms():
        template_res_idx = fwd_map[atom.GetIdx()]
        template_res_atom = template_res_atoms[template_res_idx]

        if template_res_atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(template_res_atom.GetFormalCharge())

    # Update the bonds/aromatic type based on the template
    template_bonds = {}
    for bond in template_res_mol.GetBonds():
        src_idx = bond.GetBeginAtomIdx()
        dst_idx = bond.GetEndAtomIdx()
        template_bonds[(src_idx, dst_idx)] = bond
        template_bonds[(dst_idx, src_idx)] = bond

    topology_res_mol.UpdatePropertyCache()
    for bond in topology_res_mol.GetBonds():
        src_idx = bond.GetBeginAtomIdx()
        dst_idx = bond.GetEndAtomIdx()

        template_src_idx = fwd_map[src_idx]
        template_dst_idx = fwd_map[dst_idx]
        k = (template_src_idx, template_dst_idx)
        bond.SetBondType(template_bonds[k].GetBondType())
        bond.SetIsAromatic(template_bonds[k].GetIsAromatic())

    topology_res_mol.UpdatePropertyCache()


def get_res_name(template_name: str) -> Tuple[str, bool, bool]:
    """
    Parameters
    ----------
    template_name:
        Three code of the residue.
        Can have an 'N' or 'C' prefix to indicate a capped residue.

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
    res_name = template_name
    if len(res_name) == 4 and res_name[0] == "C":
        res_name = res_name[1:]
        has_c_cap = True
    if len(res_name) == 4 and res_name[0] == "N":
        has_n_cap = True
        res_name = res_name[1:]

    assert len(res_name) == 3, f"Invalid residue name: {template_name}"
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
    query_mol = get_query_mol(Chem.MolFromSmiles(AMIDE_SMILES))
    n_atom_idx = mol.GetSubstructMatches(query_mol)[0][0]
    n_atom = mw.GetAtomWithIdx(n_atom_idx)
    n_atom.SetFormalCharge(+1)
    h_atom = mw.AddAtom(Chem.Atom("H"))

    mw.AddBond(h_atom, n_atom_idx)
    mw.CommitBatchEdit()
    return mw


def add_c_cap(mol: Mol) -> Mol:
    """
    Update the C-cap (COOH -> COO-).

    NOTE: This only works properly for residues constructed
    from the template, see `SMILES_BY_RES_NAME`.
    """

    # ((0, 1, 9, 10, 11),)
    # Need to adjust charge of the O to -1 and remove the extra H
    query_mol = Chem.MolFromSmiles(AMIDE_SMILES)
    matches = mol.GetSubstructMatches(query_mol)

    mw = Chem.RWMol(mol)
    mw.BeginBatchEdit()

    oxygen_idx = matches[0][3]
    oxygen_atom = mw.GetAtomWithIdx(oxygen_idx)
    oxygen_atom.SetFormalCharge(-1)
    for bond in oxygen_atom.GetBonds():
        if bond.GetBeginAtom().GetSymbol() == "H":
            h_atom_idx = bond.GetBeginAtomIdx()
            break
        elif bond.GetEndAtom().GetSymbol() == "H":
            h_atom_idx = bond.GetEndAtomIdx()
            break

    mw.RemoveAtom(h_atom_idx)
    mw.CommitBatchEdit()
    return mw
