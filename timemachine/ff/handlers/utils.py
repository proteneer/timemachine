import copy
from typing import List, Optional, Tuple, TypeAlias

from rdkit import Chem

Mol: TypeAlias = Chem.rdchem.Mol

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


def make_residue_mol(name: str, elements: List[str], bonds: List[int]) -> Mol:
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
    print(name, elements, bonds)
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


def update_carbonyl_bond_type(mol: Mol, atom_name_list: List[str]) -> Mol:
    """
    For the carbonyl linker, set the C=O double bond type.

    Parameters
    ----------
    mol:
        Generated with `make_residue_mol`.
    atom_name_list:
        List of atom names (using Amber types).

    Return
    ------
        Updated mol.
    """
    return mol
    mol = copy.deepcopy(mol)
    for bond in mol.GetBonds():
        atm_idx0, atm_idx1 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_names = sorted((atom_name_list[atm_idx0], atom_name_list[atm_idx1]))
        if bond_names == ["C", "O"]:
            bond.SetBondType(Chem.BondType.DOUBLE)
    mol = Chem.RemoveHs(mol, implicitOnly=True, sanitize=False)
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
        Name of the residue. Supported residues are in `SMILES_BY_RES_NAME`.

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
        print("has_c_cap", has_c_cap)
        mol = add_c_cap(mol)

    for atom in mol.GetAtoms():
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
    return mol


def update_mol_topology(topology_res_mol: Mol, template_res_mol: Mol, name_list: List[str]):
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
    name_list:
        List of Amber atom types for the `topology_res_mol` in order.
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

        new_charge = None
        if template_res_atom.GetFormalCharge() != 0:
            new_charge = template_res_atom.GetFormalCharge()

        # # For capped residues, need to get the charge right for symmtery
        # if name_list[atom.GetIdx()] == "OXT":
        #     new_charge = -1

        if new_charge is not None:
            atom.SetFormalCharge(new_charge)

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

        bond_tuple = name_list[src_idx], name_list[dst_idx]
        # # may not be matched (due to a partial molecule match)
        # # and already handled in make_residue_mol
        # if bond_tuple in [("C", "O"), ("O", "C"), ("C", "OXT"), ("OXT", "C")]:
        #     continue

        proper_src_idx = fwd_map[src_idx]
        proper_dst_idx = fwd_map[dst_idx]
        k = (proper_src_idx, proper_dst_idx)
        bond.SetBondType(template_bonds[k].GetBondType())
        bond.SetIsAromatic(template_bonds[k].GetIsAromatic())

    topology_res_mol.UpdatePropertyCache()


def get_res_name(res_name: str) -> Tuple[str, bool, bool]:
    """
    Parameters
    ----------
    res_name:
        Three letter of the residue. Can have an 'N' or 'C' prefix
        to indicate a capped residue.

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


def add_c_cap(mol: Mol) -> Mol:
    """
    Update the C-cap (COOH -> COO-).

    NOTE: This only works properly for residues constructed
    from the template, see `SMILES_BY_RES_NAME`.
    """
    mw = Chem.RWMol(mol)
    mw.BeginBatchEdit()
    # Need to adjust charge of the O to -1
    query_mol = get_query_mol(Chem.MolFromSmiles("NCC(O)=O"))
    matches = mol.GetSubstructMatches(query_mol)
    print("matches", matches)

    # find the O to adjust the charge
    h_atom_idx = None
    for i, atom in enumerate(mw.GetAtoms()):
        if i not in matches[0]:
            continue
        if atom.GetSymbol() == "O":
            if [bond.GetBondType() for bond in atom.GetBonds()] != [Chem.BondType.DOUBLE]:
                atom.SetFormalCharge(-1)
                for bond in atom.GetBonds():
                    if bond.GetBeginAtom().GetSymbol() == "H":
                        h_atom_idx = bond.GetBeginAtomIdx()
                    elif bond.GetEndAtom().GetSymbol() == "H":
                        h_atom_idx = bond.GetEndAtomIdx()
                break
    print("h_atom_idx", h_atom_idx)
    if h_atom_idx is not None:
        mw.RemoveAtom(h_atom_idx)
    mw.CommitBatchEdit()
    # mw.RemoveHs(mw, implicitOnly=False, sanitize=True)
    return mw
