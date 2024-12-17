from rdkit import Chem

from .bond import CanonicalBond, mkbond


def get_aliphatic_ring_bonds(mol: Chem.rdchem.Mol) -> set[CanonicalBond]:
    return {
        mkbond(
            mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
            mol.GetBondWithIdx(bond_idx).GetEndAtomIdx(),
        )
        for ring_bond_idxs in mol.GetRingInfo().BondRings()
        for is_aromatic in [all(mol.GetBondWithIdx(bond_idx).GetIsAromatic() for bond_idx in ring_bond_idxs)]
        if not is_aromatic
        for bond_idx in ring_bond_idxs
    }


def get_rotatable_bonds(mol: Chem.rdchem.Mol) -> set[CanonicalBond]:
    """Identify rotatable bonds in a molecule.

    NOTE: This uses the same (non-strict) pattern for a rotatable bond as RDKit:

        https://github.com/rdkit/rdkit/blob/e640915d4eb2140fbca76a820b69a8e15216a908/rdkit/Chem/Lipinski.py#L41

    Parameters
    ----------
    mol: ROMol
        Input molecule

    Returns
    -------
    set of CanonicalBond
        Set of bonds identified as rotatable
    """

    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(pattern, uniquify=1)
    return {mkbond(i, j) for i, j in matches}
