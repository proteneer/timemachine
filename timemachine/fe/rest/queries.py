from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

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
    """
    Identify rotatable bonds in a molecule.

    NOTE: This is an extremely crude and inaccurate method. This misses simple cases like benzoic acids, amides, etc.

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

    # sanity check
    assert len(matches) >= rdMolDescriptors.CalcNumRotatableBonds(mol)

    return {mkbond(i, j) for i, j in matches}
