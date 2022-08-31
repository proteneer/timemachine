import numpy as np
from rdkit.Chem import rdFMCS


class CompareDist(rdFMCS.MCSAtomCompare):
    """Custom atom comparison: use positions within generated conformer"""

    def __init__(self, threshold=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def compare(self, p, mol1, atom1, mol2, atom2):
        """Atoms match if within 0.5 Å

        Signature from super method:
        (MCSAtomCompareParameters)parameters, (Mol)mol1, (int)atom1, (Mol)mol2, (int)atom2) -> bool
        """
        x_i = mol1.GetConformer(0).GetPositions()[atom1]
        x_j = mol2.GetConformer(0).GetPositions()[atom2]
        return bool(np.linalg.norm(x_i - x_j) <= self.threshold)  # must convert from np.bool_ to Python bool!



class CompareDistNonterminal(rdFMCS.MCSAtomCompare):
    """
    Custom comparator used in the FMCS code.
    This allows two atoms to match if:
        1. Neither atom is a terminal atom (H, F, Cl, Halogens etc.)
        2. They are within 1 angstrom of each other.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compare(self, p, mol1, atom1, mol2, atom2):

        if mol1.GetAtomWithIdx(atom1).GetDegree() == 1:
            return False
        if mol2.GetAtomWithIdx(atom2).GetDegree() == 1:
            return False

        x_i = mol1.GetConformer(0).GetPositions()[atom1]
        x_j = mol2.GetConformer(0).GetPositions()[atom2]

        threshold = 1.0  # angstroms
        return bool(np.linalg.norm(x_i - x_j) <= threshold)