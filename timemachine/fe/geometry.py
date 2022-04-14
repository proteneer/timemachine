# Utility functions to help assign and identify local geometry points

from enum import Enum
from typing import List

from rdkit import Chem
from rdkit.Chem import HybridizationType


class LocalGeometry(Enum):
    G1_TERMINAL = 0  # R-X
    G2_KINK = 1  # R-X-H
    G2_LINEAR = 2  # R-X#N
    G3_PLANAR = 3  # R-X(=O)O
    G3_PYRAMIDAL = 4  # R-X(-H)H
    G4_TETRAHEDRAL = 5  # R-X(-H)(-H)H


def assign_atom_geometry(atom):
    """
    Heuristic using hybridization information to assign local description
    of geometry.
    """
    nbrs = list(atom.GetNeighbors())
    hybridization = atom.GetHybridization()
    if len(nbrs) == 0:
        assert 0, "Ion not supported"
    elif len(nbrs) == 1:
        return LocalGeometry.G1_TERMINAL
    elif len(nbrs) == 2:
        if hybridization == HybridizationType.SP3:
            return LocalGeometry.G2_KINK
        elif hybridization == HybridizationType.SP2:
            return LocalGeometry.G2_KINK
        elif hybridization == HybridizationType.SP:
            return LocalGeometry.G2_LINEAR
        else:
            assert 0, "Unknown 2-nbr geometry!"
    elif len(nbrs) == 3:
        if hybridization == HybridizationType.SP3:
            return LocalGeometry.G3_PYRAMIDAL
        elif hybridization == HybridizationType.SP2:
            return LocalGeometry.G3_PLANAR
        else:
            assert 0, "Unknown 3-nbr geometry"
    elif len(nbrs) == 4:
        if hybridization == HybridizationType.SP3:
            return LocalGeometry.G4_TETRAHEDRAL
        else:
            assert 0, "Unknown 4-nbr geometry"
    else:
        assert 0, "Too many neighbors"


def classify_geometry(mol: Chem.Mol) -> List[LocalGeometry]:
    """
    Identify the local geometry of the molecule. This current uses a heuristic but we
    should really be generating this from gas-phase simulations of the real forcefield.

    Currently, 3D coordinates are not required, but this may change in the future.

    Parameters
    ----------
    mol: Chem.Mol
        Input molecule.

    Returns
    -------
    List[LocalGeometry]
        List of per atom geometries

    """

    geometry_types = []
    for a in mol.GetAtoms():
        geometry_types.append(assign_atom_geometry(a))

    return geometry_types
