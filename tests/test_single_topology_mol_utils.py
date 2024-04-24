from io import StringIO

import numpy as np
import pytest
from rdkit import Chem

from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield

pytestmark = [pytest.mark.nogpu]


def get_mol_a():
    # cyclobutane ring
    return Chem.MolFromMolBlock(
        """mol_a
                    3D
 Structure written by MMmdl.
 10 10  0  0  1  0            999 V2000
   61.0392  -33.5497  -35.2867 F   0  0  0  0  0  0
   60.3271  -33.5255  -36.4685 C   0  0  1  0  0  0
   61.0252  -33.1997  -37.8026 S   0  0  0  0  0  0
   60.0215  -34.1193  -38.5010 C   0  0  0  0  0  0
   59.8393  -34.8162  -37.1515 C   0  0  0  0  0  0
   59.4787  -32.8508  -36.3286 H   0  0  0  0  0  0
   60.4356  -34.7522  -39.2862 H   0  0  0  0  0  0
   59.1170  -33.6045  -38.8304 H   0  0  0  0  0  0
   58.8113  -35.1061  -36.9244 H   0  0  0  0  0  0
   60.5281  -35.6523  -37.0189 H   0  0  0  0  0  0
  1  2  1  0  0  0
  2  3  1  0  0  0
  2  5  1  0  0  0
  2  6  1  0  0  0
  3  4  1  0  0  0
  4  5  1  0  0  0
  4  7  1  0  0  0
  4  8  1  0  0  0
  5  9  1  0  0  0
  5 10  1  0  0  0
M  END

$$$$""",
        removeHs=False,
    )


def get_mol_b():
    # cyclohexane ring
    return Chem.MolFromMolBlock(
        """mol_b
                    3D
 Structure written by MMmdl.
 14 14  0  0  1  0            999 V2000
   60.9363  -33.6820  -35.2687 F   0  0  0  0  0  0
   60.3022  -33.9870  -36.4558 C   0  0  1  0  0  0
   60.2039  -32.8187  -37.4561 S   0  0  0  0  0  0
   59.6848  -33.3119  -38.8168 C   0  0  0  0  0  0
   60.5213  -34.3539  -39.3031 O   0  0  0  0  0  0
   60.4244  -35.5141  -38.4856 C   0  0  0  0  0  0
   61.0143  -35.1860  -37.1086 C   0  0  0  0  0  0
   59.2791  -34.2938  -36.2244 H   0  0  0  0  0  0
   59.7052  -32.4975  -39.5385 H   0  0  0  0  0  0
   58.6547  -33.6631  -38.7462 H   0  0  0  0  0  0
   60.9837  -36.3255  -38.9505 H   0  0  0  0  0  0
   59.3881  -35.8507  -38.4036 H   0  0  0  0  0  0
   60.9455  -36.0595  -36.4589 H   0  0  0  0  0  0
   62.0769  -34.9691  -37.2220 H   0  0  0  0  0  0
  1  2  1  0  0  0
  2  3  1  0  0  0
  2  7  1  0  0  0
  2  8  1  0  0  0
  3  4  1  0  0  0
  4  5  1  0  0  0
  4  9  1  0  0  0
  4 10  1  0  0  0
  5  6  1  0  0  0
  6  7  1  0  0  0
  6 11  1  0  0  0
  6 12  1  0  0  0
  7 13  1  0  0  0
  7 14  1  0  0  0
M  END

$$$$
""",
        removeHs=False,
    )


def test_st_mol():
    mol_a = get_mol_a()
    mol_b = get_mol_b()

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [6, 4], [8, 5], [4, 6], [5, 7], [7, 8], [9, 12]])

    # at lambda=0,lambda=0.01: bond (6,8) should be missing, bond (3,4) should be present
    # at lambda=99,lambda=1: bond (6,8) should be present, and bond (3,4) should be missing
    # near lambda=0.5, both bonds are present (this is implementation dependent, and may break later on)
    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)

    for lamb in [0.0, 0.01]:
        mol = st.mol(lamb)
        assert mol.GetBondBetweenAtoms(6, 8) is None
        assert mol.GetBondBetweenAtoms(3, 4) is not None

    for lamb in [0.99, 1.0]:
        mol = st.mol(lamb)
        assert mol.GetBondBetweenAtoms(6, 8) is not None
        assert mol.GetBondBetweenAtoms(3, 4) is None

    for lamb in [0.49, 0.5, 0.51]:
        mol = st.mol(lamb)
        assert mol.GetBondBetweenAtoms(6, 8) is not None
        assert mol.GetBondBetweenAtoms(3, 4) is not None

    x_a = get_romol_conf(mol_a)
    x_b = get_romol_conf(mol_b)
    # test that SDWriter is able to write the resulting mols even though the valence rules
    # may be inconsistent on the intermediate mols
    for lamb in np.linspace(0, 1, 5):
        fh = StringIO()
        writer = Chem.SDWriter(fh)
        mol_conf = Chem.Conformer(mol.GetNumAtoms())
        mol_copy = Chem.Mol(mol)
        coords = st.combine_confs(x_a, x_b, lamb)
        for a_idx, pos in enumerate(coords):
            mol_conf.SetAtomPosition(a_idx, (pos * 10).astype(np.float64))
        mol_copy.AddConformer(mol_conf)
        writer.write(mol_copy)
