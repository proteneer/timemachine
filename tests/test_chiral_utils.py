import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.fe import chiral_utils, utils
from timemachine.potentials.chiral_restraints import U_chiral_atom_batch, U_chiral_bond_batch


def test_setup_chiral_atom_restraints():
    """On a methane conformer, assert that permuting coordinates or permuting restr_idxs
    both independently toggle the chiral restraint"""
    mol = Chem.MolFromMolBlock(
        """
  Mrv2202 06072215563D

  5  4  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3633   -0.5138    0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0900    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3633    1.0277    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3633   -0.5138   -0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    # needs to be batched in order for jax to play nicely
    x0 = utils.get_romol_conf(mol)
    normal_restr_idxs = chiral_utils.setup_chiral_atom_restraints(mol, x0, 0)

    x0_inverted = x0[[0, 2, 1, 3, 4]]  # swap two atoms
    inverted_restr_idxs = chiral_utils.setup_chiral_atom_restraints(mol, x0_inverted, 0)

    # check the sign of the resulting idxs
    k = 1000.0
    assert np.all(np.asarray(U_chiral_atom_batch(x0, normal_restr_idxs, k)) == 0)
    assert np.all(np.asarray(U_chiral_atom_batch(x0, inverted_restr_idxs, k)) > 0)
    assert np.all(np.asarray(U_chiral_atom_batch(x0_inverted, normal_restr_idxs, k)) > 0)
    assert np.all(np.asarray(U_chiral_atom_batch(x0_inverted, inverted_restr_idxs, k)) == 0)


def test_setup_chiral_bond_restraints():
    """On a 'Cl/C(F)=N/F' conformer, assert that flipping a dihedral angle or permuting restr_idxs
    both independently toggle the chiral bond restraint"""

    mol_cis = Chem.MolFromSmiles(r"Cl\C(F)=N/F")
    mol_trans = Chem.MolFromSmiles(r"Cl\C(F)=N\F")

    AllChem.EmbedMolecule(mol_cis)
    AllChem.EmbedMolecule(mol_trans)

    # needs to be batched in order for jax to play nicely
    x0_cis = utils.get_romol_conf(mol_cis)
    x0_trans = utils.get_romol_conf(mol_trans)
    src_atom = 1
    dst_atom = 3
    normal_restr_idxs, signs = chiral_utils.setup_chiral_bond_restraints(mol_cis, x0_cis, src_atom, dst_atom)

    inverted_restr_idxs, inverted_signs = chiral_utils.setup_chiral_bond_restraints(
        mol_trans, x0_trans, src_atom, dst_atom
    )
    k = 1000.0

    assert np.all(np.asarray(U_chiral_bond_batch(x0_cis, normal_restr_idxs, k, signs)) == 0)
    assert np.all(np.asarray(U_chiral_bond_batch(x0_cis, inverted_restr_idxs, k, inverted_signs)) > 0)
    assert np.all(np.asarray(U_chiral_bond_batch(x0_trans, normal_restr_idxs, k, signs)) > 0)
    assert np.all(np.asarray(U_chiral_bond_batch(x0_trans, inverted_restr_idxs, k, inverted_signs)) == 0)


def test_find_chiral_atoms():
    # test that we can identify chiral atoms ub a couple of tetrahedral and pyramidal
    # molecules.

    mol = Chem.MolFromSmiles(r"FC(Cl)Br")
    res = chiral_utils.find_chiral_atoms(mol)
    assert res == set([1])

    mol = Chem.MolFromSmiles(r"FN(Cl)Br")
    res = chiral_utils.find_chiral_atoms(mol)
    assert res == set([])

    mol = Chem.MolFromSmiles(r"FS(Cl)Br")
    res = chiral_utils.find_chiral_atoms(mol)
    assert res == set([1])

    mol = Chem.MolFromSmiles(r"FP(Cl)Br")
    res = chiral_utils.find_chiral_atoms(mol)
    assert res == set([1])

    mol = Chem.AddHs(Chem.MolFromSmiles(r"C1CC2CCC3CCC1N23"))
    res = chiral_utils.find_chiral_atoms(mol)
    assert res == set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    mol = Chem.AddHs(Chem.MolFromSmiles(r"n1ccccc1"))
    res = chiral_utils.find_chiral_atoms(mol)
    assert res == set()


def test_find_chiral_bonds():
    # test that we can identify chiral bonds in a couple of simple systems
    # involving double bonds and amides

    mol = Chem.MolFromSmiles(r"FOOF")
    res = chiral_utils.find_chiral_bonds(mol)
    assert res == set([])

    mol = Chem.MolFromSmiles(r"FN=NF")
    res = chiral_utils.find_chiral_bonds(mol)
    assert res == set([(1, 2)])

    mol = Chem.AddHs(Chem.MolFromSmiles(r"c1cocc1"))
    res = chiral_utils.find_chiral_bonds(mol)
    assert res == set()

    mol = Chem.AddHs(Chem.MolFromSmiles(r"C1=CCC=CO1"))
    res = chiral_utils.find_chiral_bonds(mol)
    assert res == set([(0, 1), (3, 4)])

    mol = Chem.AddHs(Chem.MolFromSmiles(r"FNC=O"))
    res = chiral_utils.find_chiral_bonds(mol)
    assert res == set([(1, 2)])

    # tautomer of the above
    mol = Chem.AddHs(Chem.MolFromSmiles(r"O\C=N\F"))
    res = chiral_utils.find_chiral_bonds(mol)
    assert res == set([(1, 2)])

    mol = Chem.AddHs(Chem.MolFromSmiles(r"N(C)C(C)=O"))
    res = chiral_utils.find_chiral_bonds(mol)
    assert res == set([(0, 2)])
