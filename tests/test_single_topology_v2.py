import functools

import numpy as np
import pytest
from rdkit import Chem

from timemachine.fe import single_topology, utils
from timemachine.fe.single_topology import setup_dummy_interactions
from timemachine.ff import Forcefield


def test_find_chiral_atoms():

    # trivalent nitrogens always achiral
    # (probably wrong, but decent approximation for now)
    mol = Chem.AddHs(Chem.MolFromSmiles("FN(F)F"))
    chiral_atoms = single_topology.find_chiral_atoms(mol)
    assert chiral_atoms == set()

    # tetravalent nitrogens always chiral
    mol = Chem.AddHs(Chem.MolFromSmiles("F[N+](F)(F)F"))
    chiral_atoms = single_topology.find_chiral_atoms(mol)
    assert chiral_atoms == set([1])

    # trivalent phosphor/sulfur should throw an assertion
    # tetravalent phosphor/sulfur okay
    mol = Chem.AddHs(Chem.MolFromSmiles("CP(C)C"))
    with pytest.raises(AssertionError):
        single_topology.find_chiral_atoms(mol)

    mol = Chem.AddHs(Chem.MolFromSmiles("COP([O-])([O-])=O"))
    chiral_atoms = single_topology.find_chiral_atoms(mol)
    assert chiral_atoms == set([0, 2])

    mol = Chem.AddHs(Chem.MolFromSmiles("C[S+](C)C"))
    with pytest.raises(AssertionError):
        single_topology.find_chiral_atoms(mol)

    mol = Chem.AddHs(Chem.MolFromSmiles("CS(C)=O"))
    with pytest.raises(AssertionError):
        single_topology.find_chiral_atoms(mol)

    mol = Chem.AddHs(Chem.MolFromSmiles("CS(C)(=O)=O"))
    chiral_atoms = single_topology.find_chiral_atoms(mol)
    assert chiral_atoms == set([0, 1, 2])


def test_benzene_to_phenol():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))

    # do not allow H to map to O
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # add [O,H] as the dummy-group
    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [6, 12], anchor=5)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(5, 6), (6, 12)])
    assert set(angle_idxs) == set([(5, 6, 12)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0, 4), 5, 6)])

    # allow H to map to O
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [11, 6]])
    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [12], anchor=6)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs
    assert set(bond_idxs) == set([(6, 12)])
    assert set(angle_idxs) == set([(5, 6, 12)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set()


def test_benzene_to_benzoic_acid():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C(=O)O"))
    # map only the benzene
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # note that the C-C bond in benzoic acid is rotatable, despite being SP2 hybridized at both ends
    # this is because of steric hindrance effects, the only planar non-rotatable bond is O=C-C-H
    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [6, 7, 8, 14], anchor=5)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(5, 6), (6, 7), (6, 8), (8, 14)])
    assert set(angle_idxs) == set([(5, 6, 7), (5, 6, 8), (6, 8, 14), (7, 6, 8)])
    # assert set(proper_idxs) == set([(5, 6, 8, 14)]) # used to be a stereo-bond
    assert set(proper_idxs) == set([(5, 6, 8, 14), (7, 6, 8, 14)])
    assert set(improper_idxs) == set([(6, 5, 7, 8), (5, 8, 7, 6), (6, 8, 5, 7)])
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0, 4), 5, 6)])

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [11, 6]])
    # no longer a stereo bond
    with pytest.raises(AssertionError):
        setup_dummy_interactions(ff, mol_a, mol_b, core, [7, 8, 14], anchor=6)


def test_benzene_to_aniline():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1N"))
    # map only the benzene
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [6, 12, 13], anchor=5)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(5, 6), (6, 12), (6, 13)])
    assert set(angle_idxs) == set([(5, 6, 13), (5, 6, 12), (12, 6, 13)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set([(6, 13, 5, 12), (6, 5, 12, 13), (5, 13, 12, 6)])
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0, 4), 5, 6)])

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [11, 6]])
    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [12, 13], anchor=6)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs
    assert set(bond_idxs) == set([(6, 12), (6, 13)])
    assert set(angle_idxs) == set([(5, 6, 13), (5, 6, 12), (12, 6, 13)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()  # disable impropers
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set()


def test_benzene_to_benzonitrile():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C#N"))

    # map only the benzene
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [6, 7], anchor=5)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(5, 6), (6, 7)])
    assert set(angle_idxs) == set([(5, 6, 7)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0, 4), 5, 6)])

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [11, 6]])
    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [7], anchor=6)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs
    assert set(bond_idxs) == set([(6, 7)])
    assert set(angle_idxs) == set([(5, 6, 7)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()  # disable impropers
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set()


def test_benzene_to_toluene():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C"))

    # map hydrogen to carbon
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [11, 6]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [12, 13, 14], anchor=6)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(6, 12), (6, 13), (6, 14)])
    assert set(angle_idxs) == set([(5, 6, 12), (5, 6, 13), (5, 6, 14), (12, 6, 13), (12, 6, 14), (13, 6, 14)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()  # disable impropers
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set()


def test_ethanol_to_carboxylate():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("CO"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))

    core = np.array([[0, 0], [1, 1], [5, 3]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [2], anchor=1)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(1, 2)])
    assert set(angle_idxs) == set()
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()  # disable impropers
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0, 3), 1, 2)])


def test_ethanol_to_ammonium():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("CO"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("CN(F)Br"))

    core = np.array([[0, 0], [1, 1], [5, 3]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [2], anchor=1)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(1, 2)])
    assert set(angle_idxs) == set()
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()  # disable impropers
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0, 3), 1, 2)])


def test_ammonium_to_tetrahedral():
    mol_a = Chem.MolFromSmiles("N(F)(Cl)Br")
    mol_b = Chem.MolFromSmiles("C(F)(Cl)(Br)I")
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [4], anchor=0)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs
    assert set(bond_idxs) == set([(0, 4)])
    assert set(angle_idxs) == set()
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()
    assert set(x_angle_idxs) == set([((0, 1), (0, 2), (0, 4))])

    core = np.array([[0, 0], [1, 1], [2, 2]])
    all_idxs, _ = setup_dummy_interactions(ff, mol_a, mol_b, core, [3, 4], anchor=0)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs
    assert set(bond_idxs) == set([(0, 3), (0, 4)])
    assert set(angle_idxs) == set([(3, 0, 4)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()
    assert set(x_angle_idxs) == set([((0, 1), (0, 2), (0, 3)), ((0, 1), (0, 2), (0, 4))])


def test_enumerate_anchor_groups():

    #                                                          1  1 1
    #                          0 12  3  4    5   6  7   8  9   0  1 2
    config = Chem.SmilesParserParams()
    config.removeHs = False
    mol = Chem.MolFromSmiles(r"[H]OC1=C([H])\[N+](C([H])=C1[H])=C(\F)Cl", config)
    bond_idxs = utils.get_romol_bonds(mol)
    core_idxs = [1, 2, 3, 5, 6, 8]
    enumerate_fn = functools.partial(single_topology.enumerate_anchor_groups, bond_idxs=bond_idxs, core_idxs=core_idxs)

    nbs_1, nbs_2 = enumerate_fn(1)
    assert nbs_1 == set([2])
    assert nbs_2 == set([(2, 3), (2, 8)])

    nbs_1, nbs_2 = enumerate_fn(2)
    assert nbs_1 == set([1, 3, 8])
    assert nbs_2 == set([(3, 5), (8, 6)])

    nbs_1, nbs_2 = enumerate_fn(3)
    assert nbs_1 == set([2, 5])
    assert nbs_2 == set([(2, 8), (2, 1), (5, 6)])

    nbs_1, nbs_2 = enumerate_fn(5)
    assert nbs_1 == set([3, 6])
    assert nbs_2 == set([(3, 2), (6, 8)])

    nbs_1, nbs_2 = enumerate_fn(6)
    assert nbs_1 == set([5, 8])
    assert nbs_2 == set([(5, 3), (8, 2)])

    nbs_1, nbs_2 = enumerate_fn(8)
    assert nbs_1 == set([2, 6])
    assert nbs_2 == set([(2, 1), (2, 3), (6, 5)])


def test_check_stability():
    # check code detecting stability of bond and angle terms.

    # wrong idxs
    result = single_topology.check_bond_stability(0, 1, bond_idxs=[[1, 2]], bond_params=[[1000.0, 0.2]])
    assert result is False

    # weak force constant
    result = single_topology.check_bond_stability(0, 1, bond_idxs=[[1, 0]], bond_params=[[10.0, 0.0]])
    assert result is False

    # bond length too short
    result = single_topology.check_bond_stability(0, 1, bond_idxs=[[1, 0]], bond_params=[[1000.0, 0.04]])
    assert result is False

    # okay bond
    result = single_topology.check_bond_stability(0, 1, bond_idxs=[[1, 0]], bond_params=[[1000.0, 0.1]])
    assert result is True

    # wrong idxs
    result = single_topology.check_angle_stability(0, 1, 2, angle_idxs=[[1, 2, 0]], angle_params=[[100.0, 0.5]])
    assert result is False

    # weak force constant
    result = single_topology.check_angle_stability(0, 1, 2, angle_idxs=[[0, 1, 2]], angle_params=[[10.0, 1.2]])
    assert result is False

    # collapsed
    with pytest.raises(AssertionError):
        single_topology.check_angle_stability(0, 1, 2, angle_idxs=[[0, 1, 2]], angle_params=[[100.0, 0.0]])

    # linear
    result = single_topology.check_angle_stability(0, 1, 2, angle_idxs=[[0, 1, 2]], angle_params=[[100.0, 3.1415]])
    assert result is False

    # everything should be fine
    result = single_topology.check_angle_stability(0, 1, 2, angle_idxs=[[0, 1, 2]], angle_params=[[100.0, 1.2]])
    assert result is True
