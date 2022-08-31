import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.fe.restraints import setup_relative_restraints_by_distance, setup_relative_restraints_using_smarts
from timemachine.fe.utils import get_romol_conf


def test_setting_up_restraints_using_distance():
    seed = 814
    smi_a = "CCCONNN"
    smi_b = "CCCNNN"
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_a = Chem.AddHs(mol_a)

    mol_b = Chem.MolFromSmiles(smi_b)
    mol_b = Chem.AddHs(mol_b)
    for mol in [mol_a, mol_b]:
        AllChem.EmbedMolecule(mol, randomSeed=seed)

    mol_a_coords = get_romol_conf(mol_a)
    mol_b_coords = get_romol_conf(mol_b)

    core = setup_relative_restraints_by_distance(mol_a, mol_b)
    assert core.shape == (5, 2)

    # If we have a 0 cutoff, expect nothing to overlap
    core = setup_relative_restraints_by_distance(mol_a, mol_b, cutoff=0.0)
    assert core.shape == (0,)

    for cutoff in [0.08, 0.1, 0.2, 1.0]:
        core = setup_relative_restraints_by_distance(mol_a, mol_b, cutoff=cutoff)
        assert core.size > 0
        for a, b in core.tolist():
            assert np.linalg.norm(mol_a_coords[a] - mol_b_coords[b]) < cutoff

    # Adds seven hydrogen (terminal) atoms if allow terminal matches
    core = setup_relative_restraints_by_distance(mol_a, mol_b, terminal=True)
    assert core.shape == (12, 2)


def test_setting_up_restraints_using_smarts():

    smi_a = "CCCONNN"
    smi_b = "CCCNNN"
    mol_a = Chem.AddHs(Chem.MolFromSmiles(smi_a))
    mol_b = Chem.AddHs(Chem.MolFromSmiles(smi_b))

    smarts = "[#6]-[#6]-[#6]-[#7,#8]-[#7]-[#7]"

    core = setup_relative_restraints_using_smarts(mol_a, mol_b, smarts)

    expected_num_atoms = Chem.MolFromSmarts(smarts).GetNumAtoms()

    assert core.shape == (expected_num_atoms, 2)
