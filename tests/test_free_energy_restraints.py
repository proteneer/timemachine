import pytest
from rdkit import Chem

from timemachine.fe.restraints import setup_relative_restraints_using_smarts

pytestmark = [pytest.mark.nocuda]


def test_setting_up_restraints_using_smarts():
    smi_a = "CCCONNN"
    smi_b = "CCCNNN"
    mol_a = Chem.AddHs(Chem.MolFromSmiles(smi_a))
    mol_b = Chem.AddHs(Chem.MolFromSmiles(smi_b))

    smarts = "[#6]-[#6]-[#6]-[#7,#8]-[#7]-[#7]"

    # setup_relative_restraints_using_smarts assumes conformers approximately aligned
    for mol in [mol_a, mol_b]:
        mol.Compute2DCoords()

    core = setup_relative_restraints_using_smarts(mol_a, mol_b, smarts)

    expected_num_atoms = Chem.MolFromSmarts(smarts).GetNumAtoms()

    assert core.shape == (expected_num_atoms, 2)
