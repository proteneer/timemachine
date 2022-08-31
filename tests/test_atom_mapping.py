from rdkit import Chem

from timemachine.fe.atom_mapping import get_core_by_mcs, mcs
from timemachine.testsystems.relative import hif2a_ligand_pair


def test_mcs_map():
    mcs_result = mcs(hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, threshold=0.5)
    assert mcs_result.queryMol is not None
    assert mcs_result.numAtoms > 1


def test_get_core_by_mcs():
    mol_a, mol_b = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b
    query = mcs(mol_a, mol_b).queryMol
    core = get_core_by_mcs(mol_a, mol_b, query)
    assert core.shape[1] == 2


def test_ring_size_change():
    """On a pair of molecules containing a 10-atom+ common substructure & a ring size change,
    assert that the MCS is not trivial"""

    smi_a = "C(C1CCCC1)C1=CC2=C(CCC2)C=C1"
    smi_b = "C(C1CCC1)C1=CC2=C(CCC2)C=C1"

    mol_a = Chem.AddHs(Chem.MolFromSmiles(smi_a))
    mol_b = Chem.AddHs(Chem.MolFromSmiles(smi_b))

    mol_a.Compute2DCoords()
    mol_b.Compute2DCoords()

    mcs_result = mcs(mol_a, mol_b)
    assert mcs_result.numBonds > 1
