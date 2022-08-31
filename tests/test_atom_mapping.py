from timemachine.fe.atom_mapping import get_core_by_mcs, mcs_map
from timemachine.testsystems.relative import hif2a_ligand_pair


def test_mcs_map():
    mcs_result = mcs_map(hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, threshold=0.5)
    assert mcs_result.queryMol is not None
    assert mcs_result.numAtoms > 1


def test_get_core_by_mcs():
    mol_a, mol_b = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b
    query = mcs_map(mol_a, mol_b).queryMol
    core = get_core_by_mcs(mol_a, mol_b, query)
    assert core.shape[1] == 2
