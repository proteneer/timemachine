import pytest
from rdkit import Chem

from timemachine.fe.atom_mapping import AtomMappingError, get_core_by_mcs, mcs
from timemachine.testsystems.relative import hif2a_ligand_pair


def test_mcs():
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
    assert that the MCS is not trivial, and that mcs(a, b) didn't modify a or b in-place"""

    smi_a = "C(C1CCCC1)C1=CC2=C(CCC2)C=C1"
    smi_b = "C(C1CCC1)C1=CC2=C(CCC2)C=C1"

    mol_a = Chem.AddHs(Chem.MolFromSmiles(smi_a))
    mol_b = Chem.AddHs(Chem.MolFromSmiles(smi_b))

    # to later check that mols weren't modified in place...
    hash_a = hash(Chem.MolToSmiles(mol_a))
    hash_b = hash(Chem.MolToSmiles(mol_b))

    # test default settings, which assume approx. aligned conformers
    mol_a.Compute2DCoords()
    mol_b.Compute2DCoords()

    mcs_result = mcs(mol_a, mol_b)
    assert mcs_result.numBonds > 1

    # expect failure if retry=False
    with pytest.raises(AtomMappingError) as e:
        _ = mcs(mol_a, mol_b, retry=False)
    assert "MCS" in str(e)

    # also test graph-only setting
    mcs_result = mcs(mol_a, mol_b, conformer_aware=False)
    assert mcs_result.numBonds > 1

    # again expect failure if retry=False
    with pytest.raises(AtomMappingError) as e:
        _ = mcs(mol_a, mol_b, conformer_aware=False, retry=False)
    assert "MCS" in str(e)

    # assert that the previous 4 calls to mcs didn't modify the molecules in-place
    assert hash(Chem.MolToSmiles(mol_a)) == hash_a
    assert hash(Chem.MolToSmiles(mol_b)) == hash_b
