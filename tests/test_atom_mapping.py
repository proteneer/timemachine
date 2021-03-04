# TODO: test mcs atom mapping
# TODO: move run-time distance check here?

# TODO: test distance atom mapping

from fe.atom_mapping import mcs_map, transformation_size, compute_all_pairs_mcs
from fe.atom_mapping import compute_transformation_size_matrix
from fe.atom_mapping import get_core_by_mcs, get_core_by_smarts, get_core_by_geometry
from testsystems.relative import hif2a_ligand_pair


def test_mcs_map():
    mcs_result = mcs_map(hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, threshold=0.5)
    assert mcs_result.queryMol is not None
    assert mcs_result.numAtoms > 1


def test_transformation_size():
    assert transformation_size(n_A=5, n_B=5, n_MCS=5) == 0
    assert transformation_size(n_A=5, n_B=5, n_MCS=0) > 0


def test_compute_all_pairs_mcs():
    mols = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b
    mcs_s = compute_all_pairs_mcs(mols)
    assert (mcs_s.shape == (len(mols), len(mols)))


def test_compute_transformation_size_matrix():
    mols = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b
    mcs_s = compute_all_pairs_mcs(mols)

    transformation_sizes = compute_transformation_size_matrix(mols, mcs_s)

    assert (transformation_sizes >= 0).all()
    assert (transformation_sizes.shape == (len(mols), len(mols)))


def test_get_core_by_mcs():
    mol_a, mol_b = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b
    query = mcs_map(mol_a, mol_b).queryMol
    core = get_core_by_mcs(mol_a, mol_b, query)
    assert core.shape[1] == 2


def test_get_core_by_geometry():
    mol_a, mol_b = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b
    core = get_core_by_geometry(mol_a, mol_b)
    assert core.shape[1] == 2
