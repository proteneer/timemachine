# This file tests geometry classification for a wide variety of molecules,
# selected from common drugs and baran's heterocyclic chemistry

from rdkit import Chem

from timemachine.fe import geometry
from timemachine.fe.geometry import LocalGeometry as LG


def test_assign_aspirin():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))
    atom_types = geometry.classify_geometry(mol)

    expected_types = [
        LG.G4_TETRAHEDRAL,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G2_KINK,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_assign_nitrogens():
    """Test assignment with weird SP and SP2 nitrogens."""

    mol = Chem.AddHs(Chem.MolFromSmiles("CC1C(N)C2=C(C=NC(=C2)C#N)C1=O"))
    atom_types = geometry.classify_geometry(mol)

    expected_types = [
        LG.G4_TETRAHEDRAL,
        LG.G4_TETRAHEDRAL,
        LG.G4_TETRAHEDRAL,
        LG.G3_PYRAMIDAL,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G2_LINEAR,
        LG.G1_TERMINAL,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_assign_truncated_sildenafil():
    # useful for having a mixture of sulfur groups, and common heterocycles
    mol = Chem.AddHs(Chem.MolFromSmiles("CCOC1=C(C=C(C=C1)S(=O)(=O)N1CCN(C)CC1)C1=NC=CC(=O)N1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G4_TETRAHEDRAL,
        LG.G4_TETRAHEDRAL,
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G4_TETRAHEDRAL,  # S
        LG.G1_TERMINAL,  # O-S
        LG.G1_TERMINAL,  # O-S
        LG.G3_PYRAMIDAL,  # N
        LG.G4_TETRAHEDRAL,  # C
        LG.G4_TETRAHEDRAL,  # C
        LG.G3_PYRAMIDAL,  # N
        LG.G4_TETRAHEDRAL,  # C-N
        LG.G4_TETRAHEDRAL,  # C
        LG.G4_TETRAHEDRAL,  # C
        LG.G3_PLANAR,
        LG.G2_KINK,  # N
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,  # C=O
        LG.G3_PLANAR,  # N
        LG.G1_TERMINAL,  # Hydrogens below
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_hif2a_set():
    # test that we can successfully assign all of the hif2a set without assertions
    suppl = Chem.SDMolSupplier("timemachine/testsystems/data/ligands_40.sdf", removeHs=False)
    for mol in suppl:
        geometry.classify_geometry(mol)


def test_baran_pyrrole():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc[nH]1"))
    atom_types = geometry.classify_geometry(mol)

    expected_types = [
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_pyrrolidine():
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCN1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G4_TETRAHEDRAL,
        LG.G4_TETRAHEDRAL,
        LG.G4_TETRAHEDRAL,
        LG.G4_TETRAHEDRAL,
        LG.G3_PYRAMIDAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_isoindole():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1cccc2c1c[nH]c2"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_carbazole():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc2c(c1)c3ccccc3[nH]2"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_furan():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccco1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_thiophene():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1cccs1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_2h_pyran():
    mol = Chem.AddHs(Chem.MolFromSmiles("C1C=CC=CO1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G4_TETRAHEDRAL,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_pyridine():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccn1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_quinoline():
    mol = Chem.AddHs(Chem.MolFromSmiles("C1=CC=C2C(=C1)C=CC=N2"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_oxazole():
    mol = Chem.AddHs(Chem.MolFromSmiles("o1cncc1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_thiazole():
    mol = Chem.AddHs(Chem.MolFromSmiles("s1cncc1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_imidazole():
    mol = Chem.AddHs(Chem.MolFromSmiles("[nH]1cncc1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_triazole():
    mol = Chem.AddHs(Chem.MolFromSmiles("[nH]1nncc1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_morpholine():
    mol = Chem.AddHs(Chem.MolFromSmiles("O1CC[NH]CC1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G2_KINK,
        LG.G4_TETRAHEDRAL,
        LG.G4_TETRAHEDRAL,
        LG.G3_PYRAMIDAL,
        LG.G4_TETRAHEDRAL,
        LG.G4_TETRAHEDRAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_piperazine():
    mol = Chem.AddHs(Chem.MolFromSmiles("[nH]1CC[nH]CC1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PYRAMIDAL,
        LG.G4_TETRAHEDRAL,
        LG.G4_TETRAHEDRAL,
        LG.G3_PYRAMIDAL,
        LG.G4_TETRAHEDRAL,
        LG.G4_TETRAHEDRAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_pyrazine():
    mol = Chem.AddHs(Chem.MolFromSmiles("n1ccncc1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_baran_tetrazole():
    mol = Chem.AddHs(Chem.MolFromSmiles("[nH]1nnnc1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G2_KINK,
        LG.G2_KINK,
        LG.G2_KINK,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


# test baran examples with reasonable titratable sites
def test_protonated_pyridine():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1cccc[nH+]1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types


def test_protonated_imidazole():
    mol = Chem.AddHs(Chem.MolFromSmiles("[nH]1c[nH+]cc1"))
    atom_types = geometry.classify_geometry(mol)
    expected_types = [
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G3_PLANAR,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
        LG.G1_TERMINAL,
    ]

    assert atom_types == expected_types
