from rdkit import Chem

from timemachine.ff import Forcefield
from timemachine.md import builders, minimizer


def test_minimizer():

    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
        "tests/data/hif2a_nowater_min.pdb"
    )

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    ff = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")

    # these methods will throw if the minimization failed
    minimizer.minimize_host_4d([mol_a, mol_b], complex_system, complex_coords, ff, complex_box)
    minimizer.minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
    minimizer.minimize_host_4d([mol_b], complex_system, complex_coords, ff, complex_box)


def test_equilibrate_host():
    host_system, host_coords, host_box, _ = builders.build_water_system(4.0)

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    mol = next(suppl)

    ff = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")

    coords, box = minimizer.equilibrate_host(mol, host_system, host_coords, 300, 1.0, ff, host_box, 25, seed=2022)
    assert coords.shape[0] == host_coords.shape[0] + mol.GetNumAtoms()
    assert coords.shape[1] == host_coords.shape[1]
    assert box.shape == host_box.shape
