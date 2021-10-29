import numpy as np

from rdkit import Chem

from md import minimizer, builders

from ff.handlers.deserialize import deserialize_handlers
from ff import Forcefield


def test_minimizer():

    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
        "tests/data/hif2a_nowater_min.pdb"
    )

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    ff = Forcefield(deserialize_handlers(open("ff/params/smirnoff_1_1_0_ccc.py").read()))

    # these methods will throw if the minimization failed
    minimized_coords = minimizer.minimize_host_4d([mol_a, mol_b], complex_system, complex_coords, ff, complex_box)
    minimized_coords = minimizer.minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
    minimized_coords = minimizer.minimize_host_4d([mol_b], complex_system, complex_coords, ff, complex_box)
