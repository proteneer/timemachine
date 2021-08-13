import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from md.builders import build_water_system
from md.minimizer import minimize_host_4d

from fe.free_energy import RelativeFreeEnergy

from testsystems.relative import hif2a_ligand_pair


class TestJIT(TestCase):

    def test_random_directory(self):
        with TemporaryDirectory(prefix="timemachine") as temp_dir:
            orig_dir = os.getcwd()
            os.chdir(temp_dir)
            try:
                # build a pair of alchemical ligands in a water box
                mol_a, mol_b, core, ff = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core, hif2a_ligand_pair.ff
                complex_system, complex_coords, complex_box, complex_top = build_water_system(2.6)

                # Creates a custom_ops.Context which triggers JIT
                min_complex_coords = minimize_host_4d([mol_a, mol_b], complex_system, complex_coords, ff, complex_box)
            finally:
                os.chdir(orig_dir)
