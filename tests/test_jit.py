import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from timemachine.fe.free_energy import HostConfig
from timemachine.ff import Forcefield
from timemachine.md.builders import build_water_system
from timemachine.md.minimizer import minimize_host_4d
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


class TestJIT(TestCase):
    def test_random_directory(self):
        with TemporaryDirectory(prefix="timemachine") as temp_dir:
            orig_dir = os.getcwd()
            os.chdir(temp_dir)
            try:
                # build a pair of alchemical ligands in a water box
                mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()
                ff = Forcefield.load_default()
                complex_system, complex_coords, complex_box, complex_top = build_water_system(2.6, ff.water_ff)

                # Creates a custom_ops.Context which triggers JIT
                host_config = HostConfig(complex_system, complex_coords, complex_box, complex_coords.shape[0])
                minimize_host_4d([mol_a, mol_b], host_config, ff)
            finally:
                os.chdir(orig_dir)
