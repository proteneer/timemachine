import os
from unittest import TestCase

from md import builders
from fe.model import RBFEModel
from fe.free_energy import construct_lambda_schedule
from testsystems.relative import hif2a_ligand_pair
from parallel.client import CUDAPoolClient
from parallel.utils import get_gpu_count

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NUM_GPUS = get_gpu_count()


class TestRBFEModel(TestCase):

    def test_predict(self):
        """Just to verify that we can handle the most basic RBFE prediction"""
        complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(os.path.join(DATA_DIR, "hif2a_nowater_min.pdb"))

        # build the water system
        solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

        client = CUDAPoolClient(NUM_GPUS)

        model = RBFEModel(
            client=client,
            ff=hif2a_ligand_pair.ff,
            complex_system=complex_system,
            complex_coords=complex_coords,
            complex_box=complex_box,
            complex_schedule=construct_lambda_schedule(2),
            solvent_system=solvent_system,
            solvent_coords=solvent_coords,
            solvent_box=solvent_box,
            solvent_schedule=construct_lambda_schedule(2),
            equil_steps=100,
            prod_steps=1000,
        )

        ordered_params = hif2a_ligand_pair.ff.get_ordered_params()
        mol_a = hif2a_ligand_pair.mol_a
        mol_b = hif2a_ligand_pair.mol_b
        core = hif2a_ligand_pair.core

        ddg, results = model.predict(ordered_params, mol_a, mol_b, core)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(ddg, float)