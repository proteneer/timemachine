import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from md import builders
from fe.model import RBFEModel
from fe.free_energy import construct_lambda_schedule
from testsystems.relative import hif2a_ligand_pair
from parallel.client import CUDAPoolClient
from parallel.utils import get_gpu_count

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")
NUM_GPUS = get_gpu_count()


class TestRBFEModel(TestCase):
    def test_predict(self):
        """Just to verify that we can handle the most basic RBFE prediction"""
        complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
            os.path.join(DATA_DIR, "hif2a_nowater_min.pdb")
        )

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
            equil_steps=10,
            prod_steps=100,
        )

        ordered_params = hif2a_ligand_pair.ff.get_ordered_params()
        mol_a = hif2a_ligand_pair.mol_a
        mol_b = hif2a_ligand_pair.mol_b
        core = hif2a_ligand_pair.core

        ddg, results = model.predict(ordered_params, mol_a, mol_b, core)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(ddg, float)

    def test_pre_equilibration(self):
        """Verify that equilibration of edges up front functions as expected"""
        complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
            os.path.join(DATA_DIR, "hif2a_nowater_min.pdb")
        )

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
            equil_steps=10,
            prod_steps=100,
        )

        mol_a = hif2a_ligand_pair.mol_a
        mol_b = hif2a_ligand_pair.mol_b
        core = hif2a_ligand_pair.core
        assert len(model._equil_cache) == 0
        with TemporaryDirectory() as tempdir:
            cache_path = os.path.join(tempdir, "equil_cache.pkl")
            # If model.pre_equilibrate is false, its a noop
            model.equilibrate_edges([(mol_a, mol_b, core)], equilibration_steps=10, cache_path=cache_path)
            assert len(model._equil_cache) == 0

            # Enable pre-equilibration
            model.pre_equilibrate = True
            model.equilibrate_edges([(mol_a, mol_b, core)], equilibration_steps=10, cache_path=cache_path)
            # Cache should contain starting coords for both solvent and complex stages
            assert len(model._equil_cache) == 2
