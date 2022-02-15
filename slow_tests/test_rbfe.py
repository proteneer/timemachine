import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from timemachine.fe.free_energy import construct_lambda_schedule
from timemachine.fe.model import RBFEModel
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.parallel.client import CUDAPoolClient
from timemachine.parallel.utils import get_gpu_count
from timemachine.testsystems.relative import hif2a_ligand_pair

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")
NUM_GPUS = get_gpu_count()


class TestRBFEModel(TestCase):
    def test_predict(self):
        """Just to verify that we can handle the most basic RBFE prediction"""
        # Use the Simple Charges to verify determinism of model. Needed as one endpoint uses the ff definition
        forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
            os.path.join(DATA_DIR, "hif2a_nowater_min.pdb")
        )

        # build the water system
        solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

        client = CUDAPoolClient(NUM_GPUS)

        model = RBFEModel(
            client=client,
            ff=forcefield,
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

        ordered_params = forcefield.get_ordered_params()
        mol_a = hif2a_ligand_pair.mol_a
        mol_b = hif2a_ligand_pair.mol_b
        core = hif2a_ligand_pair.core

        ddg, results = model.predict(ordered_params, mol_a, mol_b, core, seed=2022)
        self.assertEqual(len(results), 2)
        np.testing.assert_equal(ddg, 12.32189679129123)

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
