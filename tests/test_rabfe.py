import os
from unittest import TestCase

import numpy as np

from md import builders, minimizer
from fe.model_rabfe import RelativeBindingModel, AbsoluteConversionModel
from fe.free_energy import construct_lambda_schedule
from fe.free_energy_rabfe import setup_relative_restraints_by_distance, get_romol_conf
from fe.frames import all_frames
from timemachine.potentials import rmsd
from testsystems.relative import hif2a_ligand_pair

from parallel.client import CUDAPoolClient
from parallel.utils import get_gpu_count

from common import temporary_working_dir

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NUM_GPUS = get_gpu_count()


class TestRABFEModels(TestCase):
    def test_predict_complex_decouple(self):
        """Just to verify that we can handle the most basic complex decoupling RABFE prediction"""
        complex_system, complex_coords, _, _, complex_box, complex_topology = builders.build_protein_system(
            os.path.join(DATA_DIR, "hif2a_nowater_min.pdb")
        )

        # build the water system
        solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

        temperature = 300.0
        pressure = 1.0
        dt = 2.5e-3

        client = CUDAPoolClient(NUM_GPUS)

        model = RelativeBindingModel(
            client,
            hif2a_ligand_pair.ff,
            complex_system,
            construct_lambda_schedule(2),
            complex_topology,
            temperature,
            pressure,
            dt,
            10,
            50,
            frame_filter=all_frames,
        )
        mol_a = hif2a_ligand_pair.mol_a
        mol_b = hif2a_ligand_pair.mol_b

        core_idxs = setup_relative_restraints_by_distance(mol_a, mol_b)

        ref_coords = get_romol_conf(mol_a)
        mol_coords = get_romol_conf(mol_b)  # original coords

        # Align using core_idxs
        R, t = rmsd.get_optimal_rotation_and_translation(
            x1=ref_coords[core_idxs[:, 1]],  # reference core atoms
            x2=mol_coords[core_idxs[:, 0]],  # mol core atoms
        )

        aligned_mol_coords = rmsd.apply_rotation_and_translation(mol_coords, R, t)
        complex_coords = minimizer.minimize_host_4d(
            [mol_a, mol_b],
            complex_system,
            complex_coords,
            hif2a_ligand_pair.ff,
            complex_box,
            [ref_coords, aligned_mol_coords],
        )
        complex_x0 = np.concatenate([complex_coords, ref_coords, aligned_mol_coords])

        ordered_params = hif2a_ligand_pair.ff.get_ordered_params()
        with temporary_working_dir() as temp_dir:
            dG, dG_err = model.predict(ordered_params, mol_a, mol_b, core_idxs, complex_x0, complex_box, "prefix")
            self.assertIsInstance(dG, float)
            self.assertIsInstance(dG_err, float)
            created_files = os.listdir(temp_dir)
            # 3 npz, 1 pdb and 1 npy per mol due to a->b and b->a
            self.assertEqual(len(created_files), 10)
            self.assertEqual(len([x for x in created_files if x.endswith(".pdb")]), 2)
            self.assertEqual(len([x for x in created_files if x.endswith(".npy")]), 2)
            self.assertEqual(len([x for x in created_files if x.endswith(".npz")]), 6)

    def test_predict_complex_conversion(self):
        """Just to verify that we can handle the most basic complex conversion RABFE prediction"""
        complex_system, complex_coords, _, _, complex_box, complex_topology = builders.build_protein_system(
            os.path.join(DATA_DIR, "hif2a_nowater_min.pdb")
        )

        # build the water system
        solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

        temperature = 300.0
        pressure = 1.0
        dt = 2.5e-3

        client = CUDAPoolClient(NUM_GPUS)

        model = AbsoluteConversionModel(
            client,
            hif2a_ligand_pair.ff,
            complex_system,
            construct_lambda_schedule(2),
            complex_topology,
            temperature,
            pressure,
            dt,
            10,
            50,
            frame_filter=all_frames,
        )
        mol_a = hif2a_ligand_pair.mol_a
        mol_b = hif2a_ligand_pair.mol_b

        core_idxs = setup_relative_restraints_by_distance(mol_a, mol_b)

        ref_coords = get_romol_conf(mol_a)
        mol_coords = get_romol_conf(mol_b)  # original coords

        # Use core_idxs to generate
        R, t = rmsd.get_optimal_rotation_and_translation(
            x1=ref_coords[core_idxs[:, 1]],  # reference core atoms
            x2=mol_coords[core_idxs[:, 0]],  # mol core atoms
        )

        aligned_mol_coords = rmsd.apply_rotation_and_translation(mol_coords, R, t)
        complex_coords = minimizer.minimize_host_4d(
            [mol_b], complex_system, complex_coords, hif2a_ligand_pair.ff, complex_box, [aligned_mol_coords]
        )
        complex_x0 = np.concatenate([complex_coords, aligned_mol_coords])

        ordered_params = hif2a_ligand_pair.ff.get_ordered_params()
        with temporary_working_dir() as temp_dir:
            dG, dG_err = model.predict(
                ordered_params, mol_b, complex_x0, complex_box, "prefix", core_idxs=core_idxs[:, 0]
            )
            self.assertIsInstance(dG, float)
            self.assertIsInstance(dG_err, float)
            created_files = os.listdir(temp_dir)
            # 2 npz, 1 pdb and 1 npy per mol due to a->b and b->a
            self.assertEqual(len(created_files), 4)
            self.assertEqual(len([x for x in created_files if x.endswith(".pdb")]), 1)
            self.assertEqual(len([x for x in created_files if x.endswith(".npy")]), 1)
            self.assertEqual(len([x for x in created_files if x.endswith(".npz")]), 2)
