import os
from unittest import TestCase

import numpy as np

from timemachine.md import builders, minimizer
from timemachine.fe.model_rabfe import RelativeBindingModel, AbsoluteConversionModel, AbsoluteStandardHydrationModel
from timemachine.fe.free_energy import construct_lambda_schedule
from timemachine.fe.free_energy_rabfe import (
    setup_relative_restraints_by_distance,
    get_romol_conf,
    RelativeFreeEnergy,
    AbsoluteFreeEnergy,
)
from timemachine.fe.frames import all_frames
from timemachine.potentials import rmsd
from timemachine.lib.potentials import NonbondedInterpolated, Nonbonded
from testsystems.relative import hif2a_ligand_pair

from parallel.client import CUDAPoolClient
from parallel.utils import get_gpu_count

from common import temporary_working_dir

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NUM_GPUS = get_gpu_count()


class TestRABFEModels(TestCase):
    def test_endpoint_parameters_match_decoupling_and_conversion_complex(self):
        """Verifies that the parameters at the endpoint of conversion match with the starting parameters of
        the decoupling. Done on a complex model, as the hydration models differ

        Conv: P_start -> P_independent
        Decouple: P_independent -> P_arbitrary

        """
        host_system, host_coords, host_box, host_topology = builders.build_water_system(4.0)

        num_host_atoms = host_coords.shape[0]

        ff_params = hif2a_ligand_pair.ff.get_ordered_params()

        temperature = 300.0
        pressure = 1.0
        dt = 2.5e-3

        client = CUDAPoolClient(NUM_GPUS)

        decouple_model = RelativeBindingModel(
            client,
            hif2a_ligand_pair.ff,
            host_system,
            construct_lambda_schedule(2),
            host_topology,
            temperature,
            pressure,
            dt,
            10,
            50,
            frame_filter=all_frames,
        )

        blocker = hif2a_ligand_pair.mol_a
        ligand = hif2a_ligand_pair.mol_b

        decouple_topo = decouple_model.setup_topology(blocker, ligand)
        decouple_ref = RelativeFreeEnergy(decouple_topo)

        decouple_unbound_potentials, decouple_sys_params, _ = decouple_ref.prepare_host_edge(
            ff_params, decouple_model.host_system
        )

        conv_model = AbsoluteConversionModel(
            client,
            hif2a_ligand_pair.ff,
            host_system,
            construct_lambda_schedule(2),
            host_topology,
            temperature,
            pressure,
            dt,
            10,
            50,
            frame_filter=all_frames,
        )

        conv_topo = conv_model.setup_topology(ligand)
        conv_ref = AbsoluteFreeEnergy(ligand, conv_topo)

        conv_unbound_potentials, conv_sys_params, _ = conv_ref.prepare_host_edge(ff_params, conv_model.host_system)

        assert len(conv_sys_params) == len(decouple_sys_params)
        seen_nonbonded = False
        for i, decouple_pot in enumerate(decouple_unbound_potentials):
            if not isinstance(decouple_pot, NonbondedInterpolated):
                continue
            seen_nonbonded = True

            conv_pot = conv_unbound_potentials[i]
            assert isinstance(conv_pot, NonbondedInterpolated)

            conv_nonbonded_params = conv_sys_params[i]
            decouple_nonbonded_params = decouple_sys_params[i]

            # Shapes of parameters
            # Conversion Leg [src_ligand, dest_ligand]
            # Decouple Leg [dest_blocker, dest_ligand, blocker_halved, ligand_halved]

            # Should have the same number of parameters besides the blocker. Since params are interpolated, multiply by 2
            assert conv_nonbonded_params.shape[0] == decouple_nonbonded_params.shape[0] - blocker.GetNumAtoms() * 2
            # Should both share the same number of parameters types
            assert conv_nonbonded_params.shape[1] == decouple_nonbonded_params.shape[1]

            conv_params = conv_nonbonded_params[num_host_atoms * 2 :]
            decouple_params = decouple_nonbonded_params[num_host_atoms * 2 :]

            assert conv_params.shape[0] == decouple_params.shape[0] - blocker.GetNumAtoms() * 2

            # Verify the dest params of conv match the src params of decouple
            np.testing.assert_array_equal(
                conv_params[len(conv_params) // 2 :],
                decouple_params[len(decouple_params) // 2 + blocker.GetNumAtoms() :],
            )

        assert seen_nonbonded, "Found no NonbondedInterpolated potential"

    def test_endpoint_parameters_match_decoupling_and_conversion_solvent(self):
        """Verifies that the parameters at the endpoint of conversion match with the starting parameters of
        the decoupling. Done on a solvent model, as the complex models differ

        Conv: P_start -> P_independent
        Decouple: P_independent -> P_arbitrary

        """
        host_system, host_coords, host_box, host_topology = builders.build_water_system(4.0)

        num_host_atoms = host_coords.shape[0]

        ff_params = hif2a_ligand_pair.ff.get_ordered_params()

        temperature = 300.0
        pressure = 1.0
        dt = 2.5e-3

        client = CUDAPoolClient(NUM_GPUS)

        decouple_model = AbsoluteStandardHydrationModel(
            client,
            hif2a_ligand_pair.ff,
            host_system,
            construct_lambda_schedule(2),
            host_topology,
            temperature,
            pressure,
            dt,
            10,
            50,
            frame_filter=all_frames,
        )

        ligand = hif2a_ligand_pair.mol_b

        decouple_topo = decouple_model.setup_topology(ligand)
        decouple_ref = AbsoluteFreeEnergy(ligand, decouple_topo)

        decouple_unbound_potentials, decouple_sys_params, _ = decouple_ref.prepare_host_edge(
            ff_params, decouple_model.host_system
        )

        conv_model = AbsoluteConversionModel(
            client,
            hif2a_ligand_pair.ff,
            host_system,
            construct_lambda_schedule(2),
            host_topology,
            temperature,
            pressure,
            dt,
            10,
            50,
            frame_filter=all_frames,
        )

        conv_topo = conv_model.setup_topology(ligand)
        conv_ref = AbsoluteFreeEnergy(ligand, conv_topo)

        conv_unbound_potentials, conv_sys_params, _ = conv_ref.prepare_host_edge(ff_params, conv_model.host_system)

        assert len(conv_sys_params) == len(decouple_sys_params)
        seen_nonbonded = False
        for i, decouple_pot in enumerate(decouple_unbound_potentials):
            if not isinstance(decouple_pot, Nonbonded):
                continue
            seen_nonbonded = True

            conv_pot = conv_unbound_potentials[i]
            assert isinstance(conv_pot, NonbondedInterpolated)

            conv_nonbonded_params = conv_sys_params[i]
            decouple_nonbonded_params = decouple_sys_params[i]

            # Shapes of parameters
            # Conversion Leg [src, dest]
            # Decouple Leg [dest]

            # The conv has twice the parameters that the decouple does, as it is interpolated
            assert conv_nonbonded_params.shape[0] == decouple_nonbonded_params.shape[0] * 2
            # Should both share the same number of parameters types
            assert conv_nonbonded_params.shape[1] == decouple_nonbonded_params.shape[1]

            conv_params = conv_nonbonded_params[num_host_atoms * 2 :]
            decouple_params = decouple_nonbonded_params[num_host_atoms:]

            assert conv_params.shape[0] == decouple_params.shape[0] * 2

            # Verify the dest params of conv match the src params of decouple
            np.testing.assert_array_equal(
                conv_params[len(conv_params) // 2 :],
                decouple_params,
            )

        assert seen_nonbonded, "Found no NonbondedInterpolated potential"

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
