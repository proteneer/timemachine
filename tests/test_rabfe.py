import os
from unittest import TestCase

import numpy as np
from common import temporary_working_dir

from timemachine.fe.frames import all_frames
from timemachine.fe.free_energy import construct_lambda_schedule
from timemachine.fe.free_energy_rabfe import (
    AbsoluteFreeEnergy,
    RelativeFreeEnergy,
    get_romol_conf,
    setup_relative_restraints_by_distance,
)
from timemachine.fe.model_rabfe import (
    AbsoluteConversionModel,
    AbsoluteDecouplingModel,
    RelativeBindingModel,
    RelativeConversionModel,
)
from timemachine.ff import Forcefield
from timemachine.lib.potentials import Nonbonded, NonbondedInterpolated
from timemachine.md import builders, minimizer
from timemachine.parallel.client import CUDAPoolClient
from timemachine.parallel.utils import get_gpu_count
from timemachine.potentials import rmsd
from timemachine.testsystems.relative import hif2a_ligand_pair

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NUM_GPUS = get_gpu_count()


class TestRABFEModels(TestCase):
    def test_endpoint_parameters_match_decoupling_and_conversion_complex(self):
        r"""Verifies that the parameters are consistent for the following thermodynamic cycle:

        B                              A
        |                              |
        --\   X  ----\   Y   ----\  Z  --\
        A*|  --> B-A*|  -->  B*-A| --> B*|
        --/      ----/       ----/     --/

        X,Z are the decoupling stages
        Y is the conversion stages.

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

        mol_a = hif2a_ligand_pair.mol_a
        mol_b = hif2a_ligand_pair.mol_b

        # b is decoupled, a stays put
        decouple_ref_ab = RelativeFreeEnergy(decouple_model.setup_topology(mol_a, mol_b))
        decouple_unbound_potentials_ab, decouple_sys_params_ab, _ = decouple_ref_ab.prepare_host_edge(
            ff_params, decouple_model.host_system
        )

        # a is decoupled, b stays put
        decouple_ref_ba = RelativeFreeEnergy(decouple_model.setup_topology(mol_b, mol_a))
        decouple_unbound_potentials_ba, decouple_sys_params_ba, _ = decouple_ref_ba.prepare_host_edge(
            ff_params, decouple_model.host_system
        )

        # convert a charges into b charges
        conv_model_ab = RelativeConversionModel(
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

        conv_ref = RelativeFreeEnergy(conv_model_ab.setup_topology(mol_a, mol_b))
        conv_unbound_potentials, conv_sys_params, _ = conv_ref.prepare_host_edge(ff_params, conv_model_ab.host_system)

        assert len(conv_sys_params) == len(decouple_sys_params_ab)
        assert len(conv_sys_params) == len(decouple_sys_params_ba)

        seen_nonbonded = False
        for i, (decouple_pot_ab, decouple_pot_ba) in enumerate(
            zip(decouple_unbound_potentials_ab, decouple_unbound_potentials_ba)
        ):

            if not isinstance(decouple_pot_ab, Nonbonded):
                continue

            seen_nonbonded = True

            conv_pot = conv_unbound_potentials[i]
            assert isinstance(conv_pot, NonbondedInterpolated)

            conv_nonbonded_params = conv_sys_params[i]
            decouple_nonbonded_params_ab = decouple_sys_params_ab[i]
            decouple_nonbonded_params_ba = decouple_sys_params_ba[i]

            # decouple_ab params matches conv src
            # decouple_ba params matches conv dst
            conv_nonbonded_params_src = conv_nonbonded_params[: len(conv_nonbonded_params) // 2]
            conv_nonbonded_params_dst = conv_nonbonded_params[len(conv_nonbonded_params) // 2 :]

            # decouple_nonbonded_params_ba [host, mol_a, mol_b]
            np.testing.assert_array_equal(conv_nonbonded_params_src, decouple_nonbonded_params_ab)

            # re-order the parameters
            # conv_nonbonded_params_src [host, mol_a, mol_b]
            # decouple_nonbonded_params_ba [host, mol_b, mol_a]
            H = num_host_atoms
            B = mol_b.GetNumAtoms()

            nb_h = decouple_nonbonded_params_ba[:H]
            nb_b = decouple_nonbonded_params_ba[H : H + B]
            nb_a = decouple_nonbonded_params_ba[H + B :]

            expected_params = np.concatenate([nb_h, nb_a, nb_b])

            np.testing.assert_array_equal(conv_nonbonded_params_dst, expected_params)

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

        decouple_model = AbsoluteDecouplingModel(
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

    def test_predict_relative_binding_model(self):
        """Just to verify that we can handle the most basic decoupling RABFE prediction"""
        # Use the Simple Charges to verify determinism of model. Needed as one endpoint uses the ff definition
        forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
        # build the water system
        solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

        temperature = 300.0
        pressure = 1.0
        dt = 2.5e-3

        client = CUDAPoolClient(NUM_GPUS)

        model = RelativeBindingModel(
            client,
            forcefield,
            solvent_system,
            construct_lambda_schedule(2),
            solvent_topology,
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
        solvent_coords = minimizer.minimize_host_4d(
            [mol_a, mol_b],
            solvent_system,
            solvent_coords,
            forcefield,
            solvent_box,
            [ref_coords, aligned_mol_coords],
        )
        solvent_x0 = np.concatenate([solvent_coords, ref_coords, aligned_mol_coords])

        ordered_params = forcefield.get_ordered_params()
        with temporary_working_dir() as temp_dir:
            dG, dG_err = model.predict(
                ordered_params, mol_a, mol_b, core_idxs, solvent_x0, solvent_box, "prefix", seed=2022
            )
            # Since this is FF independent no issues around AM1BCC charge differences on OS/Conf
            np.testing.assert_almost_equal(dG, -0.447669, decimal=5)
            np.testing.assert_almost_equal(dG_err, 7.433752e-08, decimal=5)
            created_files = os.listdir(temp_dir)
            # 3 npz, 1 pdb and 1 npy per mol due to a->b and b->a
            self.assertEqual(len(created_files), 10)
            self.assertEqual(len([x for x in created_files if x.endswith(".pdb")]), 2)
            self.assertEqual(len([x for x in created_files if x.endswith(".npy")]), 2)
            self.assertEqual(len([x for x in created_files if x.endswith(".npz")]), 6)

    def test_predict_relative_conversion_model(self):
        """
        Verify that we can handle the most basic decoupling RABFE prediction.

        In particular, we check that RelativeConversionModel's predict function runs,
        is deterministic, and creates the expected number of pdb, npy, npz files
        """
        # Use the Simple Charges to verify determinism of model. Needed as one endpoint uses the ff definition
        forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
        # build the water system
        solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

        temperature = 300.0
        pressure = 1.0
        dt = 2.5e-3

        client = CUDAPoolClient(NUM_GPUS)

        model = RelativeConversionModel(
            client,
            forcefield,
            solvent_system,
            construct_lambda_schedule(2),
            solvent_topology,
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
        solvent_coords = minimizer.minimize_host_4d(
            [mol_a, mol_b],
            solvent_system,
            solvent_coords,
            forcefield,
            solvent_box,
            [ref_coords, aligned_mol_coords],
        )
        solvent_x0 = np.concatenate([solvent_coords, ref_coords, aligned_mol_coords])

        ordered_params = forcefield.get_ordered_params()
        with temporary_working_dir() as temp_dir:
            dG, dG_err = model.predict(
                ordered_params, mol_a, mol_b, core_idxs, solvent_x0, solvent_box, "prefix", seed=2022
            )
            # Since this is FF independent no issues around AM1BCC charge differences on OS/Conf
            np.testing.assert_almost_equal(dG, -24.215886, decimal=5)
            np.testing.assert_almost_equal(dG_err, 0.0)
            created_files = os.listdir(temp_dir)

            self.assertEqual(len(created_files), 4)
            self.assertEqual(len([x for x in created_files if x.endswith(".pdb")]), 1)
            self.assertEqual(len([x for x in created_files if x.endswith(".npy")]), 1)
            self.assertEqual(len([x for x in created_files if x.endswith(".npz")]), 2)

    def test_predict_absolute_conversion(self):
        """Verify that we can handle the most basic absolute conversion RABFE prediction.

        In particular, we check that AbsoluteConversionModel's predict function runs,
        is deterministic, and creates the expected number of pdb, npy, npz files
        """
        # Use the Simple Charges to verify determinism of model. Needed as one endpoint uses the ff definition
        forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        # build the water system
        solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

        temperature = 300.0
        pressure = 1.0
        dt = 2.5e-3

        client = CUDAPoolClient(NUM_GPUS)

        model = AbsoluteConversionModel(
            client,
            forcefield,
            solvent_system,
            construct_lambda_schedule(2),
            solvent_topology,
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
        solvent_coords = minimizer.minimize_host_4d(
            [mol_b], solvent_system, solvent_coords, forcefield, solvent_box, [aligned_mol_coords]
        )
        solvent_x0 = np.concatenate([solvent_coords, aligned_mol_coords])

        ordered_params = forcefield.get_ordered_params()
        with temporary_working_dir() as temp_dir:
            dG, dG_err = model.predict(
                ordered_params, mol_b, solvent_x0, solvent_box, "prefix", core_idxs=core_idxs[:, 0], seed=2022
            )
            np.testing.assert_almost_equal(dG, 49.177369, decimal=5)
            np.testing.assert_almost_equal(dG_err, 0.0)
            created_files = os.listdir(temp_dir)
            # 2 npz, 1 pdb and 1 npy per mol due to a->b and b->a
            self.assertEqual(len(created_files), 4)
            self.assertEqual(len([x for x in created_files if x.endswith(".pdb")]), 1)
            self.assertEqual(len([x for x in created_files if x.endswith(".npy")]), 1)
            self.assertEqual(len([x for x in created_files if x.endswith(".npz")]), 2)
