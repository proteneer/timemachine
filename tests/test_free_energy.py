import numpy as np
import unittest

from rdkit import Chem

from ff.handlers import openmm_deserializer
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from fe import free_energy

from md import builders, minimizer

from timemachine.lib import LangevinIntegrator, custom_ops


class TestFreeEnergy(unittest.TestCase):

    def test_absolute_free_energy(self):

        suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
        all_mols = [x for x in suppl]
        mol = all_mols[1]

        complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')
        complex_box += np.eye(3)*0.1 # BFGS this later

        # build the water system.
        solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
        solvent_box += np.eye(3)*0.1 # BFGS this later

        for host_system, host_coords, host_box in [
            (complex_system, complex_coords, complex_box),
            (solvent_system, solvent_coords, solvent_box)]:

            ff = Forcefield(deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read()))

            # minimize the host to avoid clashes
            host_coords = minimizer.minimize_host_4d(mol, host_system, host_coords, ff, host_box)

            afe = free_energy.AbsoluteFreeEnergy(mol, ff)

            final_potentials, final_vjp_and_handles, combined_masses, combined_coords = afe.prepare_host_edge(host_system, host_coords)

            seed = 2021

            intg = LangevinIntegrator(
                300.0,
                1.5e-3,
                1.0,
                combined_masses,
                seed
            ).impl()

            x0 = combined_coords
            v0 = np.zeros_like(x0)

            bound_potentials = [bp.bound_impl(np.float32) for bp in final_potentials]

            ctxt = custom_ops.Context(
                x0,
                v0,
                host_box,
                intg,
                bound_potentials
            )

            lamb = 0.5
            lambda_schedule = np.ones(10000)*lamb

            ctxt.multiple_steps(lambda_schedule)

            assert np.all(np.abs(ctxt.get_x_t() < 50))
            assert np.all(np.abs(ctxt._get_du_dx_t_minus_1() < 10000))

            # for fn, arg in zip(fns, args):
            bonded_us, nonbonded_us, grads_and_handles = afe.host_edge(lamb, host_system, host_coords, host_box, 5000, 5000)

            # check that means and standard deviations are well defined
            assert np.abs(bonded_us[0]) < 500.0
            assert np.abs(bonded_us[1]) < 500.0

            assert np.abs(nonbonded_us[0]) < 1000.0
            assert np.abs(nonbonded_us[1]) < 1000.0

            for g, h in grads_and_handles:
                assert np.all(np.abs(g) < 10000)


    def test_relative_free_energy(self):
        # test that we can properly build a single topology host guest system and
        # that we can run a few steps in a stable way. This tests runs both the complex
        # and the solvent stages.

        suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
        all_mols = [x for x in suppl]
        mol_a = all_mols[1]
        mol_b = all_mols[4]

        complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')
        complex_box += np.eye(3)*0.1 # BFGS this later

        # build the water system.
        solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
        solvent_box += np.eye(3)*0.1 # BFGS this later

        for host_system, host_coords, host_box in [
            (complex_system, complex_coords, complex_box),
            (solvent_system, solvent_coords, solvent_box)]:

            core = np.array([
                [ 0,  0],
                [ 2,  2],
                [ 1,  1],
                [ 6,  6],
                [ 5,  5],
                [ 4,  4],
                [ 3,  3],
                [15, 16],
                [16, 17],
                [17, 18],
                [18, 19],
                [19, 20],
                [20, 21],
                [32, 30],
                [26, 25],
                [27, 26],
                [ 7,  7],
                [ 8,  8],
                [ 9,  9],
                [10, 10],
                [29, 11],
                [11, 12],
                [12, 13],
                [14, 15],
                [31, 29],
                [13, 14],
                [23, 24],
                [30, 28],
                [28, 27],
                [21, 22]
            ])

            ff = Forcefield(deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read()))

            # minimize the host to avoid clashes
            host_coords = minimizer.minimize_host_4d(mol_a, host_system, host_coords, ff, host_box)

            rfe = free_energy.RelativeFreeEnergy(mol_a, mol_b, core, ff)

            fns  = [rfe.prepare_vacuum_edge, rfe.prepare_host_edge]
            args = [tuple(), (host_system, host_coords)]

            for fn, arg in zip(fns, args):

                final_potentials, final_vjp_and_handles, combined_masses, combined_coords = fn(*arg)

                seed = 2021

                intg = LangevinIntegrator(
                    300.0,
                    1.5e-3,
                    1.0,
                    combined_masses,
                    seed
                ).impl()

                x0 = combined_coords
                v0 = np.zeros_like(x0)

                bound_potentials = [bp.bound_impl(np.float32) for bp in final_potentials]

                ctxt = custom_ops.Context(
                    x0,
                    v0,
                    host_box,
                    intg,
                    bound_potentials
                )

                lamb = 0.5
                lambda_schedule = np.ones(10000)*lamb

                ctxt.multiple_steps(lambda_schedule)

                assert np.all(np.abs(ctxt.get_x_t() < 50))
                assert np.all(np.abs(ctxt._get_du_dx_t_minus_1() < 10000))


            # test the all-inclusive methods
            fns  = [rfe.vacuum_edge, rfe.host_edge]
            args = [(lamb, 5000, 5000), (lamb, host_system, host_coords, host_box, 5000, 5000)]

            for fn, arg in zip(fns, args):
                bonded_us, nonbonded_us, grads_and_handles = fn(*arg)

                # check that means and standard deviations are well defined
                assert np.abs(bonded_us[0]) < 500.0
                assert np.abs(bonded_us[1]) < 500.0

                assert np.abs(nonbonded_us[0]) < 1000.0
                assert np.abs(nonbonded_us[1]) < 1000.0

                for g, h in grads_and_handles:
                    assert np.all(np.abs(g) < 10000)
