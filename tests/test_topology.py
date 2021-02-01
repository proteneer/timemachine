from jax.config import config; config.update("jax_enable_x64", True)

import unittest
import numpy as np

from fe import topology as topology

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from ff.handlers import openmm_deserializer
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from md import builders
import jax

from timemachine.lib import potentials, custom_ops

from timemachine.lib import LangevinIntegrator

import time

from md import minimizer


def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm


class BenzenePhenolSparseTest(unittest.TestCase):

    def setUp(self, *args, **kwargs):

        suppl = Chem.SDMolSupplier('tests/data/benzene_phenol_sparse.sdf', removeHs=False)
        all_mols = [x for x in suppl]

        self.mol_a = all_mols[0]
        self.mol_b = all_mols[1]

        # atom type free
        ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_recharge.py').read())

        self.ff = Forcefield(ff_handlers)

        super(BenzenePhenolSparseTest, self).__init__(*args, **kwargs)


    def test_bonded(self):
        # other bonded terms use an identical protocol, so we assume they're correct if the harmonic bond tests pass.
        # leaving benzene H unmapped, and phenol OH unmapped
        core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
        ], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)

        combined_params, vjp_fn, combined_potential = jax.vjp(st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True)

        # test that vjp_fn works
        vjp_fn(np.random.rand(*combined_params.shape))

        # we expect 15 bonds in total, of which 6 are duplicated.
        assert len(combined_potential.get_idxs() == 15)

        combined_idxs = combined_potential.get_idxs()
        src_idxs = set([tuple(x) for x in combined_potential.get_idxs()[:6]])
        dst_idxs = set([tuple(x) for x in combined_potential.get_idxs()[6:12]])

        np.testing.assert_equal(src_idxs, dst_idxs)

        cc = self.ff.hb_handle.lookup_smirks("[#6X3:1]:[#6X3:2]")
        cH = self.ff.hb_handle.lookup_smirks("[#6X3:1]-[#1:2]")
        cO = self.ff.hb_handle.lookup_smirks("[#6X3:1]-[#8X2H1:2]")
        OH = self.ff.hb_handle.lookup_smirks("[#8:1]-[#1:2]")

        params_src = combined_params[:6]
        params_dst = combined_params[6:12]
        params_uni = combined_params[12:]

        np.testing.assert_array_equal(params_src, [cc, cc, cc, cc, cc, cc])
        np.testing.assert_array_equal(params_dst, [cc, cc, cc, cc, cc, cc])
        np.testing.assert_array_equal(params_uni, [cH, cO, OH])

        # map H to O
        core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6]
        ], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)

        combined_params, vjp_fn, combined_potential = jax.vjp(st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True)

        assert len(combined_potential.get_idxs() == 15)

        combined_idxs = combined_potential.get_idxs()
        src_idxs = set([tuple(x) for x in combined_potential.get_idxs()[:7]])
        dst_idxs = set([tuple(x) for x in combined_potential.get_idxs()[7:14]])

        params_src = combined_params[:7]
        params_dst = combined_params[7:14]
        params_uni = combined_params[14:]

        np.testing.assert_array_equal(params_src, [cc, cc, cc, cc, cc, cc, cH])
        np.testing.assert_array_equal(params_dst, [cc, cc, cc, cc, cc, cc, cO])
        np.testing.assert_array_equal(params_uni, [OH])

        # test that vjp_fn works
        vjp_fn(np.random.rand(*combined_params.shape))

    def test_nonbonded(self):

        # leaving benzene H unmapped, and phenol OH unmapped
        core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
        ], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)
        x_a = get_romol_conf(self.mol_a)
        x_b = get_romol_conf(self.mol_b)

        # test interpolation of coordinates.
        x_src, x_dst = st.interpolate_params(x_a, x_b)
        x_avg = np.mean([x_src, x_dst], axis=0)

        assert x_avg.shape == (st.get_num_atoms(), 3)

        np.testing.assert_array_equal((x_a[:6] + x_b[:6])/2, x_avg[:6]) # C
        np.testing.assert_array_equal(x_a[6], x_avg[6]) # H
        np.testing.assert_array_equal(x_b[6:], x_avg[7:]) # OH

        res = st.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        params, vjp_fn, pot_c = jax.vjp(
            st.parameterize_nonbonded,
            self.ff.q_handle.params,
            self.ff.lj_handle.params,
            has_aux=True
        )

        vjp_fn(np.random.rand(*params.shape))

        assert params.shape == (2*st.get_num_atoms(), 3) # qlj

        # test interpolation of parameters
        bt_a = topology.BaseTopology(self.mol_a, self.ff)
        qlj_a, pot_a = bt_a.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)
        bt_b = topology.BaseTopology(self.mol_b, self.ff)
        qlj_b, pot_b = bt_b.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        n_base_params = len(params)//2 # params is actually interpolated, so its 2x number of base params

        qlj_c = np.mean([
            params[:n_base_params],
            params[n_base_params:]],
        axis=0)

        params_src = params[:n_base_params]
        params_dst = params[n_base_params:]

        # core testing
        np.testing.assert_array_equal(qlj_a[:6], params_src[:6])
        np.testing.assert_array_equal(qlj_b[:6], params_dst[:6])


        # test r-group in A
        np.testing.assert_array_equal(qlj_a[6], params_src[6])
        np.testing.assert_array_equal(np.array([0, qlj_a[6][1], 0]), params_dst[6])

        # test r-group in B
        np.testing.assert_array_equal(qlj_b[6:], params_dst[7:])
        np.testing.assert_array_equal(np.array([
            [0, qlj_b[6][1], 0],
            [0, qlj_b[7][1], 0]]), params_src[7:]
        )


    def test_nonbonded_optimal_map(self):
        """Similar test as test_nonbonbed, ie. assert that coordinates and nonbonded parameters
        can be averaged in benzene -> phenol transformation. However, use the maximal mapping possible."""

        # map benzene H to phenol O, leaving a dangling phenol H
        core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6]
        ], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)
        x_a = get_romol_conf(self.mol_a)
        x_b = get_romol_conf(self.mol_b)

        # test interpolation of coordinates.
        x_src, x_dst = st.interpolate_params(x_a, x_b)
        x_avg = np.mean([x_src, x_dst], axis=0)

        assert x_avg.shape == (st.get_num_atoms(), 3)

        np.testing.assert_array_equal((x_a[:7] + x_b[:7])/2, x_avg[:7]) # core parts
        np.testing.assert_array_equal(x_b[-1], x_avg[7]) # dangling H

        params, vjp_fn, pot_c = jax.vjp(
            st.parameterize_nonbonded,
            self.ff.q_handle.params,
            self.ff.lj_handle.params,
            has_aux=True
        )

        vjp_fn(np.random.rand(*params.shape))

        assert params.shape == (2*st.get_num_atoms(), 3) # qlj

        # test interpolation of parameters
        bt_a = topology.BaseTopology(self.mol_a, self.ff)
        qlj_a, pot_a = bt_a.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)
        bt_b = topology.BaseTopology(self.mol_b, self.ff)
        qlj_b, pot_b = bt_b.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        n_base_params = len(params)//2 # params is actually interpolated, so its 2x number of base params

        qlj_c = np.mean([params[:n_base_params], params[n_base_params:]], axis=0)

        params_src = params[:n_base_params]
        params_dst = params[n_base_params:]

        # core testing
        np.testing.assert_array_equal(qlj_a[:7], params_src[:7])
        np.testing.assert_array_equal(qlj_b[:7], params_dst[:7])

        # r-group atoms in A are all part of the core. so no testing is needed.

        # test r-group in B
        np.testing.assert_array_equal(qlj_b[7], params_dst[8])
        np.testing.assert_array_equal(np.array([0, qlj_b[7][1], 0]), params_src[8])


class TestFactorizability(unittest.TestCase):

    def test_bad_factor(self):
        # test a bad mapping that results in a non-cancellable endpoint
        suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
        all_mols = [x for x in suppl]
        mol_a = all_mols[0]
        mol_b = all_mols[1]

        ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_recharge.py').read())
        ff = Forcefield(ff_handlers)

        core = np.array([[ 4,  1],
            [ 5,  2],
            [ 6,  3],
            [ 7,  4],
            [ 8,  5],
            [ 9,  6],
            [10,  7],
            [11,  8],
            [12,  9],
            [13, 10],
            [15, 11],
            [16, 12],
            [18, 14],
            [34, 31],
            [17, 13],
            [23, 23],
            [33, 30],
            [32, 28],
            [31, 27],
            [30, 26],
            [19, 15],
            [20, 16],
            [21, 17]]
        )

        with self.assertRaises(topology.AtomMappingError):
            st = topology.SingleTopology(mol_a, mol_b, core, ff)

    def test_good_factor(self):
        # test a good mapping
        suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
        all_mols = [x for x in suppl]
        mol_a = all_mols[1]
        mol_b = all_mols[4]

        ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_recharge.py').read())
        ff = Forcefield(ff_handlers)

        core = np.array([[ 0,  0],
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
           [21, 22]]
        )

        st = topology.SingleTopology(mol_a, mol_b, core, ff)

        # test that the vjps work
        _ = jax.vjp(st.parameterize_harmonic_bond, ff.hb_handle.params, has_aux=True)
        _ = jax.vjp(st.parameterize_harmonic_angle, ff.ha_handle.params, has_aux=True)
        _ = jax.vjp(st.parameterize_proper_torsion, ff.pt_handle.params, has_aux=True)
        _ = jax.vjp(st.parameterize_improper_torsion, ff.it_handle.params, has_aux=True)
        _ = jax.vjp(st.parameterize_nonbonded, ff.q_handle.params, ff.lj_handle.params, has_aux=True)

class TestHostGuest(unittest.TestCase):

    def test_host_guest_single_topology(self):
        # test that we can properly build a single topology host guest system.

        suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
        all_mols = [x for x in suppl]
        mol_a = all_mols[1]
        mol_b = all_mols[4]

        # host_system = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')[0]
        host_system, host_coords, _, _, host_box, _ = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')

        host_box += np.eye(3)*0.1 # BFGS this later

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

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
        top = topology.SingleTopology(mol_a, mol_b, core, ff)

        minimized_host_coords = minimizer.minimize_host_4d(mol_a, host_system, host_coords, ff, host_box)

        print(minimized_host_coords)
        # assert 0

        hgt = topology.HostGuestTopology(host_bps, top)

        ff_tuples = [
            [hgt.parameterize_harmonic_bond, (ff.hb_handle,)],
            [hgt.parameterize_harmonic_angle, (ff.ha_handle,)],
            [hgt.parameterize_periodic_torsion, (ff.pt_handle, ff.it_handle)],
            [hgt.parameterize_nonbonded, (ff.q_handle, ff.lj_handle)]
        ]

        final_potentials = []

        for fn, handles in ff_tuples:
            combined_params, vjp_fn, combined_potential = jax.vjp(fn, *[handle.params for handle in handles], has_aux=True)
            final_potentials.append(combined_potential.bind(combined_params))


        ligand_masses_a = [a.GetMass() for a in mol_a.GetAtoms()]
        ligand_masses_b = [b.GetMass() for b in mol_b.GetAtoms()]

        ligand_coords_a = get_romol_conf(mol_a)
        ligand_coords_b = get_romol_conf(mol_b)

        combined_masses = np.concatenate([host_masses, np.mean(top.interpolate_params(ligand_masses_a, ligand_masses_b), axis=0)])
        combined_coords = np.concatenate([minimized_host_coords, np.mean(top.interpolate_params(ligand_coords_a, ligand_coords_b), axis=0)])

        seed = np.random.randint(np.iinfo(np.int32).max)

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



        num_batches = 20
        steps_per_batch = 1000
        seconds_per_day = 86400
        batch_times = []

        lamb = 0.5

        lambda_schedule = np.ones(steps_per_batch)*lamb

        print("START STEP")
        # run once before timer starts
        ctxt.multiple_steps(lambda_schedule)

        start = time.time()

        for batch in range(num_batches):

            # time the current batch
            batch_start = time.time()
            ctxt.multiple_steps(lambda_schedule)
            batch_end = time.time()

            delta = batch_end - batch_start

            batch_times.append(delta)

            steps_per_second = steps_per_batch / np.mean(batch_times)
            steps_per_day = steps_per_second*seconds_per_day

            dt = 1.5e-3
            ps_per_day = dt*steps_per_day
            ns_per_day = ps_per_day*1e-3

            print(f'steps per second: {steps_per_second:.3f}')
            print(f'ns per day: {ns_per_day:.3f}')


            # combined_potential = fn(*[handle.params for handle in handles])
            # final_potentials.append(guest_potential.bind(guest_params))
            # final_vjp_and_handles.append((vjp_fn, handle))

        print(ctxt.get_x_t())