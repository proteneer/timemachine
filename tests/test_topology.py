from jax.config import config; config.update("jax_enable_x64", True)

import unittest
import numpy as np

from fe import topology as topology

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

import jax


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

        (params_src, params_dst, params_uni), vjp_fn, (potential_src, potential_dst, potential_uni) = jax.vjp(st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True)

        # test that vjp_fn works
        vjp_fn([np.random.rand(*params_src.shape), np.random.rand(*params_dst.shape), np.random.rand(*params_uni.shape)])

        assert len(potential_src.get_idxs() == 6)
        assert len(potential_dst.get_idxs() == 6)
        assert len(potential_uni.get_idxs() == 3)

        cc = self.ff.hb_handle.lookup_smirks("[#6X3:1]:[#6X3:2]")
        cH = self.ff.hb_handle.lookup_smirks("[#6X3:1]-[#1:2]")
        cO = self.ff.hb_handle.lookup_smirks("[#6X3:1]-[#8X2H1:2]")
        OH = self.ff.hb_handle.lookup_smirks("[#8:1]-[#1:2]")

        np.testing.assert_array_equal(params_src, [cc, cc, cc, cc, cc, cc])
        np.testing.assert_array_equal(params_dst, [cc, cc, cc, cc, cc, cc])
        np.testing.assert_array_equal(params_uni, [cH, cO, OH])


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

        (params_src, params_dst, params_uni), vjp_fn, (potential_src, potential_dst, potential_uni) = jax.vjp(st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True)

        assert len(potential_src.get_idxs() == 7)
        assert len(potential_dst.get_idxs() == 7)
        assert len(potential_uni.get_idxs() == 1)

        np.testing.assert_array_equal(params_src, [cc, cc, cc, cc, cc, cc, cH])
        np.testing.assert_array_equal(params_dst, [cc, cc, cc, cc, cc, cc, cO])
        np.testing.assert_array_equal(params_uni, [OH])

        # test that vjp_fn works
        vjp_fn([np.random.rand(*params_src.shape), np.random.rand(*params_dst.shape), np.random.rand(*params_uni.shape)])

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

        n_base_params = len(params//2) # params is actually interpolated, so its 2x number of base params

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

        n_base_params = len(params//2) # params is actually interpolated, so its 2x number of base params

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
