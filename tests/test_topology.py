from jax.config import config

config.update("jax_enable_x64", True)

import unittest

import jax
import numpy as np
from rdkit import Chem

from timemachine.fe import topology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield


class BenzenePhenolSparseTest(unittest.TestCase):
    def setUp(self, *args, **kwargs):

        suppl = Chem.SDMolSupplier("tests/data/benzene_phenol_sparse.sdf", removeHs=False)
        all_mols = [x for x in suppl]

        self.mol_a = all_mols[0]
        self.mol_b = all_mols[1]

        # atom type free
        self.ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        super(BenzenePhenolSparseTest, self).__init__(*args, **kwargs)

    def test_bonded(self):
        # other bonded terms use an identical protocol, so we assume they're correct if the harmonic bond tests pass.
        # leaving benzene H unmapped, and phenol OH unmapped
        core = np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
            ],
            dtype=np.int32,
        )

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)

        combined_params, vjp_fn, combined_potential = jax.vjp(
            st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True
        )

        # test that vjp_fn works
        vjp_fn(np.random.rand(*combined_params.shape))

        # we expect 15 bonds in total, of which 6 are duplicated.
        assert len(combined_potential.get_idxs() == 15)

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
        core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)

        combined_params, vjp_fn, combined_potential = jax.vjp(
            st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True
        )

        assert len(combined_potential.get_idxs() == 15)

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
        core = np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
            ],
            dtype=np.int32,
        )

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)
        x_a = get_romol_conf(self.mol_a)
        x_b = get_romol_conf(self.mol_b)

        # test interpolation of coordinates.
        x_src, x_dst = st.interpolate_params(x_a, x_b)
        x_avg = np.mean([x_src, x_dst], axis=0)

        assert x_avg.shape == (st.get_num_atoms(), 3)

        np.testing.assert_array_equal((x_a[:6] + x_b[:6]) / 2, x_avg[:6])  # C
        np.testing.assert_array_equal(x_a[6], x_avg[6])  # H
        np.testing.assert_array_equal(x_b[6:], x_avg[7:])  # OH

        # NOTE: unused result
        st.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        params, vjp_fn, pot_c = jax.vjp(
            st.parameterize_nonbonded, self.ff.q_handle.params, self.ff.lj_handle.params, has_aux=True
        )

        vjp_fn(np.random.rand(*params.shape))

        assert params.shape == (2 * st.get_num_atoms(), 3)  # qlj

        # test interpolation of parameters
        bt_a = topology.BaseTopology(self.mol_a, self.ff)
        qlj_a, pot_a = bt_a.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)
        bt_b = topology.BaseTopology(self.mol_b, self.ff)
        qlj_b, pot_b = bt_b.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        n_base_params = len(params) // 2  # params is actually interpolated, so its 2x number of base params

        # qlj_c = np.mean([params[:n_base_params], params[n_base_params:]], axis=0)

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
        np.testing.assert_array_equal(np.array([[0, qlj_b[6][1], 0], [0, qlj_b[7][1], 0]]), params_src[7:])

    def test_nonbonded_optimal_map(self):
        """Similar test as test_nonbonbed, ie. assert that coordinates and nonbonded parameters
        can be averaged in benzene -> phenol transformation. However, use the maximal mapping possible."""

        # map benzene H to phenol O, leaving a dangling phenol H
        core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)
        x_a = get_romol_conf(self.mol_a)
        x_b = get_romol_conf(self.mol_b)

        # test interpolation of coordinates.
        x_src, x_dst = st.interpolate_params(x_a, x_b)
        x_avg = np.mean([x_src, x_dst], axis=0)

        assert x_avg.shape == (st.get_num_atoms(), 3)

        np.testing.assert_array_equal((x_a[:7] + x_b[:7]) / 2, x_avg[:7])  # core parts
        np.testing.assert_array_equal(x_b[-1], x_avg[7])  # dangling H

        params, vjp_fn, pot_c = jax.vjp(
            st.parameterize_nonbonded, self.ff.q_handle.params, self.ff.lj_handle.params, has_aux=True
        )

        vjp_fn(np.random.rand(*params.shape))

        assert params.shape == (2 * st.get_num_atoms(), 3)  # qlj

        # test interpolation of parameters
        bt_a = topology.BaseTopology(self.mol_a, self.ff)
        qlj_a, pot_a = bt_a.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)
        bt_b = topology.BaseTopology(self.mol_b, self.ff)
        qlj_b, pot_b = bt_b.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        n_base_params = len(params) // 2  # params is actually interpolated, so its 2x number of base params

        # qlj_c = np.mean([params[:n_base_params], params[n_base_params:]], axis=0)

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
        suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
        all_mols = [x for x in suppl]
        mol_a = all_mols[0]
        mol_b = all_mols[1]

        ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        core = np.array(
            [
                [4, 1],
                [5, 2],
                [6, 3],
                [7, 4],
                [8, 5],
                [9, 6],
                [10, 7],
                [11, 8],
                [12, 9],
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
                [21, 17],
            ]
        )

        with self.assertRaises(topology.AtomMappingError):
            topology.SingleTopology(mol_a, mol_b, core, ff)

    def test_good_factor(self):
        # test a good mapping
        suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
        all_mols = [x for x in suppl]
        mol_a = all_mols[1]
        mol_b = all_mols[4]

        ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        core = np.array(
            [
                [0, 0],
                [2, 2],
                [1, 1],
                [6, 6],
                [5, 5],
                [4, 4],
                [3, 3],
                [15, 16],
                [16, 17],
                [17, 18],
                [18, 19],
                [19, 20],
                [20, 21],
                [32, 30],
                [26, 25],
                [27, 26],
                [7, 7],
                [8, 8],
                [9, 9],
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
                [21, 22],
            ]
        )

        st = topology.SingleTopology(mol_a, mol_b, core, ff)

        # test that the vjps work
        _ = jax.vjp(st.parameterize_harmonic_bond, ff.hb_handle.params, has_aux=True)
        _ = jax.vjp(st.parameterize_harmonic_angle, ff.ha_handle.params, has_aux=True)
        _ = jax.vjp(st.parameterize_periodic_torsion, ff.pt_handle.params, ff.it_handle.params, has_aux=True)
        _ = jax.vjp(st.parameterize_nonbonded, ff.q_handle.params, ff.lj_handle.params, has_aux=True)


class StandardQLJTyperTestCase(unittest.TestCase):
    def verify_smiles_typing(self, smi):
        romol = Chem.AddHs(Chem.MolFromSmiles(smi))
        qlj_types = topology.standard_qlj_typer(romol)
        assert len(qlj_types) == romol.GetNumAtoms()

    def test_bromine_in_mol(self):
        self.verify_smiles_typing("O=C1N(Br)C(=O)CC1")

    def test_iodine_in_mol(self):
        self.verify_smiles_typing("O=I(=O)OI(=O)=O")

    def verify_charge(self, smi, formal_charge):
        romol = Chem.AddHs(Chem.MolFromSmiles(smi))
        qlj_types = topology.standard_qlj_typer(romol)
        test_charges = qlj_types[:, 0]
        np.testing.assert_array_equal(test_charges, formal_charge / romol.GetNumAtoms())

    def test_charge_correctness(self):
        # test that charges are correct on various molecules based on formal charge
        self.verify_charge("CCC([O-])=O", -1)
        self.verify_charge("[O-]C1CCC([O-])CC1", -2)
        self.verify_charge("c1ccccc1", 0)
        self.verify_charge("N[NH2++]", 2)
        self.verify_charge("[NH3+][O-]", 0)  # zwitterion
