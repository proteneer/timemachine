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


class ImproperTorsionTest(unittest.TestCase):

    def setUp(self, *args, **kwargs):

        self.mol_a = Chem.MolFromSmiles("C(C)(C)(C)(C)") # CC4
        self.mol_b = Chem.MolFromSmiles("C(=C)(C)C")
        # the recharge parameters here do not need a geometry
        ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_recharge.py').read())

        self.ff = Forcefield(ff_handlers)

        super(ImproperTorsionTest, self).__init__(*args, **kwargs)

    def test_improper_torsions_full_core(self):
        core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
        ])

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)
        params, vjp_fn, potential = jax.vjp(st.parameterize_improper_torsion, self.ff.it_handle.params, has_aux=True)

        src = params[:len(params)//2]
        dst = params[len(params)//2:]

        assert len(params) == 6

        for k, _, _ in src:
            assert k == 0
        for k, _, _ in dst:
            assert k != 0

    def test_improper_torsions_part_core(self):
        core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
        ])

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)
        params, vjp_fn, potential = jax.vjp(st.parameterize_improper_torsion, self.ff.it_handle.params, has_aux=True)

        assert len(params) == 6

        for k, _, _ in params:
            assert k != 0


class CarboxylateInterpolationTest(unittest.TestCase):


    def setUp(self, *args, **kwargs):

        self.mol_a = Chem.MolFromSmiles("[O-]C(=O)C")
        self.mol_b = Chem.MolFromSmiles("C[N+](=O)[O-]")
        # atom type free
        ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_recharge.py').read())

        self.ff = Forcefield(ff_handlers)

        super(CarboxylateInterpolationTest, self).__init__(*args, **kwargs)


    def test_ketone_full_core_bonds(self):

        full_core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3]
        ])

        st = topology.SingleTopology(self.mol_a, self.mol_b, full_core, self.ff)
        params, vjp_fn, potential = jax.vjp(st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True)

        params_src = params[:len(params)//2]
        params_dst = params[len(params)//2:]

        C_sgl_C = self.ff.hb_handle.lookup_smirks("[#6X4:1]-[#6X3:2]=[#8X1+0]")
        C_any_O = self.ff.hb_handle.lookup_smirks("[#6X3:1](~[#8X1])~[#8X1:2]")

        O_any_N = self.ff.hb_handle.lookup_smirks("[#7:1]~[#8X1:2]")
        C_sgl_N = self.ff.hb_handle.lookup_smirks("[#6:1]-[#7:2]")
        CN = self.ff.hb_handle.lookup_smirks("[#6X3:1]=[#7X2,#7X3+1:2]")

        expected_src = [
            C_sgl_C,
            C_any_O,
            C_any_O
        ]

        expected_dst = [
            O_any_N,
            C_sgl_N,
            O_any_N
        ]

        np.testing.assert_array_equal(params_src, expected_src)
        np.testing.assert_array_equal(params_dst, expected_dst)

        O_C_O = self.ff.ha_handle.lookup_smirks("[#8X1:1]~[#6X3:2]~[#8:3]")
        X_C_X = self.ff.ha_handle.lookup_smirks("[*:1]~[#6X3:2]~[*:3]")
        CNO_NO_O = self.ff.ha_handle.lookup_smirks("[#6,#7,#8:1]-[#7X3:2](~[#8X1])~[#8X1:3]")
        O_N_O = self.ff.ha_handle.lookup_smirks("[#8X1:1]~[#7X3:2]~[#8X1:3]")

        params, vjp_fn, potential = jax.vjp(st.parameterize_harmonic_angle, self.ff.ha_handle.params, has_aux=True)

        params_src = params[:len(params)//2]
        params_dst = params[len(params)//2:]

        expected_src = [
            O_C_O,
            X_C_X,
            X_C_X
        ]

        expected_dst = [
            CNO_NO_O,
            CNO_NO_O,
            O_N_O
        ]

        np.testing.assert_array_equal(params_src, expected_src)
        np.testing.assert_array_equal(params_dst, expected_dst)

class BezenePhenolSparseTest(unittest.TestCase):

    def setUp(self, *args, **kwargs):

        suppl = Chem.SDMolSupplier('tests/data/benzene_phenol_sparse.sdf', removeHs=False)
        all_mols = [x for x in suppl]

        self.mol_a = all_mols[0]
        self.mol_b = all_mols[1]

        # atom type free
        ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_recharge.py').read())

        self.ff = Forcefield(ff_handlers)

        super(BezenePhenolSparseTest, self).__init__(*args, **kwargs)


    def test_torsions_part_core(self):

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

        params, vjp_fn, potential = jax.vjp(st.parameterize_improper_torsion, self.ff.it_handle.params, has_aux=True)

        # there are two sets of improper torsions (6 terms), which are interpolated twice = 12
        assert len(params) == 12

        for k, _, _ in params:
            assert k != 0

        params, vjp_fn, potential = jax.vjp(st.parameterize_proper_torsion, self.ff.pt_handle.params, has_aux=True)

        # every torsion should be complete
        for k, _, _ in params:
            assert k != 0 

        # every torsion is comprised of a single term for this pattern
        # we have 6 core torsions, 2 A torsions, 4 B torsions
        assert len(params) == (6+2+4)*2

    def test_torsions_full_core(self):

        # leaving benzene H unmapped, and phenol OH unmapped
        core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6] # map H to O
        ], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)

        params, vjp_fn, potential = jax.vjp(st.parameterize_improper_torsion, self.ff.it_handle.params, has_aux=True)

        # one set of improper torsions
        assert len(params) == 6

        for k, _, _ in params:
            assert k != 0

        params, vjp_fn, potential = jax.vjp(st.parameterize_proper_torsion, self.ff.it_handle.params, has_aux=True)

        for k, _, _ in params:
            assert k != 0

        # we have 8 core torsions, and 2 B torsions, with interpolation
        assert len(params) == (8+2)*2

    def test_nonbonded_part_core(self):

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

        params, vjp_fn, pot_c = jax.vjp(
            st.parameterize_nonbonded,
            self.ff.q_handle.params, 
            self.ff.lj_handle.params,
            has_aux=True
        )

        assert params.shape == (2*st.get_num_atoms(), 3) # qlj
        
        bt_a = topology.BaseTopology(self.mol_a, self.ff)
        qlj_a, pot_a = bt_a.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)
        bt_b = topology.BaseTopology(self.mol_b, self.ff)
        qlj_b, pot_b = bt_b.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        qlj_c = np.mean([params[:len(params)//2], params[len(params)//2:]], axis=0)

        np.testing.assert_array_equal((qlj_a[:6] + qlj_b[:6])/2, qlj_c[:6])
        np.testing.assert_array_equal(qlj_a[6], qlj_c[6]) # H
        np.testing.assert_array_equal(qlj_b[6:], qlj_c[7:]) # OH


class CompareDist(rdFMCS.MCSAtomCompare):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compare(self, p, mol1, atom1, mol2, atom2):
        x_i = mol1.GetConformer(0).GetPositions()[atom1]
        x_j = mol2.GetConformer(0).GetPositions()[atom2]
        if np.linalg.norm(x_i-x_j) > 0.5:
            return False
        else:
            return True

class TestLigandSet(unittest.TestCase):

    def test_hif2a_ligands_dry_run(self):
        suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
        # test every combination in a dry run to ensure correctness
        all_mols = [x for x in suppl][:4]

        ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_recharge.py').read())

        ff = Forcefield(ff_handlers)

        mcs_params = rdFMCS.MCSParameters()
        mcs_params.AtomTyper = CompareDist()

        for mol_a in all_mols:
            for mol_b in all_mols:

                res = rdFMCS.FindMCS(
                    [mol_a, mol_b],
                    mcs_params
                )
                
                pattern = Chem.MolFromSmarts(res.smartsString)
                core_a = mol_a.GetSubstructMatch(pattern)
                core_b = mol_b.GetSubstructMatch(pattern)
                core = np.stack([core_a, core_b], axis=-1)

                st = topology.SingleTopology(mol_a, mol_b, core, ff)
                params, vjp_fn, pot = jax.vjp(st.parameterize_harmonic_bond, ff.hb_handle.params, has_aux=True)
                params, vjp_fn, pot = jax.vjp(st.parameterize_harmonic_angle, ff.ha_handle.params, has_aux=True)
                params, vjp_fn, pot = jax.vjp(st.parameterize_proper_torsion, ff.pt_handle.params, has_aux=True)
                params, vjp_fn, pot = jax.vjp(st.parameterize_improper_torsion, ff.it_handle.params, has_aux=True)
                params, vjp_fn, pot = jax.vjp(st.parameterize_nonbonded, ff.q_handle.params, ff.lj_handle.params, has_aux=True)
