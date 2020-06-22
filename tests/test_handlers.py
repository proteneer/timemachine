from jax.config import config; config.update("jax_enable_x64", True)

import unittest
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from ff.handlers import nonbonded

class TestNonbondedHandlers(unittest.TestCase):

    def test_exclusions(self):

        mol = Chem.MolFromSmiles("FC(F)=C(F)F")
        exc_idxs, scales = nonbonded.generate_exclusion_idxs(
            mol,
            scale12=0.0,
            scale13=0.2,
            scale14=0.5
        )

        for pair, scale in zip(exc_idxs, scales):
            src, dst = pair
            assert src < dst

        expected_idxs = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 3],
            [2, 4],
            [2, 5],
            [3, 4],
            [3, 5],
            [4, 5]]
        )

        np.testing.assert_equal(exc_idxs, expected_idxs)


        expected_scales = [0., 0.2, 0.2, 0.5, 0.5, 0., 0., 0.2, 0.2, 0.2, 0.5, 0.5, 0., 0., 0.2]
        np.testing.assert_equal(scales, expected_scales)

    def test_am1_bcc(self):
        # currently takes no parameters
        am1h = nonbonded.AM1BCCHandler()
        mol = Chem.AddHs(Chem.MolFromSmiles("C1CNCOC1F"))
        AllChem.EmbedMolecule(mol)
        charges, vjp_fn = am1h.parameterize(mol)

        assert len(charges) == mol.GetNumAtoms()

        charges_adjoints = np.random.randn(*charges.shape)

        assert vjp_fn(charges_adjoints) == None

    def test_simple_charge_handler(self):

        patterns = [
            ['[#1:1]', 99.],
            ['[#1:1]-[#6X4]', 99.],
            ['[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]', 99.],
            ['[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]', 99.],
            ['[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]', 99.],
            ['[#1:1]-[#6X4]~[*+1,*+2]', 99.],
            ['[#1:1]-[#6X3]', 99.],
            ['[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]', 99.],
            ['[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]', 99.],
            ['[#1:1]-[#6X2]', 99.],
            ['[#1:1]-[#7]', 99.],
            ['[#1:1]-[#8]', 99.],
            ['[#1:1]-[#16]', 99.],
            ['[#6:1]', 0.7],
            ['[#6X2:1]', 99.],
            ['[#6X4:1]', 0.1],
            ['[#8:1]', 99.],
            ['[#8X2H0+0:1]', 0.5],
            ['[#8X2H1+0:1]', 99.],
            ['[#7:1]', 0.3],
            ['[#16:1]', 99.],
            ['[#15:1]', 99.],
            ['[#9:1]', 1.0],
            ['[#17:1]', 99.],
            ['[#35:1]', 99.],
            ['[#53:1]', 99.],
            ['[#3+1:1]', 99.],
            ['[#11+1:1]', 99.],
            ['[#19+1:1]', 99.],
            ['[#37+1:1]', 99.],
            ['[#55+1:1]', 99.],
            ['[#9X0-1:1]', 99.],
            ['[#17X0-1:1]', 99.],
            ['[#35X0-1:1]', 99.],
            ['[#53X0-1:1]', 99.],
        ]

        smirks = [x[0] for x in patterns]
        params = np.array([x[1] for x in patterns])

        sch = nonbonded.SimpleChargeHandler(smirks, params)

        mol = Chem.MolFromSmiles("C1CNCOC1F")

        NL = mol.GetNumAtoms()
        NP = 13
        aux_es_params = np.random.rand(NP,) + 10

        es_params, es_vjp_fn = sch.parameterize(mol, aux_es_params)

        assert es_params.shape == (NP + NL,)

        np.testing.assert_almost_equal(es_params[NL:], aux_es_params)

        ligand_params = np.array([
            0.1, # C
            0.1, # C
            0.3, # N
            0.1, # C
            0.5, # O
            0.1, # C
            1.0  # F
        ])

        np.testing.assert_almost_equal(es_params[:NL], ligand_params)

        es_params_adjoints = np.random.randn(*es_params.shape)

        # test that we can use the adjoints
        adjoints = es_vjp_fn(es_params_adjoints)[0]

        # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
        mask = np.argwhere(params > 90)
        assert np.all(adjoints[mask] == 0.0) == True

    def test_lennard_jones_handler(self):

        patterns = [
            ['[#1:1]', 99., 999.],
            ['[#1:1]-[#6X4]', 99., 999.],
            ['[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]', 99., 999.],
            ['[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]', 99., 999.],
            ['[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]', 99., 999.],
            ['[#1:1]-[#6X4]~[*+1,*+2]', 99., 999.],
            ['[#1:1]-[#6X3]', 99., 999.],
            ['[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]', 99., 999.],
            ['[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]', 99., 999.],
            ['[#1:1]-[#6X2]', 99., 999.],
            ['[#1:1]-[#7]', 99., 999.],
            ['[#1:1]-[#8]', 99., 999.],
            ['[#1:1]-[#16]', 99., 999.],
            ['[#6:1]', 0.7, 0.8],
            ['[#6X2:1]', 99., 999.],
            ['[#6X4:1]', 0.1, 0.2],
            ['[#8:1]', 99., 999.],
            ['[#8X2H0+0:1]', 0.5, 0.6],
            ['[#8X2H1+0:1]', 99., 999.],
            ['[#7:1]', 0.3, 0.4],
            ['[#16:1]', 99., 999.],
            ['[#15:1]', 99., 999.],
            ['[#9:1]', 1.0, 1.1],
            ['[#17:1]', 99., 999.],
            ['[#35:1]', 99., 999.],
            ['[#53:1]', 99., 999.],
            ['[#3+1:1]', 99., 999.],
            ['[#11+1:1]', 99., 999.],
            ['[#19+1:1]', 99., 999.],
            ['[#37+1:1]', 99., 999.],
            ['[#55+1:1]', 99., 999.],
            ['[#9X0-1:1]', 99., 999.],
            ['[#17X0-1:1]', 99., 999.],
            ['[#35X0-1:1]', 99., 999.],
            ['[#53X0-1:1]', 99., 999.],
        ]

        smirks = [x[0] for x in patterns]
        params = np.array([[x[1], x[2]] for x in patterns])

        ljh = nonbonded.LennardJonesHandler(smirks, params)

        mol = Chem.MolFromSmiles("C1CNCOC1F")

        NL = mol.GetNumAtoms()
        NP = 13
        aux_lj_params = np.random.rand(NP, 2) + 10

        lj_params, lj_vjp_fn = ljh.parameterize(mol, aux_lj_params)

        assert lj_params.shape == (NP + NL, 2)

        np.testing.assert_almost_equal(lj_params[NL:], aux_lj_params)

        ligand_params = np.array([
            [0.1, 0.2], # C
            [0.1, 0.2], # C
            [0.3, 0.4], # N
            [0.1, 0.2], # C
            [0.5, 0.6], # O
            [0.1, 0.2], # C
            [1.0, 1.1]  # F
        ])

        np.testing.assert_almost_equal(lj_params[:NL], ligand_params)

        lj_params_adjoints = np.random.randn(*lj_params.shape)

        # test that we can use the adjoints
        adjoints = lj_vjp_fn(lj_params_adjoints)[0]

        # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
        mask = np.argwhere(params > 90)
        assert np.all(adjoints[mask] == 0.0) == True
