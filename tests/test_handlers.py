from jax.config import config

config.update("jax_enable_x64", True)
import jax

import pytest
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from ff.handlers import nonbonded, bonded
from ff.handlers.deserialize import deserialize_handlers
from ff.charges import AM1CCC_CHARGES

import functools


def test_harmonic_bond():

    patterns = [
        ["[#6X4:1]-[#6X4:2]", 0.1, 0.2],
        ["[#6X4:1]-[#6X3:2]", 99.0, 99.0],
        ["[#6X4:1]-[#6X3:2]=[#8X1+0]", 99.0, 99.0],
        ["[#6X3:1]-[#6X3:2]", 99.0, 99.0],
        ["[#6X3:1]:[#6X3:2]", 99.0, 99.0],
        ["[#6X3:1]=[#6X3:2]", 99.0, 99.0],
        ["[#6:1]-[#7:2]", 0.1, 0.2],
        ["[#6X3:1]-[#7X3:2]", 99.0, 99.0],
        ["[#6X4:1]-[#7X3:2]-[#6X3]=[#8X1+0]", 99.0, 99.0],
        ["[#6X3:1](=[#8X1+0])-[#7X3:2]", 99.0, 99.0],
        ["[#6X3:1]-[#7X2:2]", 99.0, 99.0],
        ["[#6X3:1]:[#7X2,#7X3+1:2]", 99.0, 99.0],
        ["[#6X3:1]=[#7X2,#7X3+1:2]", 99.0, 99.0],
        ["[#6:1]-[#8:2]", 99.0, 99.0],
        ["[#6X3:1]-[#8X1-1:2]", 99.0, 99.0],
        ["[#6X4:1]-[#8X2H0:2]", 0.3, 0.4],
        ["[#6X3:1]-[#8X2:2]", 99.0, 99.0],
        ["[#6X3:1]-[#8X2H1:2]", 99.0, 99.0],
        ["[#6X3a:1]-[#8X2H0:2]", 99.0, 99.0],
        ["[#6X3:1](=[#8X1])-[#8X2H0:2]", 99.0, 99.0],
        ["[#6:1]=[#8X1+0,#8X2+1:2]", 99.0, 99.0],
        ["[#6X3:1](~[#8X1])~[#8X1:2]", 99.0, 99.0],
        ["[#6X3:1]~[#8X2+1:2]~[#6X3]", 99.0, 99.0],
        ["[#6X2:1]-[#6:2]", 99.0, 99.0],
        ["[#6X2:1]-[#6X4:2]", 99.0, 99.0],
        ["[#6X2:1]=[#6X3:2]", 99.0, 99.0],
        ["[#6:1]#[#7:2]", 99.0, 99.0],
        ["[#6X2:1]#[#6X2:2]", 99.0, 99.0],
        ["[#6X2:1]-[#8X2:2]", 99.0, 99.0],
        ["[#6X2:1]-[#7:2]", 99.0, 99.0],
        ["[#6X2:1]=[#7:2]", 99.0, 99.0],
        ["[#16:1]=[#6:2]", 99.0, 99.0],
        ["[#6X2:1]=[#16:2]", 99.0, 99.0],
        ["[#7:1]-[#7:2]", 99.0, 99.0],
        ["[#7X3:1]-[#7X2:2]", 99.0, 99.0],
        ["[#7X2:1]-[#7X2:2]", 99.0, 99.0],
        ["[#7:1]:[#7:2]", 99.0, 99.0],
        ["[#7:1]=[#7:2]", 99.0, 99.0],
        ["[#7+1:1]=[#7-1:2]", 99.0, 99.0],
        ["[#7:1]#[#7:2]", 99.0, 99.0],
        ["[#7:1]-[#8X2:2]", 99.0, 99.0],
        ["[#7:1]~[#8X1:2]", 99.0, 99.0],
        ["[#8X2:1]-[#8X2:2]", 99.0, 99.0],
        ["[#16:1]-[#6:2]", 99.0, 99.0],
        ["[#16:1]-[#1:2]", 99.0, 99.0],
        ["[#16:1]-[#16:2]", 99.0, 99.0],
        ["[#16:1]-[#9:2]", 99.0, 99.0],
        ["[#16:1]-[#17:2]", 99.0, 99.0],
        ["[#16:1]-[#35:2]", 99.0, 99.0],
        ["[#16:1]-[#53:2]", 99.0, 99.0],
        ["[#16X2,#16X1-1,#16X3+1:1]-[#6X4:2]", 99.0, 99.0],
        ["[#16X2,#16X1-1,#16X3+1:1]-[#6X3:2]", 99.0, 99.0],
        ["[#16X2:1]-[#7:2]", 99.0, 99.0],
        ["[#16X2:1]-[#8X2:2]", 99.0, 99.0],
        ["[#16X2:1]=[#8X1,#7X2:2]", 99.0, 99.0],
        ["[#16X4,#16X3!+1:1]-[#6:2]", 99.0, 99.0],
        ["[#16X4,#16X3:1]~[#7:2]", 99.0, 99.0],
        ["[#16X4,#16X3:1]-[#8X2:2]", 99.0, 99.0],
        ["[#16X4,#16X3:1]~[#8X1:2]", 99.0, 99.0],
        ["[#15:1]-[#1:2]", 99.0, 99.0],
        ["[#15:1]~[#6:2]", 99.0, 99.0],
        ["[#15:1]-[#7:2]", 99.0, 99.0],
        ["[#15:1]=[#7:2]", 99.0, 99.0],
        ["[#15:1]~[#8X2:2]", 99.0, 99.0],
        ["[#15:1]~[#8X1:2]", 99.0, 99.0],
        ["[#16:1]-[#15:2]", 99.0, 99.0],
        ["[#15:1]=[#16X1:2]", 99.0, 99.0],
        ["[#6:1]-[#9:2]", 99.0, 99.0],
        ["[#6X4:1]-[#9:2]", 0.6, 0.7],
        ["[#6:1]-[#17:2]", 99.0, 99.0],
        ["[#6X4:1]-[#17:2]", 99.0, 99.0],
        ["[#6:1]-[#35:2]", 99.0, 99.0],
        ["[#6X4:1]-[#35:2]", 99.0, 99.0],
        ["[#6:1]-[#53:2]", 99.0, 99.0],
        ["[#6X4:1]-[#53:2]", 99.0, 99.0],
        ["[#7:1]-[#9:2]", 99.0, 99.0],
        ["[#7:1]-[#17:2]", 99.0, 99.0],
        ["[#7:1]-[#35:2]", 99.0, 99.0],
        ["[#7:1]-[#53:2]", 99.0, 99.0],
        ["[#15:1]-[#9:2]", 99.0, 99.0],
        ["[#15:1]-[#17:2]", 99.0, 99.0],
        ["[#15:1]-[#35:2]", 99.0, 99.0],
        ["[#15:1]-[#53:2]", 99.0, 99.0],
        ["[#6X4:1]-[#1:2]", 99.0, 99.0],
        ["[#6X3:1]-[#1:2]", 99.0, 99.0],
        ["[#6X2:1]-[#1:2]", 99.0, 99.0],
        ["[#7:1]-[#1:2]", 99.0, 99.0],
        ["[#8:1]-[#1:2]", 99.0, 99.1],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2]] for x in patterns])
    props = None
    hbh = bonded.HarmonicBondHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("C1CNCOC1F")

    bond_params, bond_idxs = hbh.parameterize(mol)

    assert bond_idxs.shape == (mol.GetNumBonds(), 2)
    assert bond_params.shape == (mol.GetNumBonds(), 2)

    bonded_param_adjoints = np.random.randn(*bond_params.shape)

    bond_params_new, bond_vjp_fn, bond_idxs_new = jax.vjp(
        functools.partial(hbh.partial_parameterize, mol=mol), hbh.params, has_aux=True
    )

    np.testing.assert_array_equal(bond_params_new, bond_params)
    np.testing.assert_array_equal(bond_idxs_new, bond_idxs)

    # test that we can use the adjoints
    ff_adjoints = bond_vjp_fn(bonded_param_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(bond_params > 90)
    assert np.all(ff_adjoints[mask] == 0.0) == True


def test_proper_torsion():

    # proper torsions have a variadic number of terms

    patterns = [
        ["[*:1]-[#6X3:2]=[#6X3:3]-[*:4]", [[99.0, 99.0, 99.0]]],
        ["[*:1]-[#6X3:2]=[#6X3:3]-[#35:4]", [[99.0, 99.0, 99.0]]],
        ["[#9:1]-[#6X3:2]=[#6X3:3]-[#35:4]", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        ["[#35:1]-[#6X3:2]=[#6X3:3]-[#35:4]", [[7.0, 8.0, 9.0], [1.0, 3.0, 5.0], [4.0, 4.0, 4.0]]],
        ["[#9:1]-[#6X3:2]=[#6X3:3]-[#9:4]", [[7.0, 8.0, 9.0]]],
    ]

    smirks = [x[0] for x in patterns]
    params = [x[1] for x in patterns]
    props = None

    pth = bonded.ProperTorsionHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("FC(Br)=C(Br)F")

    torsion_params, torsion_idxs = pth.parameterize(mol)

    assert torsion_idxs.shape == (8, 4)
    assert torsion_params.shape == (8, 3)

    torsion_params_new, torsion_vjp_fn, torsion_idxs_new = jax.vjp(
        functools.partial(pth.partial_parameterize, mol=mol), pth.params, has_aux=True
    )

    np.testing.assert_array_equal(torsion_params_new, torsion_params)
    np.testing.assert_array_equal(torsion_idxs_new, torsion_idxs)

    torsion_param_adjoints = np.random.randn(*torsion_params.shape)

    ff_adjoints = torsion_vjp_fn(torsion_param_adjoints)[0]

    mask = np.argwhere(torsion_params > 90)
    assert np.all(ff_adjoints[mask] == 0.0) == True


def test_improper_torsion():

    patterns = [
        ["[*:1]~[#6X3:2](~[*:3])~[*:4]", 1.5341333333333333, 3.141592653589793, 2.0],
        ["[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]", 99.0, 99.0, 99.0],
        ["[*:1]~[#7X3$(*~[#15,#16](!-[*])):2](~[*:3])~[*:4]", 99.0, 99.0, 99.0],
        ["[*:1]~[#7X3$(*~[#6X3]):2](~[*:3])~[*:4]", 1.3946666666666667, 3.141592653589793, 2.0],
        ["[*:1]~[#7X3$(*~[#7X2]):2](~[*:3])~[*:4]", 99.0, 99.0, 99.0],
        ["[*:1]~[#7X3$(*@1-[*]=,:[*][*]=,:[*]@1):2](~[*:3])~[*:4]", 99.0, 99.0, 99.0],
        ["[*:1]~[#6X3:2](=[#7X2,#7X3+1:3])~[#7:4]", 99.0, 99.0, 99.0],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2], x[3]] for x in patterns])
    props = None
    imp_handler = bonded.ImproperTorsionHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("CNC(C)=O")  # peptide
    mol = Chem.AddHs(mol)

    torsion_params, torsion_idxs = imp_handler.parameterize(mol)

    assert torsion_idxs.shape[0] == 6  # we expect two sets of impropers, each with 3 components.
    assert torsion_idxs.shape[1] == 4

    assert torsion_params.shape[0] == 6
    assert torsion_params.shape[1] == 3

    torsion_params_new, torsion_vjp_fn, torsion_idxs_new = jax.vjp(
        functools.partial(imp_handler.partial_parameterize, mol=mol), imp_handler.params, has_aux=True
    )

    np.testing.assert_array_equal(torsion_params_new, torsion_params)
    np.testing.assert_array_equal(torsion_idxs_new, torsion_idxs)

    param_adjoints = np.random.randn(*torsion_params.shape)

    # # test that we can use the adjoints
    ff_adjoints = torsion_vjp_fn(param_adjoints)[0]

    # # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(torsion_params > 90)
    assert np.all(ff_adjoints[mask] == 0.0) == True


def test_exclusions():

    mol = Chem.MolFromSmiles("FC(F)=C(F)F")
    exc_idxs, scales = nonbonded.generate_exclusion_idxs(mol, scale12=0.0, scale13=0.2, scale14=0.5)

    for pair, scale in zip(exc_idxs, scales):
        src, dst = pair
        assert src < dst

    expected_idxs = np.array(
        [
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
            [4, 5],
        ]
    )

    np.testing.assert_equal(exc_idxs, expected_idxs)

    expected_scales = [0.0, 0.2, 0.2, 0.5, 0.5, 0.0, 0.0, 0.2, 0.2, 0.2, 0.5, 0.5, 0.0, 0.0, 0.2]
    np.testing.assert_equal(scales, expected_scales)


def test_am1_bcc():
    # currently takes no parameters
    smirks = []
    params = []
    props = None

    am1h = nonbonded.AM1BCCHandler(smirks, params, props)
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CNCOC1F"))
    AllChem.EmbedMolecule(mol)
    charges = am1h.parameterize(mol)

    assert len(charges) == mol.GetNumAtoms()

    new_charges, vjp_fn = jax.vjp(functools.partial(am1h.partial_parameterize, mol=mol))

    # charges_adjoints = np.random.randn(*charges.shape)

    # assert vjp_fn(charges_adjoints) == None


def test_am1_ccc():

    patterns = [
        ["[#6X4:1]-[#1:2]", 0.46323257920556493],
        ["[#6X3$(*=[#8,#16]):1]-[#6a:2]", 0.24281402370571598],
        ["[#6X3$(*=[#8,#16]):1]-[#8X1,#8X2:2]", 1.0620166764992722],
        ["[#6X3$(*=[#8,#16]):1]=[#8X1$(*=[#6X3]-[#8X2]):2]", 2.227759732057297],
        ["[#6X3$(*=[#8,#16]):1]=[#8X1,#8X2:2]", 2.8182928673804217],
        ["[#6a:1]-[#8X1,#8X2:2]", 0.5315976926761063],
        ["[#6a:1]-[#1:2]", 0.0],
        ["[#6a:1]:[#6a:2]", 0.0],
        ["[#6a:1]:[#6a:2]", 0.0],
        ["[#8X1,#8X2:1]-[#1:2]", -2.3692047944101415],
        ["[#16:1]-[#8:2]", 99.0],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([x[1] for x in patterns])
    props = None
    mol_sdf = """
     RDKit          3D

 21 21  0  0  0  0  0  0  0  0999 V2000
    1.9775   -1.1921   -0.0420 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.0482   -0.0099    0.4175 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.2666    0.4539    1.1431 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.9469    0.7961    0.2044 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1834    0.3655   -0.4688 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3446    0.4891   -1.8261 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4607    0.0654   -2.5014 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4866   -0.5199   -1.7874 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3616   -0.6611   -0.4259 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2246   -0.2274    0.2480 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1499   -0.4031    1.6843 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1061   -0.9387    2.2814 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0450    0.0107    2.3913 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.6820    1.3873    0.6660 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.0604    0.6688    2.2138 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.0231   -0.3405    1.1004 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4612    0.9512   -2.4027 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5273    0.1914   -3.5726 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.3631   -0.8543   -2.3065 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.1913   -1.1293    0.1168 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0216    0.8969    2.8664 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  2  4  1  0
  4  5  1  0
  5  6  2  0
  6  7  1  0
  7  8  2  0
  8  9  1  0
  9 10  2  0
 10 11  1  0
 11 12  2  0
 11 13  1  0
 10  5  1  0
  3 14  1  0
  3 15  1  0
  3 16  1  0
  6 17  1  0
  7 18  1  0
  8 19  1  0
  9 20  1  0
 13 21  1  0
M  END
$$$$
    """
    am1h = nonbonded.AM1CCCHandler(smirks, params, props)
    mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False)
    es_params = am1h.parameterize(mol)

    # TBD update with AM1 Symmetrize=True
    ligand_params = np.array(
        [
            -6.10948,
            7.08286,
            -1.13097,
            -4.3096,
            2.03822,
            -1.72492,
            -0.98493,
            -1.75663,
            -0.70852,
            -1.65444,
            7.76463,
            -6.49881,
            -7.10585,
            0.93707,
            0.93707,
            0.93707,
            1.79376,
            1.6654,
            1.69322,
            1.90986,
            5.22498,
        ]
    )

    np.testing.assert_almost_equal(es_params, ligand_params, decimal=5)

    new_es_params, es_vjp_fn = jax.vjp(functools.partial(am1h.partial_parameterize, mol=mol), params)

    np.testing.assert_array_equal(es_params, new_es_params)

    es_params_adjoints = np.random.randn(*es_params.shape)

    adjoints = es_vjp_fn(es_params_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(params > 90)
    assert np.all(adjoints[mask] == 0.0) == True

    import time

    start = time.time()
    es_params_from_cache = am1h.parameterize(mol)
    end = time.time()

    # second pass should be very fast
    assert end - start < 1.0

    # should be *exactly* identical since we're loading from cache
    np.testing.assert_array_equal(es_params_from_cache, es_params)


def test_simple_charge_handler():

    patterns = [
        ["[#1:1]", 99.0],
        ["[#1:1]-[#6X4]", 99.0],
        ["[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]", 99.0],
        ["[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]", 99.0],
        ["[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]", 99.0],
        ["[#1:1]-[#6X4]~[*+1,*+2]", 99.0],
        ["[#1:1]-[#6X3]", 99.0],
        ["[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]", 99.0],
        ["[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]", 99.0],
        ["[#1:1]-[#6X2]", 99.0],
        ["[#1:1]-[#7]", 99.0],
        ["[#1:1]-[#8]", 99.0],
        ["[#1:1]-[#16]", 99.0],
        ["[#6:1]", 0.7],
        ["[#6X2:1]", 99.0],
        ["[#6X4:1]", 0.1],
        ["[#8:1]", 99.0],
        ["[#8X2H0+0:1]", 0.5],
        ["[#8X2H1+0:1]", 99.0],
        ["[#7:1]", 0.3],
        ["[#16:1]", 99.0],
        ["[#15:1]", 99.0],
        ["[#9:1]", 1.0],
        ["[#17:1]", 99.0],
        ["[#35:1]", 99.0],
        ["[#53:1]", 99.0],
        ["[#3+1:1]", 99.0],
        ["[#11+1:1]", 99.0],
        ["[#19+1:1]", 99.0],
        ["[#37+1:1]", 99.0],
        ["[#55+1:1]", 99.0],
        ["[#9X0-1:1]", 99.0],
        ["[#17X0-1:1]", 99.0],
        ["[#35X0-1:1]", 99.0],
        ["[#53X0-1:1]", 99.0],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([x[1] for x in patterns])
    props = None

    sch = nonbonded.SimpleChargeHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("C1CNCOC1F")

    es_params = sch.parameterize(mol)

    ligand_params = np.array([0.1, 0.1, 0.3, 0.1, 0.5, 0.1, 1.0])  # C  # C  # N  # C  # O  # C  # F

    np.testing.assert_almost_equal(es_params, ligand_params)

    es_params_adjoints = np.random.randn(*es_params.shape)

    new_es_params, es_vjp_fn = jax.vjp(functools.partial(sch.partial_parameterize, mol=mol), params)

    np.testing.assert_array_equal(new_es_params, es_params)
    # test that we can use the adjoints
    adjoints = es_vjp_fn(es_params_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(params > 90)
    assert np.all(adjoints[mask] == 0.0) == True


@pytest.mark.skip("gbsa is deprecated")
def test_gbsa_handler():

    patterns = [
        ["[*:1]", 99.0, 99.0],
        ["[#1:1]", 99.0, 99.0],
        ["[#1:1]~[#7]", 99.0, 99.0],
        ["[#6:1]", 0.1, 0.2],
        ["[#7:1]", 0.3, 0.4],
        ["[#8:1]", 0.5, 0.6],
        ["[#9:1]", 0.7, 0.8],
        ["[#14:1]", 99.0, 99.0],
        ["[#15:1]", 99.0, 99.0],
        ["[#16:1]", 99.0, 99.0],
        ["[#17:1]", 99.0, 99.0],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2]] for x in patterns])
    props = {"foo": "bar"}
    gbh = nonbonded.GBSAHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("C1CNCOC1F")

    gb_params, gb_vjp_fn = gbh.parameterize(mol)

    ligand_params = np.array(
        [
            [0.1, 0.2],  # C
            [0.1, 0.2],  # C
            [0.3, 0.4],  # N
            [0.1, 0.2],  # C
            [0.5, 0.6],  # O
            [0.1, 0.2],  # C
            [0.7, 0.8],  # F
        ]
    )

    np.testing.assert_almost_equal(gb_params, ligand_params)

    gb_params_adjoints = np.random.randn(*gb_params.shape)

    # test that we can use the adjoints
    adjoints = gb_vjp_fn(gb_params_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(params > 90)
    assert np.all(adjoints[mask] == 0.0) == True


def test_am1_differences():

    ff_raw = open("ff/params/smirnoff_1_1_0_ccc.py").read()
    ff_handlers = deserialize_handlers(ff_raw)
    for ccc in ff_handlers:
        if isinstance(ccc, nonbonded.AM1CCCHandler):
            break

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    smi = "[H]c1c(OP(=S)(OC([H])([H])C([H])([H])[H])OC([H])([H])C([H])([H])[H])nc(C([H])(C([H])([H])[H])C([H])([H])[H])nc1C([H])([H])[H]"
    smi = "Clc1c(Cl)c(Cl)c(-c2c(Cl)c(Cl)c(Cl)c(Cl)c2Cl)c(Cl)c1Cl"
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", "Debug")
    assert AllChem.EmbedMolecule(mol) == 0

    suppl = [mol]
    am1 = nonbonded.AM1Handler([], [], None)
    bcc = nonbonded.AM1BCCHandler([], [], None)

    for mol in suppl:

        am1_params = am1.parameterize(mol)
        ccc_params = ccc.parameterize(mol)
        bcc_params = bcc.parameterize(mol)

        if np.sum(np.abs(ccc_params - bcc_params)) > 0.1:

            print(mol.GetProp("_Name"), Chem.MolToSmiles(mol))
            print("  AM1    CCC    BCC  S ?")
            for atom_idx, atom in enumerate(mol.GetAtoms()):
                a = am1_params[atom_idx]
                b = bcc_params[atom_idx]
                c = ccc_params[atom_idx]
                print("{:6.2f}".format(a), "{:6.2f}".format(c), "{:6.2f}".format(b), atom.GetSymbol(), end="")
                if np.abs(b - c) > 0.1:
                    print(" *")
                else:
                    print(" ")

            assert 0


def test_compute_or_load_am1_charges():
    """Loop over test ligands, asserting that charges are stored in expected property and that the same charges are
    returned on repeated calls"""

    # get some molecules
    cache_key = "AM1Cache"
    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    all_mols = [mol for mol in suppl]
    all_mols = all_mols[:5]  # truncate so that whole test is ~ 10 seconds

    # don't expect AM1 cache yet
    for mol in all_mols:
        assert not mol.HasProp(cache_key)

    # compute charges once
    fresh_am1_charges = [nonbonded.compute_or_load_am1_charges(mol) for mol in all_mols]

    # expect each mol to have AM1 cache now
    for mol in all_mols:
        assert mol.HasProp(cache_key)

    # expect the same charges as the first time around
    cached_am1_charges = [nonbonded.compute_or_load_am1_charges(mol) for mol in all_mols]
    for (fresh, cached) in zip(fresh_am1_charges, cached_am1_charges):
        np.testing.assert_array_equal(fresh, cached)


def test_bond_smirks_matches():
    """Loop over test ligands, asserting that
    * returned indices are in bounds
    * returned bonds are present in the mol"""
    # get some molecules
    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    all_mols = [mol for mol in suppl]

    # get some bond smirks
    smirks_list = [smirks for (smirks, param) in AM1CCC_CHARGES["patterns"]]

    for mol in all_mols:
        bond_idxs, type_idxs = nonbonded.bond_smirks_matches(mol, smirks_list)

        # assert indices in bounds
        assert (bond_idxs.min() >= 0) and (bond_idxs.max() < mol.GetNumAtoms())
        assert (type_idxs.min() >= 0) and (type_idxs.max() < len(smirks_list))

        # assert that bond_idxs are present in the mol
        bonds = set()
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bonds.add((a, b))
            bonds.add((b, a))
        for bond in bond_idxs:
            assert tuple(bond) in bonds


def test_apply_bond_charge_corrections():
    """Assert that applying random bond charge corrections does not change net charge"""

    n_random_tests = 100
    np.random.seed(2022)

    for _ in range(n_random_tests):
        n_atoms = np.random.randint(10, 50)

        # random initial charges with integral net charge
        initial_net_charge = np.random.randint(-5, +6)
        __initial_charges = np.random.randn(n_atoms)
        _initial_charges = __initial_charges - np.mean(__initial_charges)
        assert np.isclose(np.sum(_initial_charges), 0)
        initial_charges = _initial_charges + (initial_net_charge / n_atoms)
        assert np.isclose(np.sum(initial_charges), initial_net_charge, atol=1e-5), "initial net charge not integral"

        # arbitrary: duplicates okay, reversals okay, symmetry not required
        n_directed_bonds = np.random.randint(50, 100)
        bonds = np.random.randint(0, n_atoms, size=(n_directed_bonds, 2))
        deltas = np.random.randn(n_directed_bonds)

        final_charges = nonbonded.apply_bond_charge_corrections(initial_charges, bonds, deltas)
        final_net_charge = np.sum(final_charges)

        assert (final_charges != initial_charges).any()
        np.testing.assert_almost_equal(final_net_charge, initial_net_charge)


def test_lennard_jones_handler():

    patterns = [
        ["[#1:1]", 99.0, 999.0],
        ["[#1:1]-[#6X4]", 99.0, 999.0],
        ["[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]", 99.0, 999.0],
        ["[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]", 99.0, 999.0],
        ["[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]", 99.0, 999.0],
        ["[#1:1]-[#6X4]~[*+1,*+2]", 99.0, 999.0],
        ["[#1:1]-[#6X3]", 99.0, 999.0],
        ["[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]", 99.0, 999.0],
        ["[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]", 99.0, 999.0],
        ["[#1:1]-[#6X2]", 99.0, 999.0],
        ["[#1:1]-[#7]", 99.0, 999.0],
        ["[#1:1]-[#8]", 99.0, 999.0],
        ["[#1:1]-[#16]", 99.0, 999.0],
        ["[#6:1]", 0.7, 0.8],
        ["[#6X2:1]", 99.0, 999.0],
        ["[#6X4:1]", 0.1, 0.2],
        ["[#8:1]", 99.0, 999.0],
        ["[#8X2H0+0:1]", 0.5, 0.6],
        ["[#8X2H1+0:1]", 99.0, 999.0],
        ["[#7:1]", 0.3, 0.4],
        ["[#16:1]", 99.0, 999.0],
        ["[#15:1]", 99.0, 999.0],
        ["[#9:1]", 1.0, 1.1],
        ["[#17:1]", 99.0, 999.0],
        ["[#35:1]", 99.0, 999.0],
        ["[#53:1]", 99.0, 999.0],
        ["[#3+1:1]", 99.0, 999.0],
        ["[#11+1:1]", 99.0, 999.0],
        ["[#19+1:1]", 99.0, 999.0],
        ["[#37+1:1]", 99.0, 999.0],
        ["[#55+1:1]", 99.0, 999.0],
        ["[#9X0-1:1]", 99.0, 999.0],
        ["[#17X0-1:1]", 99.0, 999.0],
        ["[#35X0-1:1]", 99.0, 999.0],
        ["[#53X0-1:1]", 99.0, 999.0],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2]] for x in patterns])
    props = None

    ljh = nonbonded.LennardJonesHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("C1CNCOC1F")

    lj_params = ljh.parameterize(mol)

    ligand_params = np.array(
        [
            [0.1 / 2, 0.2],  # C
            [0.1 / 2, 0.2],  # C
            [0.3 / 2, 0.4],  # N
            [0.1 / 2, 0.2],  # C
            [0.5 / 2, 0.6],  # O
            [0.1 / 2, 0.2],  # C
            [1.0 / 2, 1.1],  # F
        ]
    )

    np.testing.assert_almost_equal(lj_params, ligand_params)

    lj_params_adjoints = np.random.randn(*lj_params.shape)

    new_lj_params, lj_vjp_fn = jax.vjp(functools.partial(ljh.partial_parameterize, mol=mol), params)

    np.testing.assert_array_equal(new_lj_params, lj_params)

    # test that we can use the adjoints
    adjoints = lj_vjp_fn(lj_params_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(params > 90)
    assert np.all(adjoints[mask] == 0.0) == True
