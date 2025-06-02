import base64
import functools
import pickle
from collections import defaultdict
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import flatten_util
from openmm import app
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

from timemachine.constants import DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, ONE_4PI_EPS0
from timemachine.fe import topology, utils
from timemachine.ff import Forcefield
from timemachine.ff.charges import AM1CCC_CHARGES
from timemachine.ff.handlers import bonded, nonbonded
from timemachine.ff.handlers import utils as h_utils
from timemachine.md import builders
from timemachine.testsystems import fetch_freesolv
from timemachine.utils import path_to_internal_file

pytestmark = [pytest.mark.nocuda]

TEST_FEATURE_SIZE = 16


def set_nn_features(mol, seed=2024, feature_size=TEST_FEATURE_SIZE):
    """
    Add random NN features to the given mol for testing.
    """
    rng = np.random.default_rng(seed)
    num_atoms = mol.GetNumAtoms()
    bond_idxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

    # the handler expected both directions of each bond to be present
    bond_idxs_ = []
    for bond_idx in bond_idxs:
        bond_idxs_.append(bond_idx)
        bond_idxs_.append(bond_idx[::-1])
    bond_idxs = np.array(bond_idxs_)
    num_bonds = bond_idxs.shape[0]

    # create random features
    atom_features = rng.random((num_atoms, feature_size))
    bond_src_features = rng.random((num_bonds, feature_size))
    bond_dst_features = rng.random((num_bonds, feature_size))

    # symmetrize features using charges as a guide
    # NOTE: This may not be what we want to do for the actual
    # features, but is simple enough for testing
    init_mol_charges = nonbonded.oe_assign_charges(mol, charge_model=nonbonded.AM1)
    sym_charge_idxs = defaultdict(list)
    for i, q in enumerate(init_mol_charges):
        sym_charge_idxs[float(q)].append(i)

    # all atoms in the sym. group should have the same features
    for sym_group in sym_charge_idxs.values():
        sym_atom_features = np.mean([atom_features[j] for j in sym_group], axis=0)
        for j in sym_group:
            atom_features[j, :] = sym_atom_features

    # all bonds in the paired sym. group should have the same features
    bond_features_by_sym_idx = defaultdict(list)
    for i, bond_idx in enumerate(bond_idxs):
        bond_feature = np.concatenate([bond_src_features[i], bond_dst_features[i]])
        q0 = float(init_mol_charges[bond_idx[0]])
        q1 = float(init_mol_charges[bond_idx[1]])
        bond_features_by_sym_idx[(q0, q1)].append(bond_feature)

    bond_features_ = []
    for i, bond_idx in enumerate(bond_idxs):
        q0 = float(init_mol_charges[bond_idx[0]])
        q1 = float(init_mol_charges[bond_idx[1]])
        bond_features_.append(np.mean(bond_features_by_sym_idx[(q0, q1)], axis=0))
    bond_features = np.array(bond_features_)

    bond_src_features = bond_features[:, :feature_size]
    bond_dst_features = bond_features[:, feature_size:]

    features = {
        "atom_features": atom_features,
        "bond_idxs": np.array(bond_idxs, dtype=int),
        "bond_src_features": bond_src_features,
        "bond_dst_features": bond_dst_features,
    }
    mol.SetProp(nonbonded.NN_FEATURES_PROPNAME, base64.b64encode(pickle.dumps(features)))


@pytest.fixture(scope="module")
def env_nn_args():
    # Fixture that returns the smirks, params, props for a `EnvironmentNNHandler`.
    rng = np.random.default_rng(seed=2024)
    feature_size = TEST_FEATURE_SIZE

    # init params and unflatten function
    layer_sizes = [feature_size * 4, 8, 1]
    weight_sizes = list(zip(np.array(layer_sizes)[1:], np.array(layer_sizes[:-1])))
    full_params = [rng.random(weight_size) for weight_size in weight_sizes]
    flat_params, unflatten = flatten_util.ravel_pytree(full_params)
    params = [flat_params]
    enc_unflatten_str = [base64.b64encode(pickle.dumps(unflatten))]

    # enumerate all supported residues
    all_res_names = []
    for res_name in h_utils.SMILES_BY_RES_NAME.keys():
        all_res_names.append(res_name)
        if res_name not in ["ACE", "NME"]:
            all_res_names.append("N" + res_name)
            all_res_names.append("C" + res_name)

    # set random nn features for each residue
    res_nn_props = {}
    for res_name in all_res_names:
        res_mol = h_utils.make_residue_mol_from_template(res_name)
        set_nn_features(res_mol, feature_size=feature_size)
        res_nn_props[res_name] = res_mol.GetProp(nonbonded.NN_FEATURES_PROPNAME)
    props = [base64.b64encode(pickle.dumps(res_nn_props)).decode("utf-8")]

    return enc_unflatten_str, params, props


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
    assert np.all(ff_adjoints[mask] == 0.0)

    # Check molecule with no bonds
    mol = Chem.MolFromSmiles("[Na+]")
    bond_params, bond_idxs = hbh.parameterize(mol)
    assert bond_idxs.shape == (0, 2)
    assert bond_params.shape == (0, 2)


def test_harmonic_angle():
    patterns = [
        ["[*:1]-[#8:2]-[*:3]", 0.1, 0.2],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2]] for x in patterns])
    props = None

    hah = bonded.HarmonicAngleHandler(smirks, params, props)
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    angle_params, angle_idxs = hah.parameterize(mol)
    assert angle_idxs.shape == (1, 3)
    assert angle_params.shape == (1, 3)

    # Check molecule with no angles
    hah = bonded.HarmonicAngleHandler(smirks, params, props)
    mol = Chem.MolFromSmiles("O=O")
    angle_params, angle_idxs = hah.parameterize(mol)
    assert angle_idxs.shape == (0, 3)
    assert angle_params.shape == (0, 3)


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
    assert np.all(ff_adjoints[mask] == 0.0)

    # assert expected shape when no matches are found
    null_smirks = ["[#1:1]~[#1:2]~[#1:3]~[#1:4]"]  # should never match anything
    null_params = np.zeros((len(null_smirks), 3))
    null_counts = np.ones(len(null_smirks), dtype=int)

    assigned_params, proper_idxs = bonded.ProperTorsionHandler.static_parameterize(
        null_params, null_smirks, null_counts, mol
    )
    assert assigned_params.shape == (0, 3)
    assert proper_idxs.shape == (0, 4)


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
    assert np.all(ff_adjoints[mask] == 0.0)

    # assert expected shape when no matches are found
    null_smirks = ["[#1:1]~[#1:2](~[#1:3])~[#1:4]"]  # should never match anything
    null_params = np.zeros((len(null_smirks), 3))

    assigned_params, improper_idxs = bonded.ImproperTorsionHandler.static_parameterize(null_params, null_smirks, mol)
    assert assigned_params.shape == (0, 3)
    assert improper_idxs.shape == (0, 4)

    for idxs in improper_idxs:
        assert idxs[0] < idxs[-1]


def test_exclusions():
    mol = Chem.MolFromSmiles("FC(F)=C(F)F")
    exc_idxs, scales = nonbonded.generate_exclusion_idxs(mol, scale12=0.0, scale13=0.2, scale14_q=0.25, scale14_lj=0.75)

    for pair, _ in zip(exc_idxs, scales):
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

    expected_scales = [
        [0.0, 0.0],
        [0.2, 0.2],
        [0.2, 0.2],
        [0.25, 0.75],
        [0.25, 0.75],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.2, 0.2],
        [0.2, 0.2],
        [0.2, 0.2],
        [0.25, 0.75],
        [0.25, 0.75],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.2, 0.2],
    ]
    np.testing.assert_equal(scales, expected_scales)


def test_am1bcc_parameterization():
    # currently takes no parameters
    smirks = []
    params = []
    props = None

    cache_key = nonbonded.AM1BCCELF10_CHARGE_CACHE
    am1h = nonbonded.AM1BCCHandler(smirks, params, props)
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CNCOC1F"))
    AllChem.EmbedMolecule(mol)

    assert not mol.HasProp(cache_key)

    charges = am1h.parameterize(mol)

    assert len(charges) == mol.GetNumAtoms()
    assert mol.HasProp(cache_key)

    cached_charges = am1h.parameterize(mol)
    np.testing.assert_equal(charges, cached_charges)

    new_charges, vjp_fn = jax.vjp(functools.partial(am1h.partial_parameterize, None, mol))

    # charges_adjoints = np.random.randn(*charges.shape)

    # assert vjp_fn(charges_adjoints) == None


def test_resp_parameterization():
    # currently takes no parameters
    smirks = []
    params = []
    props = None

    cache_key = nonbonded.RESP
    resp = nonbonded.RESPHandler(smirks, params, props)
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CNCOC1F"))
    AllChem.EmbedMolecule(mol)

    assert not mol.HasProp(cache_key)

    charges = resp.parameterize(mol)

    assert len(charges) == mol.GetNumAtoms()
    assert mol.HasProp(cache_key)

    cached_charges = resp.parameterize(mol)
    np.testing.assert_equal(charges, cached_charges)

    new_charges, vjp_fn = jax.vjp(functools.partial(resp.partial_parameterize, None, mol))


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

    # fmt: off
    ligand_params = np.array([
        -5.32765, 6.89144, -1.62909, -4.01409, 1.99603, -2.32088,
        -0.89641, -1.86649, -0.65583, -1.64725, 7.67398, -6.09888,
        -7.27287, 0.95122, 0.95122, 0.95122, 1.78079, 1.65102,
        1.74095, 1.87874, 5.26282,
        ]
    )
    # fmt: on

    np.testing.assert_almost_equal(es_params, ligand_params, decimal=5)

    new_es_params, es_vjp_fn = jax.vjp(functools.partial(am1h.partial_parameterize, mol=mol), params)

    np.testing.assert_array_equal(es_params, new_es_params)

    es_params_adjoints = np.random.randn(*es_params.shape)

    adjoints = es_vjp_fn(es_params_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(params > 90)
    assert np.all(adjoints[mask] == 0.0)

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
    assert np.all(adjoints[mask] == 0.0)


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

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true)
    mask = np.argwhere(params > 90)
    assert np.all(adjoints[mask] == 0.0)


def test_am1ccc_throws_error_on_phosphorus():
    """Temporary, until phosphorus patterns are added to AM1CCC port"""
    ff = Forcefield.load_default()

    # contains phosphorus
    smi = "[H]c1c(OP(=S)(OC([H])([H])C([H])([H])[H])OC([H])([H])C([H])([H])[H])nc(C([H])(C([H])([H])[H])C([H])([H])[H])nc1C([H])([H])[H]"
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))

    with pytest.raises(RuntimeError) as e:
        _ = ff.q_handle.parameterize(mol)
    assert "unsupported element" in str(e)


@pytest.mark.parametrize(
    "am1bcc_ff", ["smirnoff_1_1_0_am1bcc.py", "smirnoff_2_0_0_am1bcc.py", "smirnoff_2_2_0_am1bcc.py"]
)
def test_am1bcc_handles_phosphorus(am1bcc_ff):
    """Verify that the AM1BCC forcefields handle phosphorus, unlike the CCC forcefields"""
    ff = Forcefield.load_from_file(am1bcc_ff)

    # contains phosphorus
    smi = "[H]c1c(OP(=S)(OC([H])([H])C([H])([H])[H])OC([H])([H])C([H])([H])[H])nc(C([H])(C([H])([H])[H])C([H])([H])[H])nc1C([H])([H])[H]"
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))

    _ = ff.q_handle.parameterize(mol)

    lamb = 0.0

    base_topo = topology.BaseTopology(mol, ff)
    base_topo.parameterize_nonbonded(
        ff.q_handle.params,
        ff.q_handle_intra.params,
        ff.lj_handle.params,
        ff.lj_handle_intra.params,
        lamb,
    )


def test_am1_differences():
    ff = Forcefield.load_default()

    ccc = ff.q_handle
    assert isinstance(ccc, nonbonded.AM1CCCHandler)

    smi = "Clc1c(Cl)c(Cl)c(-c2c(Cl)c(Cl)c(Cl)c(Cl)c2Cl)c(Cl)c1Cl"
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    utils.set_mol_name(mol, "Debug")
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
                print(f"{a:6.2f}", f"{c:6.2f}", f"{b:6.2f}", atom.GetSymbol(), end="")
                if np.abs(b - c) > 0.1:
                    print(" *")
                else:
                    print(" ")

            assert 0


def test_am1elf10_conformer_independence():
    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = utils.read_sdf(path_to_ligand)

    # Pick a subset of molecules with chiral centers
    mols = [mols[0], mols[2], mols[3]]

    # need to assign so embedded molecules generated below
    # have the correct stereochemistry
    for mol in mols:
        rdmolops.AssignStereochemistryFrom3D(mol, confId=0, replaceExistingTags=True)

    am1elf10_charges = [nonbonded.oe_assign_charges(mol, charge_model=nonbonded.AM1ELF10) for mol in mols]

    # regenerate the conformations
    for mol in mols:
        AllChem.EmbedMolecule(mol, randomSeed=2022, useRandomCoords=True)

    new_am1elf10_charges = [nonbonded.oe_assign_charges(mol, charge_model=nonbonded.AM1ELF10) for mol in mols]

    # new conformations should have the same charges
    for orig_charges, new_charges in zip(am1elf10_charges, new_am1elf10_charges):
        delta_charges = np.abs(np.array(orig_charges) - np.array(new_charges))
        assert np.sum(delta_charges) == pytest.approx(0.0)


def test_trans_carboxlic_acid():
    # Test fallback to turn off hydrogen sampling if charge generation failed
    # due to trans-COOH
    with path_to_internal_file("timemachine.testsystems.data", "mobley_820789.sdf") as path_to_ligand:
        mols = utils.read_sdf(path_to_ligand)
    mol = mols[0]
    rdmolops.AssignStereochemistryFrom3D(mol, confId=0, replaceExistingTags=True)
    am1elf10_charges = nonbonded.oe_assign_charges(mol, charge_model=nonbonded.AM1ELF10)

    AllChem.EmbedMolecule(mol, randomSeed=2022, useRandomCoords=True)
    new_am1elf10_charges = nonbonded.oe_assign_charges(mol, charge_model=nonbonded.AM1ELF10)
    delta_charges = np.abs(am1elf10_charges - new_am1elf10_charges)
    assert np.sum(delta_charges) == pytest.approx(0.0)


def test_freesolv_failures():
    # Test failures for 3 cases in freesolv in openeye toolkits 2022.2.2
    with path_to_internal_file("timemachine.testsystems.data", "freesolv_omega_failures.sdf") as path_to_ligand:
        mols = utils.read_sdf(path_to_ligand)

    for mol in mols:
        am1elf10_charges = nonbonded.oe_assign_charges(mol, charge_model=nonbonded.AM1ELF10)
        assert am1elf10_charges is not None


def test_compute_or_load_oe_charges():
    """Loop over test ligands, asserting that charges are stored in expected property and that the same charges are
    returned on repeated calls"""

    # get some molecules
    cache_key = nonbonded.AM1ELF10_CHARGE_CACHE
    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = utils.read_sdf(path_to_ligand)

    mols = mols[:5]  # truncate so that whole test is ~ 10 seconds

    # don't expect AM1 cache yet
    for mol in mols:
        assert not mol.HasProp(cache_key)

    # compute charges once
    fresh_am1_charges = [nonbonded.compute_or_load_oe_charges(mol) for mol in mols]

    # expect each mol to have AM1 cache now
    for mol in mols:
        assert mol.HasProp(cache_key)

    # expect the same charges as the first time around
    cached_am1_charges = [nonbonded.compute_or_load_oe_charges(mol) for mol in mols]
    for fresh, cached in zip(fresh_am1_charges, cached_am1_charges):
        np.testing.assert_array_equal(fresh, cached)


def assert_permutation_equivariance(mol, fxn, perm):
    """fxn(mol[perm]) == fxn(mol)[perm]"""
    mol = deepcopy(mol)
    qs = fxn(mol)

    permuted_mol = Chem.RenumberAtoms(mol, list(int(idx) for idx in perm))
    qs_perm = fxn(permuted_mol)

    np.testing.assert_allclose(qs_perm, qs[perm])


@pytest.mark.parametrize("mol_idx", [0, 1, 2, 3, 4, 5])
def test_partial_charge_equivariance_on_freesolv(mol_idx):
    ff = Forcefield.load_default()

    seed = 2024
    rng = np.random.default_rng(seed)

    mol = fetch_freesolv()[mol_idx]
    perm = rng.permutation(mol.GetNumAtoms())

    assert_permutation_equivariance(mol, ff.q_handle.parameterize, perm)


def test_charging_compounds_with_non_zero_charge():
    patterns = [
        ["[#6a:1]:[#6a:2]", 0.0],
        [
            "[#7X3ar5,#7X3+1,#7X3+0$(*-[#6X3$(*=[#7X3+1])]),$([#7X3](-[#8X1-1])=[#8X1]),$([#7X3](=[#8X1])=[#8X1]):1]-[#8X1,#8X2:2]",
            0.2380991882939545,
        ],
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([x[1] for x in patterns])
    props = None
    am1h = nonbonded.AM1CCCHandler(smirks, params, props)

    positive_mol = Chem.AddHs(Chem.MolFromSmiles("c1cc[nH+]cc1"))
    AllChem.EmbedMolecule(positive_mol)

    negative_mol = Chem.AddHs(Chem.MolFromSmiles("[N+](=O)([O-])[O-]"))
    AllChem.EmbedMolecule(negative_mol)

    es_params = am1h.parameterize(positive_mol)
    np.testing.assert_almost_equal(np.sum(es_params / np.sqrt(ONE_4PI_EPS0)), 1.0, decimal=5)

    es_params = am1h.parameterize(negative_mol)
    np.testing.assert_almost_equal(np.sum(es_params) / np.sqrt(ONE_4PI_EPS0), -1.0, decimal=5)


def test_precomputed_charge_handler():
    with path_to_internal_file("timemachine.testsystems.water_exchange", "bb_centered_espaloma.sdf") as path_to_ligand:
        mol = utils.read_sdf(path_to_ligand)[0]

    pch = nonbonded.PrecomputedChargeHandler([], [], None)
    params = pch.parameterize(mol)
    for a_idx, p in enumerate(params):
        assert float(mol.GetAtomWithIdx(a_idx).GetProp("PartialCharge")) * np.sqrt(ONE_4PI_EPS0) == p

    # allclose used here to deal with roundoff
    np.testing.assert_allclose(
        params,
        np.array(
            [
                0.08300972,
                0.08096719,
                0.08595583,
                0.08161837,
                0.08247714,
                -0.00184935,
                0.12134614,
                0.09675746,
                0.00247987,
                -0.00978577,
                -0.13067351,
                -0.14841685,
                -0.05102999,
                0.1550052,
                -0.01422306,
                -0.08452472,
                -0.00732864,
                -0.00732864,
                -0.08452472,
                -0.2431052,
                -0.24310519,
                -0.09801473,
                -0.09801478,
                -0.03118374,
                -0.0311838,
                -0.14239897,
                -0.00184933,
                -0.14239886,
                -0.05103004,
                0.1550052,
                -0.01422302,
                0.00286067,
                0.00286068,
                0.4839953,
                -0.02375002,
                -0.02375001,
                0.48399526,
                0.00286067,
                0.0028607,
                -0.14841686,
                0.00247987,
                0.08247715,
                -0.13067354,
                -0.00978578,
                0.08161835,
                -0.03118377,
                -0.03118373,
                -0.09801482,
                -0.09801474,
                -0.2431052,
                -0.2431052,
                -0.08452472,
                -0.00732864,
                -0.00732862,
                -0.08452476,
                -0.01422311,
                0.15500528,
                -0.05103001,
                -0.14841682,
                -0.13067353,
                -0.00978577,
                0.08300972,
                0.08595583,
                0.08096719,
                -0.1423989,
                -0.00184936,
                -0.14239885,
                0.09797411,
                0.09797414,
                -0.00184938,
                0.08247717,
                0.08161836,
                0.08595586,
                0.08096719,
                0.08300973,
                0.12134615,
                0.09675753,
                0.00247987,
                -0.05103004,
                0.15500516,
                -0.01422306,
                0.08161838,
                -0.0097857,
                -0.13067359,
                0.08247717,
                0.00247984,
                -0.14841682,
                0.08300976,
                0.08595584,
                0.08096716,
            ],
            dtype=np.float32,
        )
        * np.sqrt(ONE_4PI_EPS0),
    )


def test_compute_or_load_bond_smirks_matches():
    """Loop over test ligands, asserting that
    * verify no cache key
    * returned indices are in bounds
    * returned bonds are present in the mol
    * verify a cache key
    * verify new values match initial matches"""
    # get some molecules
    match_cache_key = nonbonded.BOND_SMIRK_MATCH_CACHE

    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = utils.read_sdf(path_to_ligand)

    # get some bond smirks
    smirks_list = [smirks for (smirks, param) in AM1CCC_CHARGES["patterns"]]

    for mol in all_mols:
        assert not mol.HasProp(match_cache_key)

    fresh_matches = []
    for mol in all_mols:
        bond_idxs, type_idxs = nonbonded.compute_or_load_bond_smirks_matches(mol, smirks_list)
        fresh_matches.append((bond_idxs, type_idxs))
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
    for mol in all_mols:
        assert mol.HasProp(match_cache_key)
    cached_matches = [nonbonded.compute_or_load_bond_smirks_matches(mol, smirks_list) for mol in all_mols]
    for (fresh_bonds, fresh_types), (cached_bonds, cached_types) in zip(fresh_matches, cached_matches):
        np.testing.assert_array_equal(fresh_bonds, cached_bonds)
        np.testing.assert_array_equal(fresh_types, cached_types)


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
    assert np.all(adjoints[mask] == 0.0)


def test_symmetric_am1ccc():
    """Assert that (symmetric_bond_smarts, +1.0) has same behavior as (symmetric_bond_smarts, 0.0) on one test mol"""

    cyclohexane = "C1CCCCC1"
    mol = Chem.MolFromSmiles(cyclohexane)
    mol = Chem.AddHs(mol)

    smirks = ["[#6:1]~[#6:2]"]
    zeros = np.zeros(len(smirks))
    ones = np.ones(len(smirks))

    ref_charges = np.array(nonbonded.AM1CCCHandler.static_parameterize(zeros, smirks, mol))
    test_charges = np.array(nonbonded.AM1CCCHandler.static_parameterize(ones, smirks, mol))

    # at https://github.com/proteneer/timemachine/tree/fd14908113315ca07c8983e7ecd4dd92178d03a8
    # set(ref_charges) == {-1.8165082, -1.8158009, 0.90795946}
    # set(test_charges) == {-3.815801, -1.8158009, 0.18349183, 0.90795946}
    np.testing.assert_array_equal(test_charges, ref_charges)


def test_nn_handler():
    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = utils.read_sdf(path_to_ligand)

    mol = all_mols[0]
    seed = 2024
    feature_size = TEST_FEATURE_SIZE
    rng = np.random.default_rng(seed)

    set_nn_features(mol, feature_size=feature_size)

    # input has the features (atom0, atom1, bond)
    layer_sizes = [feature_size * 4, 8, 1]
    weight_sizes = list(zip(np.array(layer_sizes)[1:], np.array(layer_sizes[:-1])))
    full_params = [rng.random(weight_size) for weight_size in weight_sizes]

    flat_params, unflatten = flatten_util.ravel_pytree(full_params)
    params = [flat_params]
    enc_unflatten_str = [base64.b64encode(pickle.dumps(unflatten))]
    nn = nonbonded.NNHandler(enc_unflatten_str, params, None)
    charges = nn.static_parameterize(params, enc_unflatten_str, mol)
    assert np.sum(charges) < 1e-5

    def loss_fn(params):
        return jnp.sum(jnp.abs(nn.static_parameterize(params, enc_unflatten_str, mol)))

    grad_fn = jax.grad(loss_fn)
    print("loss", loss_fn(params))  # fast
    print("grad", grad_fn(params))  # a few seconds
    print("jit grad", jax.jit(grad_fn)(params))  # also a few seconds


def test_harmonic_bonds_complete():
    """On a test molecule containing [oxygen] ~ [halogen] bonds,
    assert that a ValueError is raised."""

    mol = Chem.MolFromSmiles("O(F)F")  # 0 smirks matches using current (2022-05-16) handler

    ff = Forcefield.load_default()
    parameterize = ff.hb_handle.parameterize

    with pytest.raises(ValueError) as e:
        _, _ = parameterize(mol)
    assert "missing bonds" in str(e)


@pytest.mark.parametrize("is_nn", [True, False])
@pytest.mark.parametrize(
    "protein_path_and_symmetries",
    [
        (
            "capped_hhh.pdb",
            [
                [3, 4, 5],
                [18, 19],
                [36, 37],
                [53, 54],
                [61, 62, 63],
            ],
        ),
        (
            "capped_phenyl_tyro.pdb",
            [
                [3, 4, 5],
                [19, 20],
                [12, 13],
                [21, 22],
                [14, 15],
                [23, 24],
                [40, 41],
                [32, 33],
                [42, 43],
                [34, 35],
                [44, 45],
                [50, 51, 52],
            ],
        ),
        (
            "capped_kkmi.pdb",
            [
                [3, 4, 5],
                [17, 18],
                [19, 20],
                [21, 22],
                [23, 24],
                [25, 26, 27],
                [39, 40],
                [41, 42],
                [43, 44],
                [45, 46],
                [47, 48, 49],
                [80, 81, 82],
                [78, 79],
                [83, 84, 85],
                [60, 61],
                [62, 63],
                [64, 65, 66],
            ],
        ),
    ],
)
def test_env_bcc_peptide_symmetries(protein_path_and_symmetries, is_nn, env_nn_args):
    """
    Test that we can compute BCCs to generate per atom charge offsets and that they can be differentiated
    """
    protein_path, expected_symmetries = protein_path_and_symmetries
    with path_to_internal_file("timemachine.testsystems.data", protein_path) as path_to_pdb:
        host_pdb = app.PDBFile(str(path_to_pdb))
        topology = host_pdb.topology

    rng = np.random.default_rng(seed=2024)
    if is_nn:
        smirks, params, props = env_nn_args
        pbcc = nonbonded.EnvironmentNNHandler(smirks, params, props, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, topology)
    else:
        smirks = [x[0] for x in AM1CCC_CHARGES["patterns"]]
        params = rng.random(len(smirks)) - 0.5
        pbcc = nonbonded.EnvironmentBCCHandler(smirks, params, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, topology)

    # raw charges are correct are in the order of atoms in the topology
    raw_charges = np.array(pbcc.parameterize(np.zeros_like(params)))
    bcc_charges = np.array(pbcc.parameterize(params))

    atoms = []
    bonds = []

    for atom in topology.atoms():
        atoms.append(atom.element.symbol)

    for src_atom, dst_atom in topology.bonds():
        bonds.append((src_atom.index, dst_atom.index))

    # mol = hutils.make_residue_mol('res', atoms, bonds)
    # from rdkit.Chem import Draw
    # with open("debug.svg", "w") as fh:
    #     svg = Draw.MolsToGridImage([mol], molsPerRow=4, useSVG=True)
    #     fh.write(svg)

    # symmetry classes generated by eye
    # symmetries = [[3, 4, 5], [18, 19], [36, 37], [53, 54], [61, 62, 63]]

    for group in expected_symmetries:
        first = group[0]
        for other in group[1:]:
            np.testing.assert_almost_equal(bcc_charges[other], bcc_charges[first])
            np.testing.assert_almost_equal(raw_charges[other], raw_charges[first])


@pytest.mark.nightly(reason="Slow")
@pytest.mark.parametrize("is_nn", [True, False])
@pytest.mark.parametrize("protein_path", ["5dfr_solv_equil.pdb", "hif2a_nowater_min.pdb"])
def test_environment_bcc_full_protein(protein_path, is_nn, env_nn_args):
    """
    Test that we can compute BCCs to generate per atom charge offsets and that they can be differentiated
    """
    with path_to_internal_file("timemachine.testsystems.data", protein_path) as path_to_pdb:
        host_pdb = app.PDBFile(str(path_to_pdb))
        host_config = builders.build_protein_system(host_pdb, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF)

    if is_nn:
        patterns, params, props = env_nn_args
        pbcc = nonbonded.EnvironmentNNHandler(
            patterns, params, props, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, host_config.omm_topology
        )
    else:
        patterns = [smirks for (smirks, param) in AM1CCC_CHARGES["patterns"]]
        params = np.random.rand(len(patterns)) - 0.5
        pbcc = nonbonded.EnvironmentBCCHandler(
            patterns, params, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, host_config.omm_topology
        )

    # test that we can mechanically parameterize everything
    final_q_params = pbcc.parameterize(params)

    def loss_fn(bcc_params):
        res = pbcc.parameterize(bcc_params)
        return jnp.sum(res * res)

    grad_fn = jax.grad(loss_fn)

    print(loss_fn(params))  # fast
    print(grad_fn(params))  # a few seconds
    print(jax.jit(grad_fn)(params))  # also a few seconds

    # test that the partial handler gives the same results
    ff = Forcefield.load_default()
    if is_nn:
        partial_cc = nonbonded.EnvironmentNNPartialHandler(patterns, params, props)
    else:
        partial_cc = nonbonded.EnvironmentBCCPartialHandler(patterns, params, None)

    pbcc2 = partial_cc.get_env_handle(host_config.omm_topology, ff)
    np.testing.assert_array_equal(pbcc.parameterize(params), pbcc2.parameterize(params))

    def loss_fn2(bcc_params):
        res = pbcc.parameterize(bcc_params)
        return jnp.sum(res * res)

    grad_fn2 = jax.grad(loss_fn2)

    assert loss_fn(params) == loss_fn2(params)
    np.testing.assert_array_equal(grad_fn(params), grad_fn2(params))

    # now check for correctness on the full protein set
    # if the initial charges are the same, then the final charges should
    # also be the same by symmetry
    atom_idx_to_res_name = {}
    cur_atom = 0
    for i in range(len(pbcc.template_for_residue)):
        tfr = pbcc.template_for_residue[i]
        n_atoms = len(tfr.atoms)
        for j in range(n_atoms):
            atom_idx_to_res_name[cur_atom + j] = f"{tfr.name}"
        cur_atom += n_atoms

    # charges that are the same and belong to the same residue should
    # be the same after applying the corrections
    init_q_params = pbcc.initial_charges
    sym_charge_idxs = defaultdict(list)
    for i, q in enumerate(init_q_params):
        sym_charge_idxs[(q, atom_idx_to_res_name[i])].append(i)

    for k, group in sym_charge_idxs.items():
        first = group[0]
        for other in group[1:]:
            np.testing.assert_almost_equal(final_q_params[other], final_q_params[first])
