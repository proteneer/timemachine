from jax.config import config; config.update("jax_enable_x64", True)

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from ff.handlers import nonbonded, bonded
from ff.handlers.deserialize import deserialize


def test_harmonic_bond():

    patterns = [
        ['[#6X4:1]-[#6X4:2]', 0.1, 0.2],
        ['[#6X4:1]-[#6X3:2]', 99., 99.],
        ['[#6X4:1]-[#6X3:2]=[#8X1+0]', 99., 99.],
        ['[#6X3:1]-[#6X3:2]', 99., 99.],
        ['[#6X3:1]:[#6X3:2]', 99., 99.],
        ['[#6X3:1]=[#6X3:2]', 99., 99.],
        ['[#6:1]-[#7:2]',0.1, 0.2],
        ['[#6X3:1]-[#7X3:2]', 99., 99.],
        ['[#6X4:1]-[#7X3:2]-[#6X3]=[#8X1+0]', 99., 99.],
        ['[#6X3:1](=[#8X1+0])-[#7X3:2]', 99., 99.],
        ['[#6X3:1]-[#7X2:2]', 99., 99.],
        ['[#6X3:1]:[#7X2,#7X3+1:2]', 99., 99.],
        ['[#6X3:1]=[#7X2,#7X3+1:2]', 99., 99.],
        ['[#6:1]-[#8:2]', 99., 99.],
        ['[#6X3:1]-[#8X1-1:2]', 99., 99.],
        ['[#6X4:1]-[#8X2H0:2]', 0.3, 0.4],
        ['[#6X3:1]-[#8X2:2]', 99., 99.],
        ['[#6X3:1]-[#8X2H1:2]', 99., 99.],
        ['[#6X3a:1]-[#8X2H0:2]', 99., 99.],
        ['[#6X3:1](=[#8X1])-[#8X2H0:2]', 99., 99.],
        ['[#6:1]=[#8X1+0,#8X2+1:2]', 99., 99.],
        ['[#6X3:1](~[#8X1])~[#8X1:2]', 99., 99.],
        ['[#6X3:1]~[#8X2+1:2]~[#6X3]', 99., 99.],
        ['[#6X2:1]-[#6:2]', 99., 99.],
        ['[#6X2:1]-[#6X4:2]', 99., 99.],
        ['[#6X2:1]=[#6X3:2]', 99., 99.],
        ['[#6:1]#[#7:2]', 99., 99.],
        ['[#6X2:1]#[#6X2:2]', 99., 99.],
        ['[#6X2:1]-[#8X2:2]', 99., 99.],
        ['[#6X2:1]-[#7:2]', 99., 99.],
        ['[#6X2:1]=[#7:2]', 99., 99.],
        ['[#16:1]=[#6:2]', 99., 99.],
        ['[#6X2:1]=[#16:2]', 99., 99.],
        ['[#7:1]-[#7:2]', 99., 99.],
        ['[#7X3:1]-[#7X2:2]', 99., 99.],
        ['[#7X2:1]-[#7X2:2]', 99., 99.],
        ['[#7:1]:[#7:2]', 99., 99.],
        ['[#7:1]=[#7:2]', 99., 99.],
        ['[#7+1:1]=[#7-1:2]', 99., 99.],
        ['[#7:1]#[#7:2]', 99., 99.],
        ['[#7:1]-[#8X2:2]', 99., 99.],
        ['[#7:1]~[#8X1:2]', 99., 99.],
        ['[#8X2:1]-[#8X2:2]', 99., 99.],
        ['[#16:1]-[#6:2]', 99., 99.],
        ['[#16:1]-[#1:2]', 99., 99.],
        ['[#16:1]-[#16:2]', 99., 99.],
        ['[#16:1]-[#9:2]', 99., 99.],
        ['[#16:1]-[#17:2]', 99., 99.],
        ['[#16:1]-[#35:2]', 99., 99.],
        ['[#16:1]-[#53:2]', 99., 99.],
        ['[#16X2,#16X1-1,#16X3+1:1]-[#6X4:2]', 99., 99.],
        ['[#16X2,#16X1-1,#16X3+1:1]-[#6X3:2]', 99., 99.],
        ['[#16X2:1]-[#7:2]', 99., 99.],
        ['[#16X2:1]-[#8X2:2]', 99., 99.],
        ['[#16X2:1]=[#8X1,#7X2:2]', 99., 99.],
        ['[#16X4,#16X3!+1:1]-[#6:2]', 99., 99.],
        ['[#16X4,#16X3:1]~[#7:2]', 99., 99.],
        ['[#16X4,#16X3:1]-[#8X2:2]', 99., 99.],
        ['[#16X4,#16X3:1]~[#8X1:2]', 99., 99.],
        ['[#15:1]-[#1:2]', 99., 99.],
        ['[#15:1]~[#6:2]', 99., 99.],
        ['[#15:1]-[#7:2]', 99., 99.],
        ['[#15:1]=[#7:2]', 99., 99.],
        ['[#15:1]~[#8X2:2]', 99., 99.],
        ['[#15:1]~[#8X1:2]', 99., 99.],
        ['[#16:1]-[#15:2]', 99., 99.],
        ['[#15:1]=[#16X1:2]', 99., 99.],
        ['[#6:1]-[#9:2]', 99., 99.],
        ['[#6X4:1]-[#9:2]', 0.6, 0.7],
        ['[#6:1]-[#17:2]', 99., 99.],
        ['[#6X4:1]-[#17:2]', 99., 99.],
        ['[#6:1]-[#35:2]', 99., 99.],
        ['[#6X4:1]-[#35:2]', 99., 99.],
        ['[#6:1]-[#53:2]', 99., 99.],
        ['[#6X4:1]-[#53:2]', 99., 99.],
        ['[#7:1]-[#9:2]', 99., 99.],
        ['[#7:1]-[#17:2]', 99., 99.],
        ['[#7:1]-[#35:2]', 99., 99.],
        ['[#7:1]-[#53:2]', 99., 99.],
        ['[#15:1]-[#9:2]', 99., 99.],
        ['[#15:1]-[#17:2]', 99., 99.],
        ['[#15:1]-[#35:2]', 99., 99.],
        ['[#15:1]-[#53:2]', 99., 99.],
        ['[#6X4:1]-[#1:2]', 99., 99.],
        ['[#6X3:1]-[#1:2]', 99., 99.],
        ['[#6X2:1]-[#1:2]', 99., 99.],
        ['[#7:1]-[#1:2]', 99., 99.],
        ['[#8:1]-[#1:2]', 99., 99.1]
    ]


    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2]] for x in patterns])
    props = None
    hbh = bonded.HarmonicBondHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("C1CNCOC1F")

    bond_idxs, (bond_params, bond_vjp_fn) = hbh.parameterize(mol)

    assert bond_idxs.shape == (mol.GetNumBonds(), 2)
    assert bond_params.shape == (mol.GetNumBonds(), 2)


    bonded_param_adjoints = np.random.randn(*bond_params.shape)

    # test that we can use the adjoints
    ff_adjoints = bond_vjp_fn(bonded_param_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(bond_params > 90)
    assert np.all(ff_adjoints[mask] == 0.0) == True

def test_proper_torsion():

    # proper torsions have a variadic number of terms

    patterns = [
        ['[*:1]-[#6X3:2]=[#6X3:3]-[*:4]', [[99., 99., 99.]]],
        ['[*:1]-[#6X3:2]=[#6X3:3]-[#35:4]', [[99., 99., 99.]]],
        ['[#9:1]-[#6X3:2]=[#6X3:3]-[#35:4]', [[1., 2., 3.], [4., 5., 6.]]],
        ['[#35:1]-[#6X3:2]=[#6X3:3]-[#35:4]', [[7., 8., 9.], [1., 3., 5.], [4., 4., 4.]]],
        ['[#9:1]-[#6X3:2]=[#6X3:3]-[#9:4]', [[7., 8., 9.]]],
    ]

    smirks = [x[0] for x in patterns]
    params = [x[1] for x in patterns]
    props = None

    hbh = bonded.ProperTorsionHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("FC(Br)=C(Br)F")

    torsion_idxs, (torsion_params, torsion_vjp_fn) = hbh.parameterize(mol)

    assert torsion_idxs.shape == (8, 4)
    assert torsion_params.shape == (8, 3)

    torsion_param_adjoints = np.random.randn(*torsion_params.shape)

    ff_adjoints = torsion_vjp_fn(torsion_param_adjoints)[0]

    mask = np.argwhere(torsion_params > 90)
    assert np.all(ff_adjoints[mask] == 0.0) == True


def test_improper_torsion():

    patterns = [
        ['[*:1]~[#6X3:2](~[*:3])~[*:4]', 1.5341333333333333, 3.141592653589793, 2.0],
        ['[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]', 99., 99., 99.],
        ['[*:1]~[#7X3$(*~[#15,#16](!-[*])):2](~[*:3])~[*:4]', 99., 99., 99.],
        ['[*:1]~[#7X3$(*~[#6X3]):2](~[*:3])~[*:4]', 1.3946666666666667, 3.141592653589793, 2.0],
        ['[*:1]~[#7X3$(*~[#7X2]):2](~[*:3])~[*:4]', 99., 99., 99.],
        ['[*:1]~[#7X3$(*@1-[*]=,:[*][*]=,:[*]@1):2](~[*:3])~[*:4]', 99., 99., 99.],
        ['[*:1]~[#6X3:2](=[#7X2,#7X3+1:3])~[#7:4]', 99., 99., 99.]
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2], x[3]] for x in patterns])
    props = None
    imp_handler = bonded.ImproperTorsionHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("CNC(C)=O") # peptide
    mol = Chem.AddHs(mol)

    torsion_idxs, (params, vjp_fn) = imp_handler.parameterize(mol)

    assert torsion_idxs.shape[0] == 6 # we expect two sets of impropers, each with 3 components.
    assert torsion_idxs.shape[1] == 4

    assert params.shape[0] == 6
    assert params.shape[1] == 3

    param_adjoints = np.random.randn(*params.shape)

    # # test that we can use the adjoints
    ff_adjoints = vjp_fn(param_adjoints)[0]

    # # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(params > 90)
    assert np.all(ff_adjoints[mask] == 0.0) == True

def test_exclusions():

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

def test_am1_bcc():
    # currently takes no parameters
    smirks = []
    params = []
    props = None

    am1h = nonbonded.AM1BCCHandler(smirks, params, props)
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CNCOC1F"))
    AllChem.EmbedMolecule(mol)
    charges, vjp_fn = am1h.parameterize(mol)

    assert len(charges) == mol.GetNumAtoms()

    charges_adjoints = np.random.randn(*charges.shape)

    assert vjp_fn(charges_adjoints) == None

def test_am1_ccc():
    
    patterns = [
        ['[#6X4:1]-[#1:2]', 0.46323257920556493],
        ['[#6X3$(*=[#8,#16]):1]-[#6a:2]', 0.24281402370571598],
        ['[#6X3$(*=[#8,#16]):1]-[#8X1,#8X2:2]', 1.0620166764992722],
        ['[#6X3$(*=[#8,#16]):1]=[#8X1$(*=[#6X3]-[#8X2]):2]', 2.227759732057297],
        ['[#6X3$(*=[#8,#16]):1]=[#8X1,#8X2:2]', 2.8182928673804217],
        ['[#6a:1]-[#8X1,#8X2:2]', 0.5315976926761063],
        ['[#6a:1]-[#1:2]', 0.0],
        ['[#6a:1]:[#6a:2]', 0.0],
        ['[#6a:1]:[#6a:2]', 0.0],
        ['[#8X1,#8X2:1]-[#1:2]', -2.3692047944101415],
        ['[#16:1]-[#8:2]', 99.]
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([x[1] for x in patterns])
    props = None

    am1h = nonbonded.AM1CCCHandler(smirks, params, props)
    mol = Chem.AddHs(Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O"))
    AllChem.EmbedMolecule(mol)
    es_params, es_vjp_fn = am1h.parameterize(mol)

    # TBD update with AM1 Symmetrize=True
    ligand_params = np.array([-6.10948,  7.08286, -1.13097, -4.3096,  2.03822, -1.72492,
        -0.98493, -1.75663, -0.70852, -1.65444,  7.76463, -6.49881,
        -7.10585,  0.93707,  0.93707,  0.93707,  1.79376,  1.6654,
        1.69322,  1.90986,  5.22498])
    
    np.testing.assert_almost_equal(es_params, ligand_params, decimal=5)
 
    es_params_adjoints = np.random.randn(*es_params.shape)

    adjoints = es_vjp_fn(es_params_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(params > 90)
    assert np.all(adjoints[mask] == 0.0) == True

    import time
    start = time.time()
    es_params_from_cache, _ = am1h.parameterize(mol)
    end = time.time()

    # second pass should be very fast
    assert end-start < 1.0

    # should be *exactly* identical since we're loading from cache
    np.testing.assert_equal(es_params_from_cache, es_params)


    
def test_simple_charge_handler():

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
    props = None

    sch = nonbonded.SimpleChargeHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("C1CNCOC1F")

    es_params, es_vjp_fn = sch.parameterize(mol)

    ligand_params = np.array([
        0.1, # C
        0.1, # C
        0.3, # N
        0.1, # C
        0.5, # O
        0.1, # C
        1.0  # F
    ])

    np.testing.assert_almost_equal(es_params, ligand_params)

    es_params_adjoints = np.random.randn(*es_params.shape)

    # test that we can use the adjoints
    adjoints = es_vjp_fn(es_params_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(params > 90)
    assert np.all(adjoints[mask] == 0.0) == True

def test_gbsa_handler():

    patterns = [
       ['[*:1]', 99., 99.],
       ['[#1:1]', 99., 99.],
       ['[#1:1]~[#7]', 99., 99.],
       ['[#6:1]', 0.1, 0.2],
       ['[#7:1]', 0.3, 0.4],
       ['[#8:1]', 0.5, 0.6],
       ['[#9:1]', 0.7, 0.8],
       ['[#14:1]', 99., 99.],
       ['[#15:1]', 99., 99.],
       ['[#16:1]', 99., 99.],
       ['[#17:1]', 99., 99.]
    ]

    smirks = [x[0] for x in patterns]
    params = np.array([[x[1], x[2]] for x in patterns])
    props = {'foo': 'bar'}
    gbh = nonbonded.GBSAHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("C1CNCOC1F")

    gb_params, gb_vjp_fn = gbh.parameterize(mol)

    ligand_params = np.array([
        [0.1, 0.2], # C
        [0.1, 0.2], # C
        [0.3, 0.4], # N
        [0.1, 0.2], # C
        [0.5, 0.6], # O
        [0.1, 0.2], # C
        [0.7, 0.8]  # F
    ])

    np.testing.assert_almost_equal(gb_params, ligand_params)

    gb_params_adjoints = np.random.randn(*gb_params.shape)

    # test that we can use the adjoints
    adjoints = gb_vjp_fn(gb_params_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(params > 90)
    assert np.all(adjoints[mask] == 0.0) == True


def test_am1_differences():

    ff_raw = open("ff/params/smirnoff_1_1_0_ccc.py").read()
    ff_handlers = deserialize(ff_raw)
    for ccc in ff_handlers:
        if isinstance(ccc, nonbonded.AM1CCCHandler):
            break

    suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
    am1 = nonbonded.AM1Handler([], [], None)
    bcc = nonbonded.AM1BCCHandler([], [], None)

    for mol in suppl:

        am1_params, vjp_fn = am1.parameterize(mol)
        ccc_params, vjp_fn = ccc.parameterize(mol)
        bcc_params, vjp_fn = bcc.parameterize(mol)

        if np.sum(np.abs(ccc_params - bcc_params)) > 0.1:
        
            print(mol.GetProp("_Name"), Chem.MolToSmiles(mol))
            print("  AM1    CCC    BCC  S ?")
            for atom_idx, atom in enumerate(mol.GetAtoms()):
                a = am1_params[atom_idx]
                b = bcc_params[atom_idx]
                c = ccc_params[atom_idx]
                print("{:6.2f}".format(a), "{:6.2f}".format(c), "{:6.2f}".format(b), atom.GetSymbol(), end="")
                if np.abs(b-c) > 0.1:
                    print(" *")
                else:
                    print(" ")

            assert 0


def test_lennard_jones_handler():

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
    props = None

    ljh = nonbonded.LennardJonesHandler(smirks, params, props)

    mol = Chem.MolFromSmiles("C1CNCOC1F")

    lj_params, lj_vjp_fn = ljh.parameterize(mol)


    ligand_params = np.array([
        [0.1, 0.2], # C
        [0.1, 0.2], # C
        [0.3, 0.4], # N
        [0.1, 0.2], # C
        [0.5, 0.6], # O
        [0.1, 0.2], # C
        [1.0, 1.1]  # F
    ])

    np.testing.assert_almost_equal(lj_params, ligand_params)

    lj_params_adjoints = np.random.randn(*lj_params.shape)

    # test that we can use the adjoints
    adjoints = lj_vjp_fn(lj_params_adjoints)[0]

    # if a parameter is > 99 then its adjoint should be zero (converse isn't necessarily true since)
    mask = np.argwhere(params > 90)
    assert np.all(adjoints[mask] == 0.0) == True
