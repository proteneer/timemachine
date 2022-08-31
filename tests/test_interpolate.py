from jax.config import config

config.update("jax_enable_x64", True)

from importlib import resources

import numpy as np
import pytest
from rdkit import Chem

from timemachine.fe import atom_mapping, interpolate, single_topology_v3
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield


def test_align_harmonic_bond():
    """
    Test that we can align idxs and parameters correctly for harmonic bonds.
    We expect that decoupled terms have their force constants set to zero,
    while maintaining the same equilibrium bond lengths.
    """
    a, b, c, d, e, f, g, h, i, j = np.random.rand(10)

    src_idxs = [(4, 9), (3, 4)]
    src_params = [(a, b), (c, d)]

    dst_idxs = [(3, 4), (5, 9), (2, 3)]
    dst_params = [(e, f), (g, h), (i, j)]

    test_set = interpolate.align_harmonic_bond_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)

    ref_set = {
        ((4, 9), (a, b), (0, b)),
        ((3, 4), (c, d), (e, f)),
        ((5, 9), (0, h), (g, h)),
        ((2, 3), (0, j), (i, j)),
    }

    assert test_set == ref_set

    # test that if there are repeats we throw an assertion
    with pytest.raises(interpolate.DuplicateAlignmentKeysError):
        src_idxs = [(4, 9), (4, 9)]
        interpolate.align_harmonic_bond_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)

    # test that non-canonical idxs should assert
    with pytest.raises(AssertionError):
        src_idxs = [(9, 4), (3, 4)]
        interpolate.align_harmonic_bond_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)

    # test non-canonical idxs throw an assertion


def test_align_harmonic_angle():
    """
    Test that we can align idxs and parameters correctly for harmonic angles.
    We expect that decoupled terms have their force constants turned set to zero,
    while maintaining the same equilibrium angles.
    """
    a, b, c, d, e, f, g, h, i, j = np.random.rand(10)

    # tbd: what do we do if there are repeats?
    # merge repeats into a single term first?
    src_idxs = [(4, 9, 5), (3, 4, 6)]
    src_params = [(a, b), (c, d)]

    dst_idxs = [(3, 4, 6), (4, 5, 9), (2, 3, 4)]
    dst_params = [(e, f), (g, h), (i, j)]

    test_set = interpolate.align_harmonic_bond_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)

    ref_set = {
        ((4, 9, 5), (a, b), (0, b)),
        ((3, 4, 6), (c, d), (e, f)),
        ((4, 5, 9), (0, h), (g, h)),
        ((2, 3, 4), (0, j), (i, j)),
    }

    assert test_set == ref_set


def test_align_torsion():
    """
    Test that we can align idxs and parameters correctly for periodic torsions.
    Periodic torsions differ from bonds and angles in that their uniqueness is
    also determined by the "period", which is currently encoded as one of the parameters.

    We expect that decoupled terms have their force constants turned set to zero,
    while maintaining the same equilibrium angles for the *same* period.
    """
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p = np.random.rand(16)

    # tbd: what do we do if there are repeats?
    # merge repeats into a single term first?
    src_idxs = [(2, 3, 9, 4), (2, 1, 4, 3), (0, 1, 4, 2), (0, 1, 4, 2)]
    src_params = [(a, b, 2), (c, d, 1), (e, f, 3), (g, h, 1)]

    dst_idxs = [(2, 3, 9, 4), (2, 3, 9, 4), (0, 1, 4, 2), (3, 0, 2, 6)]
    dst_params = [(i, j, 2), (k, l, 1), (m, n, 3), (o, p, 4)]

    test_set = interpolate.align_torsion_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)

    ref_set = {
        ((2, 3, 9, 4), (a, b, 2), (i, j, 2)),
        ((2, 1, 4, 3), (c, d, 1), (0, d, 1)),
        ((0, 1, 4, 2), (e, f, 3), (m, n, 3)),
        ((0, 1, 4, 2), (g, h, 1), (0, h, 1)),
        ((2, 3, 9, 4), (0, l, 1), (k, l, 1)),
        ((3, 0, 2, 6), (0, p, 4), (o, p, 4)),
    }

    assert test_set == ref_set


def test_align_nonbonded():
    """
    Test that we can align idxs and parameters for nonbonded forces. For decoupled
    nonbonded interactions, we should set q_ij, s_ij, and e_ij all to zero.
    """
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r = np.random.rand(18)

    src_idxs = [(0, 3), (0, 2), (2, 3), (2, 4)]
    src_params = [(a, b, c), (d, e, f), (g, h, i), (j, k, l)]
    dst_idxs = [(0, 2), (4, 5)]
    dst_params = [(m, n, o), (p, q, r)]

    test_set = interpolate.align_nonbonded_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)

    ref_set = {
        ((0, 3), (a, b, c), (0, 0, 0)),
        ((0, 2), (d, e, f), (m, n, o)),
        ((2, 3), (g, h, i), (0, 0, 0)),
        ((2, 4), (j, k, l), (0, 0, 0)),
        ((4, 5), (0, 0, 0), (p, q, r)),
    }

    assert test_set == ref_set


def test_align_chiral_atoms():
    """
    Test that we can align idxs and parameters for chiral atom restraints. We should
    expect that force constants are set to zero.
    """
    a, b, c, d, e = np.random.rand(5)

    src_idxs = [(0, 3, 4, 5), (0, 4, 3, 5), (4, 3, 5, 6)]
    src_params = [a, b, c]
    dst_idxs = [(0, 4, 3, 5), (4, 3, 5, 7)]
    dst_params = [d, e]

    test_set = interpolate.align_chiral_atom_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)

    ref_set = {
        ((0, 3, 4, 5), a, 0),
        ((0, 4, 3, 5), b, d),
        ((4, 3, 5, 6), c, 0),
        ((4, 3, 5, 7), 0, e),
    }

    assert test_set == ref_set


def test_align_chiral_bonds():
    """
    Similar to the chiral_atoms test, except that we check to see if deduplication
    is using the sign information as part of the canonicalization routine.
    """

    a, b, c, d, e, f, g = np.random.rand(7)

    src_idxs = [(0, 3, 4, 5), (0, 3, 4, 5), (0, 4, 3, 5), (4, 3, 5, 6)]
    src_params = [a, b, c, d]
    src_signs = [1, -1, 1, 1]
    dst_idxs = [(4, 3, 5, 7), (0, 3, 4, 5), (4, 3, 5, 6)]
    dst_params = [e, f, g]
    dst_signs = [1, -1, -1]

    test_set = interpolate.align_chiral_bond_idxs_and_params(
        src_idxs, src_params, src_signs, dst_idxs, dst_params, dst_signs
    )

    ref_set = {
        ((0, 3, 4, 5), 1, a, 0),
        ((0, 3, 4, 5), -1, b, f),
        ((0, 4, 3, 5), 1, c, 0),
        ((4, 3, 5, 6), 1, d, 0),
        ((4, 3, 5, 6), -1, 0, g),
        ((4, 3, 5, 7), 1, 0, e),
    }

    assert test_set == ref_set

    src_signs[1] = 1
    assert len(set(list(zip(src_idxs, src_signs))[:2])) == 1  # first 2 alignment keys are duplicates
    with pytest.raises(interpolate.DuplicateAlignmentKeysError):
        interpolate.align_chiral_bond_idxs_and_params(src_idxs, src_params, src_signs, dst_idxs, dst_params, dst_signs)


def test_intermediate_states(num_pairs_to_setup=10):
    """
    Test that intermediate states evaluated at lambda=0 and lambda=1 have the same energy
    as the src and dst end-states.
    """

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)
        mols = [m for m in suppl]

    pairs = [(mol_a, mol_b) for mol_a in mols for mol_b in mols]
    np.random.seed(2023)
    np.random.shuffle(pairs)
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    for mol_a, mol_b in pairs[:num_pairs_to_setup]:

        print("Checking", mol_a.GetProp("_Name"), "->", mol_b.GetProp("_Name"))
        mcs_threshold = 0.75  # distance threshold, in nanometers
        res = atom_mapping.mcs_conformer_aware(mol_a, mol_b, mcs_threshold)
        query = Chem.MolFromSmarts(res.smartsString)
        core_pairs = atom_mapping.get_core_by_mcs(mol_a, mol_b, query, mcs_threshold)

        top = single_topology_v3.SingleTopologyV3(mol_a, mol_b, core_pairs, ff)
        x0 = top.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))

        # test end-states and check to see if the forces are the same
        system_lambda_0 = top.setup_intermediate_state(0)

        U_ref = top.src_system.get_U_fn()
        U_test = system_lambda_0.get_U_fn()

        # these are not guaranteed to be bitwise identical
        # since permuting the order of idxs will affect
        # the order of operations
        # suggestion by jfass: test random coords
        xs = [x0]
        for _ in range(10):
            xs.append(x0 + 0.01 * np.random.randn(*x0.shape))
        for x in xs:
            np.testing.assert_almost_equal(U_ref(x), U_test(x))

        system_lambda_1 = top.setup_intermediate_state(1)
        U_ref = top.dst_system.get_U_fn()
        U_test = system_lambda_1.get_U_fn()

        xs = [x0]
        for _ in range(10):
            xs.append(x0 + 0.01 * np.random.randn(*x0.shape))
        for x in xs:
            np.testing.assert_almost_equal(U_ref(x), U_test(x))
