from importlib import resources

import numpy as np
import pytest
from rdkit import Chem

from timemachine.constants import (
    DEFAULT_ATOM_MAPPING_KWARGS,
    DEFAULT_BOND_IS_PRESENT_K,
    DEFAULT_CHIRAL_ATOM_RESTRAINT_K,
)
from timemachine.fe import atom_mapping, interpolate, single_topology
from timemachine.fe.interpolate import pad
from timemachine.fe.single_topology import (
    DUMMY_A_CHIRAL_ATOM_CONVERTING_OFF_MIN_MAX,
    DUMMY_B_CHIRAL_ATOM_CONVERTING_ON_MIN_MAX,
    MissingBondsInChiralVolumeException,
    SingleTopology,
    TorsionsDefinedOverLinearAngleException,
    assert_bonds_defined_for_chiral_volumes,
    assert_torsions_defined_over_non_linear_angles,
)
from timemachine.fe.utils import get_romol_conf, read_sdf, read_sdf_mols_by_name
from timemachine.ff import Forcefield

pytestmark = [pytest.mark.nocuda]


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


def test_align_proper():
    """
    Test that we can align idxs and parameters correctly for proper torsions.
    Proper torsions differ from bonds and angles in that their uniqueness is
    also determined by the "period", which is currently encoded as one of the parameters.

    We expect that decoupled terms have their force constants turned set to zero,
    while maintaining the same equilibrium angles for the *same* period.
    """
    a, b, c, d, e, f, g, h, i, j, k, l, m, n = np.random.rand(14)

    # tbd: what do we do if there are repeats?
    # merge repeats into a single term first?
    src_idxs = [(2, 3, 9, 4), (2, 1, 4, 3), (0, 1, 4, 2), (0, 1, 4, 2)]
    src_params = [(a, b, 2), (c, d, 1), (e, f, 3), (g, h, 1)]

    dst_idxs = [(2, 3, 9, 4), (2, 3, 9, 4), (0, 1, 4, 2), (3, 0, 2, 6)]
    dst_params = [(i, b, 2), (j, k, 1), (l, f, 3), (m, n, 4)]

    test_set = interpolate.align_proper_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)

    ref_set = {
        ((2, 3, 9, 4), (a, b, 2), (i, b, 2)),
        ((2, 1, 4, 3), (c, d, 1), (0, d, 1)),
        ((0, 1, 4, 2), (e, f, 3), (l, f, 3)),
        ((0, 1, 4, 2), (g, h, 1), (0, h, 1)),
        ((2, 3, 9, 4), (0, k, 1), (j, k, 1)),
        ((3, 0, 2, 6), (0, n, 4), (m, n, 4)),
    }

    assert test_set == ref_set


def test_align_improper():
    """
    Test that we can align idxs and parameters correctly for improper torsions.

    Currently, improper torsions all have period == 2 and phase == pi
    """
    a, b, c, d, e, f, g, h, i, j, k, l, m, n = np.random.rand(14)

    src_idxs = [(0, 1, 2, 3), (2, 1, 3, 0), (3, 1, 0, 2)]
    src_params = [(a, b, 2), (c, d, 1), (e, f, 3)]
    dst_idxs = [(2, 1, 3, 0), (3, 1, 0, 2), (0, 1, 2, 3)]
    dst_params = [(i, b, 2), (j, k, 1), (l, f, 3)]

    test_set = interpolate.align_improper_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)
    ref_set = {
        ((0, 1, 2, 3), (a, b, 2), (l, f, 3)),
        ((2, 1, 3, 0), (c, d, 1), (i, b, 2)),
        ((3, 1, 0, 2), (e, f, 3), (j, k, 1)),
    }

    assert test_set == ref_set

    src_idxs = [(0, 1, 2, 3)]
    src_params = [(a, b, 2)]
    dst_idxs = [(0, 1, 3, 2)]
    dst_params = [(i, c, 2)]

    test_set = interpolate.align_improper_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)
    ref_set = {
        ((0, 1, 2, 3), (a, b, 2), (0, b, 2)),
        ((0, 1, 3, 2), (0, c, 2), (i, c, 2)),
    }

    assert test_set == ref_set


def test_align_nonbonded():
    """
    Test that we can align idxs and parameters for nonbonded forces. For decoupled
    nonbonded interactions, we should set q_ij, s_ij, and e_ij all to zero.
    """

    rng = np.random.default_rng(2022)

    def p():
        return tuple(rng.random(4))

    src_idxs = [(0, 3), (0, 2), (2, 3), (2, 4)]
    dst_idxs = [(0, 2), (4, 5)]

    src = {idxs: p() for idxs in src_idxs}
    dst = {idxs: p() for idxs in dst_idxs}

    test_set = interpolate.align_nonbonded_idxs_and_params(src_idxs, src.values(), dst_idxs, dst.values())

    zeros = (0, 0, 0, 0)
    ref_set = {
        ((0, 3), src[(0, 3)], zeros),
        ((0, 2), src[(0, 2)], dst[(0, 2)]),
        ((2, 3), src[(2, 3)], zeros),
        ((2, 4), src[(2, 4)], zeros),
        ((4, 5), zeros, dst[(4, 5)]),
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

    src_idxs = [np.array([0, 3, 4, 5]), np.array([0, 3, 4, 5]), np.array([0, 4, 3, 5]), np.array([4, 3, 5, 6])]
    src_params = [a, b, c, d]
    src_signs = [1, -1, 1, 1]
    dst_idxs = [np.array([4, 3, 5, 7]), np.array([0, 3, 4, 5]), np.array([4, 3, 5, 6])]
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
    assert (
        len(set(list(zip([tuple(x) for x in src_idxs], src_signs))[:2])) == 1
    )  # first 2 alignment keys are duplicates
    with pytest.raises(interpolate.DuplicateAlignmentKeysError):
        interpolate.align_chiral_bond_idxs_and_params(src_idxs, src_params, src_signs, dst_idxs, dst_params, dst_signs)


def test_intermediate_states(num_pairs_to_setup=10):
    """
    Test that intermediate states evaluated at lambda=0 and lambda=1 have the same energy
    as the src and dst end-states.
    """

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)

    pairs = [(mol_a, mol_b) for mol_a in mols for mol_b in mols]
    np.random.seed(2023)
    np.random.shuffle(pairs)
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    for mol_a, mol_b in pairs[:num_pairs_to_setup]:
        print("Checking", mol_a.GetProp("_Name"), "->", mol_b.GetProp("_Name"))
        all_cores = atom_mapping.get_cores(
            mol_a,
            mol_b,
            **DEFAULT_ATOM_MAPPING_KWARGS,
        )

        core = all_cores[0]

        top = single_topology.SingleTopology(mol_a, mol_b, core, ff)
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


def test_duplicate_idxs_period_pairs():
    """Check that parameter interpolation is able to handle torsion terms with duplicate ((i, j, k, l), period) pairs.
    E.g. if we only align on idxs and period, this will result in a DuplicateAlignmentKeysError."""

    # S-C(=O)-C-N
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02222413343D

 10  9  0  0  0  0            999 V2000
   -0.1061    1.3412   -0.6918 S   0  0  0  0  0  0  0  0  0  0  0  0
    0.8678    0.1840    0.1637 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1123    0.2851    0.0888 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.2538   -0.8982    0.9504 C   0  0  1  0  0  0  0  0  0  0  0  0
   -0.0339   -2.0308    0.0700 N   0  0  2  0  0  0  0  0  0  0  0  0
   -0.1850    2.1930    0.3696 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.9367   -1.2300    1.7348 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6721   -0.5618    1.4198 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4182   -2.7767    0.6529 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8664   -2.3616   -0.2827 H   0  0  0  0  0  0  0  0  0  0  0  0
  2  4  1  0  0  0  0
  1  2  1  0  0  0  0
  2  3  2  0  0  0  0
  4  5  1  0  0  0  0
  1  6  1  0  0  0  0
  4  7  1  0  0  0  0
  4  8  1  0  0  0  0
  5  9  1  0  0  0  0
  5 10  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02222413353D

 10  9  0  0  0  0            999 V2000
   -1.1655    2.3037   -0.7381 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.2084    0.6713    0.4687 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.9640    0.8139    0.3631 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6578   -0.8554    1.5785 C   0  0  1  0  0  0  0  0  0  0  0  0
   -1.0637   -2.4531    0.3365 N   0  0  2  0  0  0  0  0  0  0  0  0
   -1.2768    3.5054    0.7592 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.3056   -1.3234    2.6851 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9640   -0.3808    2.2407 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6058   -3.5054    1.1588 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.2064   -2.9198   -0.1610 H   0  0  0  0  0  0  0  0  0  0  0  0
  2  4  1  0  0  0  0
  2  3  2  0  0  0  0
  4  5  1  0  0  0  0
  4  7  1  0  0  0  0
  4  8  1  0  0  0  0
  5  9  1  0  0  0  0
  5 10  1  0  0  0  0
  1  2  1  0  0  0  0
  1  6  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    assert mol_a.GetNumAtoms() == mol_b.GetNumAtoms()
    core = np.array([[a, a] for a in range(mol_a.GetNumAtoms())])

    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)

    duplicate_proper_idxs = (0, 1, 3, 4)

    counts_kv_src = dict()
    for idxs, p in zip(st.src_system.proper.potential.idxs, st.src_system.proper.params):
        key = tuple(idxs)
        if p[2] == 2.0:
            if key not in counts_kv_src:
                counts_kv_src[key] = 0
            counts_kv_src[key] += 1  # store period

    assert counts_kv_src[duplicate_proper_idxs] == 2

    counts_kv_dst = dict()
    for idxs, p in zip(st.dst_system.proper.potential.idxs, st.dst_system.proper.params):
        key = tuple(idxs)
        if p[2] == 2.0:
            if key not in counts_kv_dst:
                counts_kv_dst[key] = 0
            counts_kv_dst[key] += 1  # store period

    assert duplicate_proper_idxs not in counts_kv_dst

    st.setup_intermediate_state(0.5)


def test_padded_interpolation():
    # verify that values in the closed interval [0, lambda_min] are exactly equal to src_params,
    # and that values in the closed in interval [lambda_max, 1] are exactly equal to dst_params,

    def interpolate_fn(*_):
        return np.random.rand() * 1000

    # simple 1-parameter case.
    src_k = np.array([1.0, 8.0])
    dst_k = np.array([9.0, 2.0])
    lamb_min = 0.2
    lamb_max = 0.6

    # note: don't replace with np.linspace, we need to test open/closed interval boundaries exactly here.
    # (np.linspace can introduce floating errors breaking expected behavior of equality operators)
    for lamb in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        res = pad(interpolate_fn, src_k, dst_k, lamb, lamb_min, lamb_max)
        if lamb <= lamb_min:
            np.testing.assert_array_equal(res[0], src_k[0])
        elif lamb >= lamb_max:
            np.testing.assert_array_equal(res[0], dst_k[0])
        else:
            pass


def get_aldehyde():
    return Chem.MolFromMolBlock(
        """
  Mrv2311 11042413482D

  4  3  0  0  0  0            999 V2000
    4.7768    0.7115    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.9518    0.7115    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    5.1893   -0.0030    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.1893    1.4260    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
M  END
$$$$
""",
        removeHs=False,
    )


def get_inv_nitrogen():
    return Chem.MolFromMolBlock(
        """
  Mrv2311 11042413493D

  4  3  0  0  0  0            999 V2000
   -0.0110   -0.0156    0.0270 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.0099    0.0080   -0.0138 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3291   -0.4654   -0.8338 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3291    0.9548   -0.0138 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )


def get_non_inv_nitrogen():
    return Chem.MolFromMolBlock(
        """
  Mrv2311 11042413503D

  4  3  0  0  0  0            999 V2000
   -0.0138   -0.0196    0.0339 N   0  0  2  0  0  0  0  0  0  0  0  0
    1.2876    0.0101   -0.0174 F   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4197   -0.5936   -1.0629 F   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4197    1.2173   -0.0174 F   0  0  0  0  0  0  0  0  0  0  0  0
  1  4  1  0  0  0  0
  1  3  1  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )


@pytest.mark.parametrize(
    "mol_a, mol_b, core",
    [
        (get_aldehyde(), get_inv_nitrogen(), np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32)),
        (get_inv_nitrogen(), get_aldehyde(), np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32)),
    ],
)
def test_core_achiral_interpolation_sp2_to_invertible_sp3(mol_a, mol_b, core):
    # test that we're using the default interpolation schedule spanning over the entirely of lambda [0,1]
    # for achiral transformations.
    ff = Forcefield.load_default()

    st = SingleTopology(mol_a, mol_b, core, ff)

    lhs = st.setup_intermediate_state(0.0)
    rhs = st.setup_intermediate_state(1.0)

    # check that chiral volume is undefined in both end-states
    assert len(lhs.chiral_atom.potential.idxs) == 0
    assert len(rhs.chiral_atom.potential.idxs) == 0

    lhs_bond_k = lhs.bond.params[0][0]
    lhs_bond_b0 = lhs.bond.params[0][1]
    lhs_angle_k = lhs.angle.params[0][0]
    lhs_angle_a0 = lhs.angle.params[0][1]

    rhs_bond_k = rhs.bond.params[0][0]
    rhs_bond_b0 = rhs.bond.params[0][1]
    rhs_angle_k = rhs.angle.params[0][0]
    rhs_angle_a0 = rhs.angle.params[0][1]

    # no chiral conversion happens, so we should be using the full schedule for bonds and angles
    atol_k = 1e-3
    atol_b0 = 1e-5
    atol_a0 = 1e-4
    for lamb in np.linspace(0.05, 0.95, 12):
        itm = st.setup_intermediate_state(lamb)
        itm_bond_k = itm.bond.params[0][0]
        itm_bond_b0 = itm.bond.params[0][1]
        itm_angle_k = itm.angle.params[0][0]
        itm_angle_a0 = itm.angle.params[0][1]

        assert abs(itm_bond_k - lhs_bond_k) > atol_k
        assert abs(itm_bond_b0 - lhs_bond_b0) > atol_b0
        assert abs(itm_angle_k - lhs_angle_k) > atol_k
        assert abs(itm_angle_a0 - lhs_angle_a0) > atol_a0

        assert abs(itm_bond_k - rhs_bond_k) > atol_k
        assert abs(itm_bond_b0 - rhs_bond_b0) > atol_b0
        assert abs(itm_angle_k - rhs_angle_k) > atol_k
        assert abs(itm_angle_a0 - rhs_angle_a0) > atol_a0


@pytest.mark.parametrize(
    "mol_a, mol_b, core, direction",
    [
        (get_aldehyde(), get_non_inv_nitrogen(), np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32), "fwd"),
        (get_inv_nitrogen(), get_non_inv_nitrogen(), np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32), "fwd"),
        (get_non_inv_nitrogen(), get_aldehyde(), np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32), "rev"),
        (get_non_inv_nitrogen(), get_inv_nitrogen(), np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32), "rev"),
    ],
)
def test_core_chiral_interpolation_sp2_to_invertible_sp3(mol_a, mol_b, core, direction):
    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)

    lhs = st.setup_intermediate_state(0.0)
    rhs = st.setup_intermediate_state(1.0)

    assert len(lhs.chiral_atom.potential.idxs) == 1
    assert len(rhs.chiral_atom.potential.idxs) == 1
    np.testing.assert_array_equal(lhs.chiral_atom.potential.idxs, rhs.chiral_atom.potential.idxs)

    if direction == "fwd":
        assert lhs.chiral_atom.params[0] == 0
        assert rhs.chiral_atom.params[0] == DEFAULT_CHIRAL_ATOM_RESTRAINT_K
    else:
        assert lhs.chiral_atom.params[0] == DEFAULT_CHIRAL_ATOM_RESTRAINT_K
        assert rhs.chiral_atom.params[0] == 0

    atol_k = 1e-3
    atol_b0 = 1e-5
    atol_a0 = 1e-4

    for lamb in np.linspace(0.0, 1.0, 12):
        itm = st.setup_intermediate_state(lamb)
        for bond_idx in range(3):  # 3 bonds
            for term_idx, term_k in zip(range(2), [atol_k, atol_b0]):  # 2 terms, (k, b0)
                _assert_bonded_term(
                    lamb,
                    bond_idx,
                    term_idx,
                    term_k,
                    single_topology.CORE_BOND_MIN_MAX,
                    itm.bond.params,
                    lhs.bond.params,
                    rhs.bond.params,
                )

        for angle_idx in range(3):  # 3 angles
            for term_idx, term_k in zip(range(2), [atol_k, atol_a0]):  # 2 terms, (k, a0)
                if direction == "fwd":
                    min_max = single_topology.CORE_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX
                else:
                    min_max = single_topology.CORE_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX

                _assert_bonded_term(
                    lamb,
                    angle_idx,
                    term_idx,
                    term_k,
                    min_max,
                    itm.angle.params,
                    lhs.angle.params,
                    rhs.angle.params,
                )

        if direction == "fwd":
            min_max = single_topology.CORE_CHIRAL_ATOM_CONVERTING_ON_MIN_MAX
        else:
            min_max = single_topology.CORE_CHIRAL_ATOM_CONVERTING_OFF_MIN_MAX

        _assert_bonded_term(
            lamb,
            0,
            None,
            atol_k,
            min_max,
            itm.chiral_atom.params,
            lhs.chiral_atom.params,
            rhs.chiral_atom.params,
        )


def get_identity_pair():
    mol = Chem.MolFromMolBlock(
        """identity_ring_pair
  Mrv2311 10092413403D

  6  6  0  0  0  0            999 V2000
    0.1292    1.5540   -0.4103 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8029    0.8102    0.1698 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0572    0.0397    0.8217 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.1063    0.7870    0.2530 C   0  0  2  0  0  0  0  0  0  0  0  0
    2.1161    1.7939    1.5497 F   0  0  0  0  0  0  0  0  0  0  0  0
    1.8910    0.0536   -0.6026 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  1  4  1  0  0  0  0
  4  5  1  0  0  0  0
  4  6  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_a = Chem.Mol(mol)
    mol_b = Chem.Mol(mol)

    return mol_a, mol_b, np.array([[1, 1], [2, 2], [3, 3], [4, 5]])


def get_oxy_ring():
    return Chem.MolFromMolBlock(
        """
  Mrv2311 11042415022D

  5  5  0  0  0  0            999 V2000
    0.4018    1.6964    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3828    1.4415    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.3155    2.5169    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.5733    0.8895    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.2268    1.6964    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  5  1  0  0  0  0
  1  4  1  0  0  0  0
  5  4  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )


def get_broken_ring():
    return Chem.MolFromMolBlock(
        """
  Mrv2311 11042415023D

  4  3  0  0  0  0            999 V2000
   -0.0110   -0.0156    0.0270 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3291   -0.4654   -0.8338 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3291    0.9548   -0.0138 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0099    0.0080   -0.0138 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
M  END
$$$$
""",
        removeHs=False,
    )


def _assert_identical_end_states(angle_idx, itm_params, lhs_params, rhs_params):
    np.testing.assert_allclose(itm_params[angle_idx], rhs_params[angle_idx])
    np.testing.assert_allclose(itm_params[angle_idx], lhs_params[angle_idx])


def _assert_bonded_term(lamb, bonded_idxs, param_idx, atol, min_max, itm_params, lhs_params, rhs_params):
    """Verify that an intermediate bonded term is interpolated correctly in the range [min, max]"""
    if abs(lhs_params[bonded_idxs][param_idx] - rhs_params[bonded_idxs][param_idx]) < 1e-5:
        np.testing.assert_allclose(itm_params[bonded_idxs][param_idx], lhs_params[bonded_idxs][param_idx])
        return

    # test end-states first
    if lamb == 0:
        np.testing.assert_array_equal(itm_params, lhs_params)
    elif lamb == 1.0:
        np.testing.assert_array_equal(itm_params, rhs_params)
    else:
        lamb_min, lamb_max = min_max

        if param_idx is None:
            # chiral atoms have dim=1 parameters
            itm_p = itm_params[bonded_idxs]
            lhs_p = lhs_params[bonded_idxs]
            rhs_p = rhs_params[bonded_idxs]
        else:
            # all other potentials have dim=2 parameters
            itm_p = itm_params[bonded_idxs][param_idx]
            lhs_p = lhs_params[bonded_idxs][param_idx]
            rhs_p = rhs_params[bonded_idxs][param_idx]

        if lamb < lamb_min:
            assert abs(itm_p - lhs_p) < atol
        elif lamb > lamb_min and lamb < lamb_max:
            assert abs(itm_p - lhs_p) > atol
            assert abs(itm_p - rhs_p) > atol
        else:
            assert abs(itm_p - rhs_p) < atol


def test_core_dummy_chiral_conversion():
    #  H1    D4      H1    O4
    #    \  / .        \  / |
    #     N0  .  -->    C0  |
    #    /  \ .        /  \ |
    #  H2    H3      H2    O3
    #
    # note: choice of core anchor and bond broken is arbitrary
    # another possibility is the N0-D4 bond being broken

    # lhs chiral volumes       rhs chiral volumes
    # OFF: N0-H1-H2-H3         ON: N0-H1-H2-H3
    #  ON: N0-H1-D4-H3         ON: N0-H1-D4-H3
    #  ON: N0-H2-D4-H3         ON: N0-H2-D4-H3
    #  ON: N0-H1-D4-H2         ON: N0-H1-D4-H2

    mol_a = get_broken_ring()
    mol_b = get_oxy_ring()
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32)

    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)
    lhs = st.setup_intermediate_state(0.0)
    rhs = st.setup_intermediate_state(1.0)

    assert len(lhs.chiral_atom.potential.idxs) == 4
    assert len(rhs.chiral_atom.potential.idxs) == 4
    np.testing.assert_array_equal(lhs.chiral_atom.potential.idxs, rhs.chiral_atom.potential.idxs)

    assert sum([x == DEFAULT_CHIRAL_ATOM_RESTRAINT_K for x in lhs.chiral_atom.params]) == 3
    assert sum([x == 0 for x in lhs.chiral_atom.params]) == 1

    assert sum([x == DEFAULT_CHIRAL_ATOM_RESTRAINT_K for x in rhs.chiral_atom.params]) == 4
    assert sum([x == 0 for x in rhs.chiral_atom.params]) == 0

    import enum

    class BondTag(enum.IntEnum):
        N0_H3 = 0  # CORE
        H3_D4 = 1  # DUMMY
        N0_H1 = 2  # CORE
        N0_D4 = 3  # DUMMY
        N0_H2 = 4  # CORE

    class AngleTag(enum.IntEnum):
        H2_N0_H3 = 0  # CORE    - CHIRAL
        H1_N0_D4 = 1  # DUMMY   - CHIRAL but non-converting
        H1_N0_H2 = 2  # CORE    - CHIRAL
        H2_N0_D4 = 3  # DUMMY   - CHIRAL but non-converting
        H1_N0_H3 = 4  # CORE    - CHIRAL
        N0_H3_D4 = 5  # DUMMY   - CHIRAL but non-converting
        H3_N0_D4 = 6  # DUMMY   - CHIRAL
        N0_D4_H3 = 7  # DUMMY   - CHIRAL but non-converting

    # print(lhs.chiral_atom.potential.idxs)

    class ChiralAtomTag(enum.IntEnum):
        N0_H1_D4_H2 = 0
        N0_H1_H3_D4 = 1
        N0_H2_H3_D4 = 2
        N0_H1_H3_H2 = 3  # CORE, CHIRAL, CONVERTING

    # no chiral conversion happens, so we should be using the full schedule for bonds and angles
    atol_k = 1e-3
    for lamb in np.linspace(0.05, 0.95, 12):
        itm = st.setup_intermediate_state(lamb)

        # ########## #
        # test bonds #
        # ########## #

        # core bonds:
        #   N0_H3, N0_H1, N0_H2
        # dummy bonds:
        #   H3-D4, N0-D4
        _assert_bonded_term(
            lamb,
            BondTag.N0_H1,
            0,
            atol_k,
            single_topology.CORE_BOND_MIN_MAX,
            itm.bond.params,
            lhs.bond.params,
            rhs.bond.params,
        )
        _assert_bonded_term(
            lamb,
            BondTag.N0_H2,
            0,
            atol_k,
            single_topology.CORE_BOND_MIN_MAX,
            itm.bond.params,
            lhs.bond.params,
            rhs.bond.params,
        )
        _assert_bonded_term(
            lamb,
            BondTag.N0_H3,
            0,
            atol_k,
            single_topology.CORE_BOND_MIN_MAX,
            itm.bond.params,
            lhs.bond.params,
            rhs.bond.params,
        )

        # dummy bond N0-D4 stays the same
        assert abs(itm.bond.params[BondTag.N0_D4][0] - lhs.bond.params[BondTag.N0_D4][0]) < atol_k
        assert abs(itm.bond.params[BondTag.N0_D4][0] - rhs.bond.params[BondTag.N0_D4][0]) < atol_k

        # dummy bond H3-D4 is interpolated over achiral bounds
        _assert_bonded_term(
            lamb,
            BondTag.H3_D4,
            0,
            atol_k,
            single_topology.DUMMY_B_BOND_MIN_MAX,
            itm.bond.params,
            lhs.bond.params,
            rhs.bond.params,
        )
        _assert_bonded_term(
            lamb,
            BondTag.N0_H2,
            0,
            atol_k,
            single_topology.CORE_BOND_MIN_MAX,
            itm.bond.params,
            lhs.bond.params,
            rhs.bond.params,
        )

        # ########### #
        # test angles #
        # ########### #

        # test angle interpolation for chiral core atoms
        _assert_bonded_term(
            lamb,
            AngleTag.H2_N0_H3,
            0,
            atol_k,
            single_topology.CORE_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX,
            itm.angle.params,
            lhs.angle.params,
            rhs.angle.params,
        )
        _assert_bonded_term(
            lamb,
            AngleTag.H1_N0_H2,
            0,
            atol_k,
            single_topology.CORE_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX,
            itm.angle.params,
            lhs.angle.params,
            rhs.angle.params,
        )

        _assert_bonded_term(
            lamb,
            AngleTag.H1_N0_H3,
            0,
            atol_k,
            single_topology.CORE_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX,
            itm.angle.params,
            lhs.angle.params,
            rhs.angle.params,
        )

        # test angle interpolation for achiral dummy atoms
        _assert_bonded_term(
            lamb,
            AngleTag.H1_N0_D4,
            0,
            atol_k,
            single_topology.DUMMY_B_ANGLE_MIN_MAX,
            itm.angle.params,
            lhs.angle.params,
            rhs.angle.params,
        )

        # lhs/rhs are the same in this case.
        _assert_bonded_term(
            lamb,
            AngleTag.H2_N0_D4,
            0,
            atol_k,
            single_topology.DUMMY_B_ANGLE_MIN_MAX,
            itm.angle.params,
            lhs.angle.params,
            rhs.angle.params,
        )

        _assert_bonded_term(
            lamb,
            AngleTag.N0_H3_D4,
            0,
            atol_k,
            single_topology.DUMMY_B_ANGLE_MIN_MAX,
            itm.angle.params,
            lhs.angle.params,
            rhs.angle.params,
        )

        _assert_bonded_term(
            lamb,
            AngleTag.N0_D4_H3,
            0,
            atol_k,
            single_topology.DUMMY_B_ANGLE_MIN_MAX,
            itm.angle.params,
            lhs.angle.params,
            rhs.angle.params,
        )

        _assert_bonded_term(
            lamb,
            AngleTag.H3_N0_D4,
            0,
            atol_k,
            single_topology.DUMMY_B_ANGLE_MIN_MAX,
            itm.angle.params,
            lhs.angle.params,
            rhs.angle.params,
        )

        # ################### #
        # test chiral volumes #
        # ################### #
        _assert_bonded_term(
            lamb,
            ChiralAtomTag.N0_H1_H3_H2,
            None,
            atol_k,
            single_topology.CORE_CHIRAL_ATOM_CONVERTING_ON_MIN_MAX,
            itm.chiral_atom.params,
            lhs.chiral_atom.params,
            rhs.chiral_atom.params,
        )

        _assert_identical_end_states(
            ChiralAtomTag.N0_H1_D4_H2, itm.chiral_atom.params, lhs.chiral_atom.params, rhs.chiral_atom.params
        )
        _assert_identical_end_states(
            ChiralAtomTag.N0_H1_H3_D4, itm.chiral_atom.params, lhs.chiral_atom.params, rhs.chiral_atom.params
        )
        _assert_identical_end_states(
            ChiralAtomTag.N0_H2_H3_D4, itm.chiral_atom.params, lhs.chiral_atom.params, rhs.chiral_atom.params
        )


def _assert_exception_raised_at_least_once_in_interval(lambda_schedule, min_max, fn, expected_exception):
    found = False
    for lam in lambda_schedule:
        if lam >= min_max[0] and lam <= min_max[1]:
            try:
                fn(lam)
            except expected_exception:
                found = True
                break
    assert found


def test_assert_bonds_present_during_chiral_interpolation():
    """We expect that if we set the threshold criteria (defining whether or not a bond is present based on the force constant)
    too low, then we should raise MissingBondsInChiralVolumeException exceptions at the intermediate states."""
    mol_a, mol_b, core = get_identity_pair()
    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)
    lambdas = np.linspace(0, 1, 48)

    for lam in lambdas:
        vs = st.setup_intermediate_state(lam)
        assert_bonds_defined_for_chiral_volumes(vs, DEFAULT_BOND_IS_PRESENT_K)

    def assert_fn(lam):
        vs = st.setup_intermediate_state(lam)
        assert_bonds_defined_for_chiral_volumes(vs, bond_k_min=200.0)

    _assert_exception_raised_at_least_once_in_interval(
        lambdas, DUMMY_B_CHIRAL_ATOM_CONVERTING_ON_MIN_MAX, assert_fn, MissingBondsInChiralVolumeException
    )
    _assert_exception_raised_at_least_once_in_interval(
        lambdas, DUMMY_A_CHIRAL_ATOM_CONVERTING_OFF_MIN_MAX, assert_fn, MissingBondsInChiralVolumeException
    )


def get_pfkfb3_nitrile_to_amide_fwd():
    with resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf") as path:
        mols_by_name = read_sdf_mols_by_name(path)
    mol_a = mols_by_name["24"]
    mol_b = mols_by_name["26"]
    core = np.array(
        [
            [20, 20],
            [26, 22],
            [25, 23],
            [24, 24],
            [23, 25],
            [22, 26],
            [21, 21],
            [19, 19],
            [18, 18],
            [17, 17],
            [16, 16],
            [15, 15],
            [14, 14],
            [13, 13],
            [12, 12],
            [11, 11],
            [10, 10],
            [9, 3],
            [8, 2],
            [7, 1],
            [6, 0],
            [5, 4],
            [4, 5],
            [3, 8],
            [2, 7],
            [27, 27],
            [1, 6],
            [0, 9],
            [45, 43],
            [44, 44],
            [43, 45],
            [42, 46],
            [41, 42],
            [40, 41],
            [39, 40],
            [38, 39],
            [37, 38],
            [36, 37],
            [35, 36],
            [34, 35],
            [33, 34],
            [32, 31],
            [31, 30],
            [30, 33],
            [29, 32],
            [28, 29],
        ]
    )

    return mol_a, mol_b, core


def get_pfkfb3_nitrile_to_amide_rev():
    mol_a, mol_b, core = get_pfkfb3_nitrile_to_amide_fwd()
    return mol_b, mol_a, core[:, ::-1]


@pytest.mark.parametrize(
    "mol_a, mol_b, core",
    [
        get_pfkfb3_nitrile_to_amide_fwd(),
        get_pfkfb3_nitrile_to_amide_rev(),
    ],
)
def test_assert_torsions_defined_over_non_linear_angles(mol_a, mol_b, core, monkeypatch):
    """We expect that if we set the threshold criteria (defining whether or not a bond is present based on the force constant)
    too low, then we should raise MissingBondsInChiralVolumeException exceptions at the intermediate states."""

    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)

    # plot_and_save(plot_core_interpolation_schedule, "pfkfb3_interpolation_schedule_core.png", st)
    # plot_and_save(plot_dummy_a_interpolation_schedule, "pfkfb3_interpolation_schedule_dummy_a.png", st)
    # plot_and_save(plot_dummy_b_interpolation_schedule, "pfkfb3_interpolation_schedule_dummy_b.png", st)

    lambdas = np.linspace(0, 1, 12)

    for lam in lambdas:
        vs = st.setup_intermediate_state(lam)
        assert_torsions_defined_over_non_linear_angles(vs)

    monkeypatch.setattr(single_topology, "CORE_TORSION_OFF_TO_ON_MIN_MAX", [0.0, 1.0])
    monkeypatch.setattr(single_topology, "CORE_TORSION_ON_TO_OFF_MIN_MAX", [0.0, 1.0])

    def assert_fn(lam):
        vs = st.setup_intermediate_state(lam)
        assert_torsions_defined_over_non_linear_angles(vs)

    _assert_exception_raised_at_least_once_in_interval(
        lambdas, (0.0, 1.0), assert_fn, TorsionsDefinedOverLinearAngleException
    )
