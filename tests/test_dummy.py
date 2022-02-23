import numpy as np
from rdkit import Chem

from timemachine.fe import dummy, dummy_draw
from timemachine.fe.dummy import (
    enumerate_anchor_groups,
    enumerate_dummy_ixns,
    flag_bonds,
    generate_optimal_dg_ag_pairs,
    identify_anchor_groups,
    identify_dummy_groups,
    identify_root_anchors,
)
from timemachine.ff import Forcefield
from timemachine.ff.handlers.deserialize import deserialize_handlers

# These tests check the various utilities used to turn off interactions
# such that the end-states are separable. A useful glossary of terms is as follows.
# The tests in this module do not test for numerical stability, but only checks for correctness.

# Many of the tests here are tediously written by hand, but should hopefully be sufficiently documented.

# dummy atom - an R-group atom that is inserted or deleted in an alchemical transformation
# dummy group - a collection of dummy atoms (eg. multiple dummy hydrogen atoms on CH3 can belong to the same dummy group, with C being a core atom).
# core atom - not a dummy atom
# anchor/core anchor - a core atom that is allowed to interact with the atoms in a given dummy group
# anchor group - a set of ordered anchor atoms (up to 3) that can be used to define bond, angle, torsion terms in specialized ways
# root-anchor - first atom in an anchor group, also the anchor atom that has direct 1-2 bonds to dummy atoms.
# partition - only applies to dummy groups, as we require that dummy groups disjointly partition dummy atoms.


def test_identify_root_anchors():
    """
    Test the identification of root anchors given a dummy atom.

    For example, if D were the dummy atom below:

            D---1---2
           /
          0

    its root anchors are the set of atoms {0, 1}

    """
    mol = Chem.MolFromSmiles("C1CCC1N")
    core = [0, 1, 2, 3]
    anchors = identify_root_anchors(mol, core, dummy_atom=4)
    assert set(anchors) == set([3])

    mol = Chem.MolFromSmiles("C1CC2NC2C1")
    core = [0, 1, 2, 4, 5]
    anchors = identify_root_anchors(mol, core, dummy_atom=3)
    assert set(anchors) == set([2, 4])

    mol = Chem.MolFromSmiles("C1OCC11CCCCC1")
    core = [3, 4, 5, 6, 7, 8]
    anchors = identify_root_anchors(mol, core, dummy_atom=0)
    assert set(anchors) == set([3])
    anchors = identify_root_anchors(mol, core, dummy_atom=1)
    assert set(anchors) == set([3])
    anchors = identify_root_anchors(mol, core, dummy_atom=2)
    assert set(anchors) == set([3])

    mol = Chem.MolFromSmiles("C1CC1.C1CCCCC1")
    core = [3, 4, 5, 6, 7, 8]
    anchors = identify_root_anchors(mol, core, dummy_atom=0)
    assert set(anchors) == set()
    anchors = identify_root_anchors(mol, core, dummy_atom=1)
    assert set(anchors) == set()
    anchors = identify_root_anchors(mol, core, dummy_atom=2)
    assert set(anchors) == set()

    mol = Chem.MolFromSmiles("C1CC2NC2C1")
    core = [0, 1, 2, 5]
    anchors = identify_root_anchors(mol, core, dummy_atom=3)
    assert set(anchors) == set([2, 5])
    anchors = identify_root_anchors(mol, core, dummy_atom=4)
    assert set(anchors) == set([2, 5])

    # cyclohexane with a nitrogen inside
    mol = Chem.MolFromSmiles("C1C2CC3CC1N23")
    core = [0, 1, 2, 3, 4, 5]
    anchors = identify_root_anchors(mol, core, dummy_atom=6)
    assert set(anchors) == set([1, 3, 5])


def assert_set_equality(a_sets, b_sets):
    # utility function to check that list of sets are equal
    frozen_a = [frozenset(a) for a in a_sets]
    frozen_b = [frozenset(b) for b in b_sets]
    assert frozenset(frozen_a) == frozenset(frozen_b)


def test_identify_dummy_groups():
    r"""
    Test the heuristic for partitioning dummy atoms into dummy groups.

    Given a system such as

        D0-D1  D2
       /     \ /
      0-------1

    The dummy groups identified should be {D0}, {D1, D2}. This partioning
    maximizes the number of bonded terms that we can leave on:

        D0 D1  D2
       /     \ /  -> 3 dummy-anchor bonds
      0-------1

    Alternatively valid, but less efficient choices of dummy groups would be:

    {D0, D1}, {D2}

        D0-D1  D2
       /       /  -> 2 dummy-anchor bonds:
      0-------1

    {D0, D1, D2}

        D0-D1..D2
       /          -> 1 dummy-anchor bond:
      0-------1

    """
    mol = Chem.MolFromSmiles("FC1CC1(F)N")
    core = [1, 2, 3]
    dg = identify_dummy_groups(mol, core)
    assert_set_equality(dg, [{0}, {4, 5}])

    mol = Chem.MolFromSmiles("FC1CC1(F)NN")
    core = [1, 2, 3]
    dg = identify_dummy_groups(mol, core)
    assert_set_equality(dg, [{0}, {4, 5, 6}])

    mol = Chem.MolFromSmiles("C1CC11OO1")
    core = [0, 1, 2]
    dg = identify_dummy_groups(mol, core)
    assert_set_equality(dg, [{3, 4}])

    mol = Chem.MolFromSmiles("C1CC2OOC12")
    core = [0, 1, 2, 5]
    dg = identify_dummy_groups(mol, core)
    assert_set_equality(dg, [{3}, {4}])

    # example above, where O's are dummy atoms, and Cs are core
    mol = Chem.MolFromSmiles("OC1COO1")
    core = [1, 2]
    dg = identify_dummy_groups(mol, core)
    assert_set_equality(dg, [{0, 4}, {3}])


def assert_anchor_group_equality(a_groups, b_groups):
    frozen_a = [tuple(a) for a in a_groups]
    frozen_b = [tuple(b) for b in b_groups]
    assert frozenset(frozen_a) == frozenset(frozen_b)


def test_identify_anchor_groups():
    """
    Test that we can correctly enumerate all anchor groups given a particular
    starting root_anchor.
    """

    mol = Chem.MolFromSmiles("C1CC1F")
    core = [0, 1, 2]
    groups_of_1, groups_of_2, groups_of_3 = identify_anchor_groups(mol, core, 0)
    assert_anchor_group_equality(groups_of_1, [[0]])
    assert_anchor_group_equality(groups_of_2, [[0, 1], [0, 2]])
    assert_anchor_group_equality(groups_of_3, [[0, 1, 2], [0, 2, 1]])

    groups_of_1, groups_of_2, groups_of_3 = identify_anchor_groups(mol, core, 1)
    assert_anchor_group_equality(groups_of_1, [[1]])
    assert_anchor_group_equality(groups_of_2, [[1, 0], [1, 2]])
    assert_anchor_group_equality(groups_of_3, [[1, 0, 2], [1, 2, 0]])

    groups_of_1, groups_of_2, groups_of_3 = identify_anchor_groups(mol, core, 2)
    assert_anchor_group_equality(groups_of_1, [[2]])
    assert_anchor_group_equality(groups_of_2, [[2, 1], [2, 0]])
    assert_anchor_group_equality(groups_of_3, [[2, 1, 0], [2, 0, 1]])

    mol = Chem.MolFromSmiles("NCCF")
    core = [1, 2]
    groups_of_1, groups_of_2, groups_of_3 = identify_anchor_groups(mol, core, 1)
    assert_anchor_group_equality(groups_of_1, [[1]])
    assert_anchor_group_equality(groups_of_2, [[1, 2]])
    assert_anchor_group_equality(groups_of_3, [])

    mol = Chem.MolFromSmiles("C(C)(C)(C)C")
    core = [0]
    groups_of_1, groups_of_2, groups_of_3 = identify_anchor_groups(mol, core, 0)
    assert_anchor_group_equality(groups_of_1, [[0]])
    assert_anchor_group_equality(groups_of_2, [])
    assert_anchor_group_equality(groups_of_3, [])

    mol = Chem.MolFromSmiles("C(C)(F)CC")
    core = [1, 0, 4, 3]
    groups_of_1, groups_of_2, groups_of_3 = identify_anchor_groups(mol, core, 0)
    assert_anchor_group_equality(groups_of_1, [[0]])
    assert_anchor_group_equality(groups_of_2, [[0, 1], [0, 3]])
    assert_anchor_group_equality(groups_of_3, [[0, 3, 4]])


def test_enumerate_anchor_groups():
    """
    Test enumeration of all reasonable anchor groups that span from all possible anchor roots.
    For clarity, we explicitly enumerate size 1, 2 and 3 anchor groups.
    """
    mol = Chem.MolFromSmiles("C1CC2OOC12")
    core = [0, 1, 2, 5]
    groups_of_1, groups_of_2, groups_of_3 = enumerate_anchor_groups(mol, core, [3, 4])
    assert_anchor_group_equality(groups_of_1, [[2], [5]])
    assert_anchor_group_equality(groups_of_2, [[2, 1], [2, 5], [5, 0], [5, 2]])
    assert_anchor_group_equality(groups_of_3, [[2, 1, 0], [2, 5, 0], [5, 0, 1], [5, 2, 1]])

    # aspirin
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    core = [3, 4, 5, 6, 7, 8, 9, 10]
    groups_of_1, groups_of_2, groups_of_3 = enumerate_anchor_groups(mol, core, [0, 1, 2, 11, 12])
    assert_anchor_group_equality(groups_of_1, [[3], [10]])
    assert_anchor_group_equality(groups_of_2, [[3, 4], [10, 9]])
    assert_anchor_group_equality(groups_of_3, [[3, 4, 5], [3, 4, 9], [10, 9, 8], [10, 9, 4]])
    groups_of_1, groups_of_2, groups_of_3 = enumerate_anchor_groups(mol, core, [0, 1, 2])
    assert_anchor_group_equality(groups_of_1, [[3]])
    assert_anchor_group_equality(groups_of_2, [[3, 4]])
    assert_anchor_group_equality(groups_of_3, [[3, 4, 5], [3, 4, 9]])
    groups_of_1, groups_of_2, groups_of_3 = enumerate_anchor_groups(mol, core, [11, 12])
    assert_anchor_group_equality(groups_of_1, [[10]])
    assert_anchor_group_equality(groups_of_2, [[10, 9]])
    assert_anchor_group_equality(groups_of_3, [[10, 9, 8], [10, 9, 4]])


def test_enumerate_allowed_dummy_ixns_3_anchors_spot_check():
    """
    Test generation of allowed dummy interactions with the anchor groups. This is a spot check.
    A more exhaustive 3_anchor one is below.
    """
    dummy_group = {0, 1, 5, 8}
    anchor_group = [2, 3, 4]

    allowed_ixns = enumerate_dummy_ixns(dummy_group, anchor_group)

    for ixn in allowed_ixns:
        assert len(ixn) == len(set(ixn))

    assert (0, 1, 2, 3) in allowed_ixns
    assert (0, 1, 2) in allowed_ixns
    assert (1, 2) in allowed_ixns
    assert (0, 2, 3) in allowed_ixns
    assert (1, 0, 2, 3) in allowed_ixns
    assert (3, 2, 1, 5) in allowed_ixns
    assert (1, 5, 2) in allowed_ixns
    assert (2, 0, 5) in allowed_ixns
    assert (1, 5, 8) in allowed_ixns
    assert (1, 5, 8) in allowed_ixns
    assert (0, 8, 5, 1) in allowed_ixns
    assert (5, 1, 0, 8) in allowed_ixns
    assert (3, 2, 8, 5) in allowed_ixns
    assert (1, 8, 2, 3) in allowed_ixns

    # root anchor = 2
    # second anchor = 3
    # third anchor = 4
    assert (0, 3) not in allowed_ixns  # dummy-bond only allowed to root anchor
    assert (0, 4) not in allowed_ixns  # dummy-bond only allowed to root anchor
    assert (0, 1, 3) not in allowed_ixns  # angles can't skip the root anchor
    assert (0, 1, 4) not in allowed_ixns  # angles can't skip the root anchor
    assert (0, 2, 4) not in allowed_ixns  # angles can't skip the second anchor
    assert (5, 3, 4) not in allowed_ixns  # angles can't skip the root anchor
    assert (1, 5, 4) not in allowed_ixns  # angles can't skip the root anchor
    assert (5, 8, 2, 4) not in allowed_ixns  # torsion can't skip the second anchor
    assert (1, 8, 3, 4) not in allowed_ixns  # torsion can't skip the root anchor


def test_enumerate_allowed_dummy_ixns_3_anchors_exhaustive():
    """
    Test generation of allowed dummy interactions with the anchor groups exhaustively.
    """
    dummy_group = {0, 1}
    anchor_group = [2, 3, 4]

    allowed_ixns = enumerate_dummy_ixns(dummy_group, anchor_group)

    for ixn in allowed_ixns:
        assert len(ixn) == len(set(ixn))

    assert len(allowed_ixns) == 12

    # 3 ixns
    assert (0, 1) in allowed_ixns
    assert (0, 2) in allowed_ixns
    assert (1, 2) in allowed_ixns

    # 5 ixns
    assert (0, 1, 2) in allowed_ixns
    assert (1, 0, 2) in allowed_ixns
    assert (0, 2, 3) in allowed_ixns
    assert (1, 2, 3) in allowed_ixns
    assert (0, 2, 1) in allowed_ixns

    # 3 ixns
    assert (1, 0, 2, 3) in allowed_ixns
    assert (0, 1, 2, 3) in allowed_ixns
    assert (0, 2, 3, 4) in allowed_ixns
    assert (1, 2, 3, 4) in allowed_ixns


def test_enumerate_allowed_dummy_ixns_2_anchors():
    """
    Same as above but with 2 anchors
    """
    dummy_group = {0, 1}
    anchor_group = [4, 5]
    allowed_ixns = enumerate_dummy_ixns(dummy_group, anchor_group)

    for ixn in allowed_ixns:
        assert len(ixn) == len(set(ixn))
    assert len(allowed_ixns) == 10

    # 3 ixns
    assert (0, 1) in allowed_ixns
    assert (0, 4) in allowed_ixns
    assert (1, 4) in allowed_ixns

    # 5 ixns
    assert (0, 1, 4) in allowed_ixns
    assert (1, 0, 4) in allowed_ixns
    assert (0, 4, 1) in allowed_ixns
    assert (0, 4, 5) in allowed_ixns
    assert (1, 4, 5) in allowed_ixns

    # 2 ixns
    assert (0, 1, 4, 5) in allowed_ixns
    assert (1, 0, 4, 5) in allowed_ixns


def test_enumerate_allowed_dummy_ixns_1_anchor():
    """
    Same as above but with 1 anchor
    """
    dummy_group = {0, 1}
    anchor_group = [4]
    allowed_ixns = enumerate_dummy_ixns(dummy_group, anchor_group)
    assert len(allowed_ixns) == 6
    assert (0, 1) in allowed_ixns
    assert (0, 4) in allowed_ixns
    assert (1, 4) in allowed_ixns
    assert (0, 1, 4) in allowed_ixns
    assert (1, 0, 4) in allowed_ixns
    assert (0, 4, 1) in allowed_ixns


def test_flag_bonds():
    """
    Test that we flag bonds on and off correctly given a set bond_idxs.
    """
    mol = Chem.MolFromSmiles("BrOC1=CC(F)=CC=N1")
    core = [2, 3, 4, 6, 7, 8]
    bond_pairs = [
        (1, [0, 1]),  # Br-O
        (1, [1, 2]),  # O-C
        (1, [2, 3]),  # C-C
        (1, [3, 4]),  # C-C
        (1, [4, 5]),  # C-F
        (1, [4, 6]),  # C-C
        (1, [6, 7]),  # C-C
        (1, [7, 8]),  # C-N
        (1, [8, 2]),  # N-C
        (1, [0, 1, 2]),  # Br-O-C
        (1, [1, 2, 3]),  # O-C-C
        (1, [1, 2, 3, 4]),  # O-C-C-C
        (0, [1, 2, 7]),  # O-C-N
        (1, [2, 3, 4]),  # C-C-C
        (1, [5, 4, 3, 2]),  # F-C-C-C
        (1, [5, 4, 3]),  # F-C-C
        (0, [5, 4, 6]),  # F-C-C
    ]

    expected_flags = [x[0] for x in bond_pairs]
    bond_idxs = [x[1] for x in bond_pairs]

    keep_flags = flag_bonds(mol, core, bond_idxs)
    assert tuple(keep_flags) == tuple(expected_flags)

    # flipping the ordering should give us identical results
    bond_idxs = [x[::-1] for x in bond_idxs]
    keep_flags = flag_bonds(mol, core, bond_idxs)
    assert tuple(keep_flags) == tuple(expected_flags)


def test_flag_bonds_core_hop():
    """
    Test that if we do a core hop, one of the bonded terms should be turned off.
    """

    mol = Chem.MolFromSmiles("FC1CO1")

    #    F0       F
    #    |        .
    #   _C1  ->  .C
    # 3O |      O |
    #   \C2       C
    core = [1, 2]
    bond_pairs = [
        (1, [0, 1]),
        (1, [1, 2]),
        (0, [2, 3]),
        (1, [3, 1]),
        (1, [0, 1, 2]),
        (1, [0, 1, 3]),
        (1, [3, 1, 2]),
        (0, [1, 3, 2]),
        (0, [1, 2, 3]),
    ]

    # TBD: we can prune this to only terms where the bonds actually exist between ijk, ijkl, etc.

    expected_flags = [x[0] for x in bond_pairs]
    bond_idxs = [x[1] for x in bond_pairs]

    keep_flags = flag_bonds(mol, core, bond_idxs)

    assert tuple(keep_flags) == tuple(expected_flags)


def test_dg_ag_missing_ixns_split():
    """
    Explicitly test enumeration when certain bond and angle terms are missing.
    """

    # test the following behavior for the molecule
    #    2
    #    |
    # 0--1--3
    # test that we correct allowed interactions

    mol = Chem.MolFromSmiles("CC(C)C")
    core = [1, 3]

    # base case, all required terms for the torsion is present
    bond_idxs = [
        [0, 1],
        [1, 2],
        [1, 3],
        [0, 2],
        [0, 1, 2],
        [0, 1, 3],
        [2, 1, 3],
        [0, 2, 1],
        [0, 2, 1, 3],
    ]

    dgs, agcs, agis = generate_optimal_dg_ag_pairs(mol, core, bond_idxs)
    for dgs, agcs, agis in zip(dgs, agcs, agis):
        assert (0, 2, 1, 3) in agis
        assert (0, 2, 1) in agis
        assert (0, 2) in agis

    # missing angle
    bond_idxs = [
        [0, 1],
        [1, 2],
        [1, 3],
        [0, 2],
        [0, 1, 2],
        [0, 1, 3],
        [2, 1, 3],
        # [0, 2, 1],
        [0, 2, 1, 3],
    ]

    dgs, agcs, agis = generate_optimal_dg_ag_pairs(mol, core, bond_idxs)
    for dgs, agcs, agis in zip(dgs, agcs, agis):
        assert (0, 2, 1, 3) not in agis
        assert (0, 2, 1) not in agis
        assert (0, 2) in agis

    # missing bond
    bond_idxs = [
        [0, 1],
        [1, 2],
        [1, 3],
        # [0, 2],
        [0, 1, 2],
        [0, 1, 3],
        [2, 1, 3],
        [0, 2, 1],
        [0, 2, 1, 3],
    ]

    dgs, agcs, agis = generate_optimal_dg_ag_pairs(mol, core, bond_idxs)
    for dgs, agcs, agis in zip(dgs, agcs, agis):
        assert (0, 2, 1, 3) not in agis
        assert (0, 2, 1) not in agis
        assert (0, 2) not in agis

    # missing bond and angle
    bond_idxs = [
        [0, 1],
        [1, 2],
        [1, 3],
        # [0, 2],
        [0, 1, 2],
        [0, 1, 3],
        [2, 1, 3],
        # [0, 2, 1],
        [0, 2, 1, 3],
    ]

    dgs, agcs, agis = generate_optimal_dg_ag_pairs(mol, core, bond_idxs)
    for dgs, agcs, agis in zip(dgs, agcs, agis):
        assert (0, 2, 1, 3) not in agis
        assert (0, 2, 1) not in agis
        assert (0, 2) not in agis


def test_dg_ag_missing_ixns_linear():
    """
    Explicitly test enumeration when certain bond and angle terms are missing.
    Identical to above test but for a linear chain.
    """

    # let the molecule be a linear chain
    # 0-1-2-3

    # we want to test correctness of the strict when parts of the
    # torsion definition is missing

    mol = Chem.MolFromSmiles("CCCC")
    core = [1, 2, 3]

    # base case, all required terms for the torsion is present
    bond_idxs = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 1, 2],
        [1, 2, 3],
        [0, 1, 2, 3],
    ]

    dgs, agcs, agis = generate_optimal_dg_ag_pairs(mol, core, bond_idxs)
    for dgs, agcs, agis in zip(dgs, agcs, agis):
        assert (0, 1, 2, 3) in agis

    # disable angle
    bond_idxs = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 1, 2],
        # [1, 2, 3],
        [0, 1, 2, 3],
    ]

    dgs, agcs, agis = generate_optimal_dg_ag_pairs(mol, core, bond_idxs)
    for dgs, agcs, agis in zip(dgs, agcs, agis):
        assert (0, 1, 2, 3) not in agis
        assert (1, 2, 3) not in agis

    # disable bond
    bond_idxs = [
        [0, 1],
        [1, 2],
        # [2, 3],
        [0, 1, 2],
        [1, 2, 3],
        [0, 1, 2, 3],
    ]

    dgs, agcs, agis = generate_optimal_dg_ag_pairs(mol, core, bond_idxs)
    for dgs, agcs, agis in zip(dgs, agcs, agis):
        assert (0, 1, 2, 3) not in agis
        assert (1, 2, 3) not in agis


def _draw_impl(mol, core, ff, fname):

    hb_params, hb_idxs = ff.hb_handle.parameterize(mol)
    ha_params, ha_idxs = ff.ha_handle.parameterize(mol)
    pt_params, pt_idxs = ff.pt_handle.parameterize(mol)
    it_params, it_idxs = ff.it_handle.parameterize(mol)

    bond_idxs = (
        [tuple(x.tolist()) for x in hb_idxs]
        + [tuple(x.tolist()) for x in ha_idxs]
        + [tuple(x.tolist()) for x in pt_idxs]
        + [tuple(x.tolist()) for x in it_idxs]
    )

    dgs, ags, ag_ixns = dummy.generate_optimal_dg_ag_pairs(mol, core, bond_idxs)

    for idx, (dummy_group, anchor_group, anchor_ixns) in enumerate(zip(dgs, ags, ag_ixns)):

        matched_ixns = []
        for idxs in bond_idxs:
            if tuple(idxs) in anchor_ixns:
                if np.all([ii in dummy_group for ii in idxs]):
                    continue
                elif np.all([ii in core for ii in idxs]):
                    continue
                matched_ixns.append(idxs)

        res = dummy_draw.draw_dummy_core_ixns(mol, core, matched_ixns, dummy_group)

        with open(fname + "_" + str(idx) + ".svg", "w") as fh:
            fh.write(res)


def test_parameterize_and_draw_interactions():
    """
    This isn't really tested, but is here to verify that drawing code at least runs.
    """

    ff_handlers = deserialize_handlers(open("timemachine/ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)

    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    core = [3, 4, 5, 6, 7, 8, 9, 10]

    _draw_impl(mol, core, ff, "aspirin")

    mol = Chem.MolFromSmiles("C(C1=CC=CC=C1)C1=CC=CC=C1")
    core = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    _draw_impl(mol, core, ff, "met_biphenyl")

    mol = Chem.MolFromSmiles("F[C@](Cl)(Br)C1=CC=CC=C1")
    core = [1, 4, 5, 6, 7, 8, 9]

    _draw_impl(mol, core, ff, "toluene")

    mol = Chem.MolFromSmiles("C1CC2=CC=CC=C12")
    core = [2, 3, 4, 5, 6, 7]

    _draw_impl(mol, core, ff, "ring_open")
