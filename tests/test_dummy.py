import pytest
from rdkit import Chem

from timemachine.fe.dummy import (
    ZeroBondAnchorWarning,
    generate_anchored_dummy_group_assignments,
    generate_dummy_group_assignments,
)
from timemachine.graph_utils import convert_to_nx

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

pytestmark = [pytest.mark.nocuda]


def equivalent_assignment(left, right):
    def to_comparable(dgas):
        return frozenset(frozenset((k, frozenset(v)) for k, v in dgs.items()) for dgs in dgas)

    return to_comparable(left) == to_comparable(right)


def test_generate_dummy_group_assignments():
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

    g = convert_to_nx(Chem.MolFromSmiles("FC1CC1(F)N"))
    core = [1, 2, 3]
    dgas = list(generate_dummy_group_assignments(g, core))
    assert equivalent_assignment(dgas, [{1: {0}, 3: {4, 5}}])

    g = convert_to_nx(Chem.MolFromSmiles("FC1CC1(F)NN"))
    core = [1, 2, 3]
    dgas = list(generate_dummy_group_assignments(g, core))
    assert equivalent_assignment(dgas, [{1: {0}, 3: {4, 5, 6}}])

    g = convert_to_nx(Chem.MolFromSmiles("C1CC11OO1"))
    core = [0, 1, 2]
    dgas = list(generate_dummy_group_assignments(g, core))
    assert equivalent_assignment(dgas, [{2: {3, 4}}])

    g = convert_to_nx(Chem.MolFromSmiles("C1CC2OOC12"))
    core = [0, 1, 2, 5]
    dgas = list(generate_dummy_group_assignments(g, core))
    assert equivalent_assignment(dgas, [{2: {3, 4}}, {5: {3, 4}}])

    # example above, where O's are dummy atoms, and Cs are core
    g = convert_to_nx(Chem.MolFromSmiles("OC1COO1"))
    core = [1, 2]
    dgas = list(generate_dummy_group_assignments(g, core))
    # one or two groups depending on choice of anchor atom for {3, 4}
    assert equivalent_assignment(dgas, [{1: {0}, 2: {3, 4}}, {1: {0, 3, 4}}])


def test_generate_dummy_group_assignments_empty_core():
    g = convert_to_nx(Chem.MolFromSmiles("OC1COO1"))
    core = []

    with pytest.warns(ZeroBondAnchorWarning):
        dgas = list(generate_dummy_group_assignments(g, core))

    assert equivalent_assignment(dgas, [{None: {0, 1, 2, 3, 4}}])


def test_generate_dummy_group_assignments_full_core():
    g = convert_to_nx(Chem.MolFromSmiles("OC1COO1"))
    core = [0, 1, 2, 3, 4]
    dgas = list(generate_dummy_group_assignments(g, core))
    assert equivalent_assignment(dgas, [{}])


def test_generate_angle_anchor_dummy_group_assignments():
    # Test that if we break a core-core bond, we only have one valid
    # choice of the angle anchor
    #
    #      O0          O0
    #      |           |
    #      C1          C1
    #     / \           \
    #    O4  C2  ->  O4  C2
    #     \ /         \ /
    #      O3          O3
    g_a = convert_to_nx(Chem.MolFromSmiles("OC1COO1"))
    g_b = convert_to_nx(Chem.MolFromSmiles("OCCOO"))
    core_a = [1, 2, 3, 4]
    core_b = [1, 2, 3, 4]

    dgas = list(generate_dummy_group_assignments(g_a, core_a))
    expected_dga = {1: {0}}

    assert equivalent_assignment(dgas, [expected_dga])

    # forward direction
    anchored_dummy_group_assignments = generate_anchored_dummy_group_assignments(expected_dga, g_a, g_b, core_a, core_b)

    anchored_dummy_group_assignments = list(anchored_dummy_group_assignments)
    assert len(anchored_dummy_group_assignments) == 1
    assert anchored_dummy_group_assignments[0] == {1: (2, frozenset({0}))}

    # reverse direction
    anchored_dummy_group_assignments = generate_anchored_dummy_group_assignments(expected_dga, g_b, g_a, core_b, core_a)

    anchored_dummy_group_assignments = list(anchored_dummy_group_assignments)
    assert len(anchored_dummy_group_assignments) == 1
    assert anchored_dummy_group_assignments[0] == {1: (2, frozenset({0}))}

    # Test that providing an empty core and a None bond anchor results in a None angle anchor
    anchored_dummy_group_assignments = generate_anchored_dummy_group_assignments(
        {None: {0, 1, 2, 3, 4}}, g_a, g_b, core_atoms_a=[], core_atoms_b=[]
    )

    anchored_dummy_group_assignments = list(anchored_dummy_group_assignments)

    assert anchored_dummy_group_assignments == [{None: (None, frozenset({0, 1, 2, 3, 4}))}]


def test_multiple_none_dummy_groups():
    # Test that if we break a core-core bond, we only have one valid
    # choice of the angle anchor
    #
    # mol_a: H-O-H.H-O-H
    # mol_b: H-O-H.H-O-H
    g_a = convert_to_nx(Chem.AddHs(Chem.MolFromSmiles("O.O")))
    g_b = convert_to_nx(Chem.AddHs(Chem.MolFromSmiles("O.O")))
    core_a = []
    core_b = []

    dgas = list(generate_dummy_group_assignments(g_a, core_a, assert_single_connected_component=False))
    expected_dga = {None: {0, 1, 2, 3, 4, 5}}

    assert equivalent_assignment(dgas, [expected_dga])

    # Test that providing an empty core and a None bond anchor results in a None angle anchor
    anchored_dummy_group_assignments = generate_anchored_dummy_group_assignments(
        {None: {0, 1, 2, 3, 4, 5}}, g_a, g_b, core_a, core_b
    )

    anchored_dummy_group_assignments = list(anchored_dummy_group_assignments)

    assert anchored_dummy_group_assignments == [{None: (None, frozenset({0, 1, 2, 3, 4, 5}))}]
