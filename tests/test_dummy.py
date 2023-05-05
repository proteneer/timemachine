import pytest
from rdkit import Chem

from timemachine.fe.dummy import convert_bond_list_to_nx, generate_dummy_group_assignments
from timemachine.fe.utils import get_romol_bonds

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

pytestmark = [pytest.mark.nogpu]


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

    def equivalent_assignment(left, right):
        def to_comparable(dgas):
            return frozenset(frozenset((k, frozenset(v)) for k, v in dgs.items()) for dgs in dgas)

        return to_comparable(left) == to_comparable(right)

    g = convert_bond_list_to_nx(get_romol_bonds(Chem.MolFromSmiles("FC1CC1(F)N")))
    core = [1, 2, 3]
    dgas = list(generate_dummy_group_assignments(g, core))
    assert equivalent_assignment(dgas, [{1: {0}, 3: {4, 5}}])

    g = convert_bond_list_to_nx(get_romol_bonds(Chem.MolFromSmiles("FC1CC1(F)NN")))
    core = [1, 2, 3]
    dgas = list(generate_dummy_group_assignments(g, core))
    assert equivalent_assignment(dgas, [{1: {0}, 3: {4, 5, 6}}])

    g = convert_bond_list_to_nx(get_romol_bonds(Chem.MolFromSmiles("C1CC11OO1")))
    core = [0, 1, 2]
    dgas = list(generate_dummy_group_assignments(g, core))
    assert equivalent_assignment(dgas, [{2: {3, 4}}])

    g = convert_bond_list_to_nx(get_romol_bonds(Chem.MolFromSmiles("C1CC2OOC12")))
    core = [0, 1, 2, 5]
    dgas = list(generate_dummy_group_assignments(g, core))
    assert equivalent_assignment(dgas, [{2: {3, 4}}, {5: {3, 4}}])

    # example above, where O's are dummy atoms, and Cs are core
    g = convert_bond_list_to_nx(get_romol_bonds(Chem.MolFromSmiles("OC1COO1")))
    core = [1, 2]
    dgas = list(generate_dummy_group_assignments(g, core))
    # one or two groups depending on choice of anchor atom for {3, 4}
    assert equivalent_assignment(dgas, [{1: {0}, 2: {3, 4}}, {1: {0, 3, 4}}])
