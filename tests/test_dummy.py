import pytest
from rdkit import Chem

from timemachine.fe.dummy import identify_dummy_groups, identify_root_anchors

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


def get_bond_idxs(mol):
    # not necessarily canonicalized!
    idxs = []
    for bond in mol.GetBonds():
        idxs.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return idxs


def test_identify_root_anchors():
    """
    Test the identification of root anchors given a dummy atom.

    For example, if D were the dummy atom below:

            D---1---2
           /
          0

    its root anchors are the set of atoms {0, 1}

    """
    mol = get_bond_idxs(Chem.MolFromSmiles("C1CCC1N"))
    core = [0, 1, 2, 3]
    anchors = identify_root_anchors(mol, core, dummy_atom=4)
    assert set(anchors) == set([3])

    mol = get_bond_idxs(Chem.MolFromSmiles("C1CC2NC2C1"))
    core = [0, 1, 2, 4, 5]
    anchors = identify_root_anchors(mol, core, dummy_atom=3)
    assert set(anchors) == set([2, 4])

    mol = get_bond_idxs(Chem.MolFromSmiles("C1OCC11CCCCC1"))
    core = [3, 4, 5, 6, 7, 8]
    anchors = identify_root_anchors(mol, core, dummy_atom=0)
    assert set(anchors) == set([3])
    anchors = identify_root_anchors(mol, core, dummy_atom=1)
    assert set(anchors) == set([3])
    anchors = identify_root_anchors(mol, core, dummy_atom=2)
    assert set(anchors) == set([3])

    mol = get_bond_idxs(Chem.MolFromSmiles("C1CC1.C1CCCCC1"))
    core = [3, 4, 5, 6, 7, 8]
    anchors = identify_root_anchors(mol, core, dummy_atom=0)
    assert set(anchors) == set()
    anchors = identify_root_anchors(mol, core, dummy_atom=1)
    assert set(anchors) == set()
    anchors = identify_root_anchors(mol, core, dummy_atom=2)
    assert set(anchors) == set()

    mol = get_bond_idxs(Chem.MolFromSmiles("C1CC2NC2C1"))
    core = [0, 1, 2, 5]
    anchors = identify_root_anchors(mol, core, dummy_atom=3)
    assert set(anchors) == set([2, 5])
    anchors = identify_root_anchors(mol, core, dummy_atom=4)
    assert set(anchors) == set([2, 5])

    # cyclohexane with a nitrogen inside
    mol = get_bond_idxs(Chem.MolFromSmiles("C1C2CC3CC1N23"))
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
    bond_idxs = get_bond_idxs(Chem.MolFromSmiles("FC1CC1(F)N"))
    core = [1, 2, 3]
    dg = identify_dummy_groups(bond_idxs, core)
    assert_set_equality(dg, [{0}, {4, 5}])

    bond_idxs = get_bond_idxs(Chem.MolFromSmiles("FC1CC1(F)NN"))
    core = [1, 2, 3]
    dg = identify_dummy_groups(bond_idxs, core)
    assert_set_equality(dg, [{0}, {4, 5, 6}])

    bond_idxs = get_bond_idxs(Chem.MolFromSmiles("C1CC11OO1"))
    core = [0, 1, 2]
    dg = identify_dummy_groups(bond_idxs, core)
    assert_set_equality(dg, [{3, 4}])

    bond_idxs = get_bond_idxs(Chem.MolFromSmiles("C1CC2OOC12"))
    core = [0, 1, 2, 5]
    dg = identify_dummy_groups(bond_idxs, core)
    assert_set_equality(dg, [{3, 4}])

    # example above, where O's are dummy atoms, and Cs are core
    bond_idxs = get_bond_idxs(Chem.MolFromSmiles("OC1COO1"))
    core = [1, 2]
    dg = identify_dummy_groups(bond_idxs, core)
    assert_set_equality(dg, [{0, 3, 4}])


def assert_anchor_group_equality(a_groups, b_groups):
    frozen_a = [tuple(a) for a in a_groups]
    frozen_b = [tuple(b) for b in b_groups]
    assert frozenset(frozen_a) == frozenset(frozen_b)
