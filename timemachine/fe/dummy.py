import warnings
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Sequence
from itertools import product
from typing import Optional, TypeVar

import networkx as nx


class MultipleAnchorWarning(UserWarning):
    pass


def generate_dummy_group_assignments(
    bond_graph: nx.Graph, core_atoms: Collection[int]
) -> Iterator[dict[int, frozenset[int]]]:
    """Returns an iterator over dummy group assignments (i.e., candidate partitionings of dummy atoms with each
    partition assigned a bond anchor atom) for a given molecule (represented as a bond graph) and set of core atoms.

    A dummy group is a set of dummy atoms that are inserted or deleted in alchemical free energy calculations. The
    bonded terms that involve dummy atoms need to be judiciously pruned such that the partition function at the
    end-states remain factorizable, and cancelleable. Dummy groups are subject to the following constraints:

    1) They must not contain atoms in the core.
    2) Dummy groups do not interact with other dummy groups.
    3) Dummy groups interact with the core only through anchor atoms.

    While the choices of dummy groups is arbitrary, this function partitions the dummy atoms into multiple dummy groups
    using a heuristic:

    1) Generate an induced subgraph containing only dummy atoms
    2) Identify connected components of the induced subgraph (see note 1 below).
    3) Generate all possible mappings (bond anchor, connected component). In general there will be multiple
       possibilities since there may be an arbitrary choice of anchor atom for each dummy group
    4) For each mapping ("assignment"), merge connected components with the same bond anchor

    Example
    -------
    >>> from rdkit import Chem
    >>> from timemachine.graph_utils import convert_to_nx
    >>> mol = Chem.MolFromSmiles("OC1COO1")
    >>> bond_graph = convert_to_nx(mol)
    >>> core = [1, 2]
    >>> list(generate_dummy_group_assignments(bond_graph, core))
    [{1: frozenset({0}), 2: frozenset({3, 4})}, {1: frozenset({0, 3, 4})}]

    Parameters
    ----------
    bond_graph: networkx.Graph
        graph with atoms as nodes and bonds as edges

    core_atoms: list of int
        atoms in the core

    Returns
    -------
    iterator of dict
        each element is a mapping from bond anchor atom to dummy group

    Notes
    -----
    1. The final dummy groups are constructed as unions of one or more of the connected components of the dummy induced
       subgraph; in particular, we can never end up with a dummy group that is a subset of a connected component (the
       latter might be desirable for efficiency but is more complicated to implement).
    """
    assert len(set(core_atoms)) == len(core_atoms)

    # weakened for host_guest_system due to presence of disconnected components
    # assert len(list(nx.connected_components(bond_graph))) == 1

    core_atoms_ = frozenset(core_atoms)
    dummy_atoms = frozenset(bond_graph.nodes()) - core_atoms_
    induced_g = nx.subgraph(bond_graph, dummy_atoms)

    def get_bond_anchors(dummy_group):
        bond_anchors = {n for dummy_atom in dummy_group for n in bond_graph.neighbors(dummy_atom) if n in core_atoms}
        if len(bond_anchors) > 1:
            warnings.warn(
                f"Multiple bond anchors {bond_anchors} found for dummy group: {dummy_group}",
                MultipleAnchorWarning,
            )
        return bond_anchors

    dummy_group_assignments = (
        union_by_key(bond_anchor_cc_pairs)
        for bond_anchor_cc_pairs in product(
            *[[(bond_anchor, cc) for bond_anchor in get_bond_anchors(cc)] for cc in nx.connected_components(induced_g)]
        )
    )

    return dummy_group_assignments


def generate_anchored_dummy_group_assignments(
    dummy_groups: dict[int, frozenset[int]],
    bond_graph_a: nx.Graph,
    bond_graph_b: nx.Graph,
    core_atoms_a: Sequence[int],
    core_atoms_b: Sequence[int],
) -> Iterator[dict[int, tuple[Optional[int], frozenset[int]]]]:
    """Returns an iterator over candidate anchored dummy group assignments.

    By convention, dummy atoms are added to A to transform it into a supergraph of B. Indices in the dummy_groups
    argument and in the return value refer to atoms in B.

    An anchored dummy group assignment is a set of triples (dummy group, j = bond anchor atom, k = angle anchor atom),
    where dummy atoms are connected to the core only through j, and k is a core neighbor of j where the bond (j, k)
    exists in both mol_a and mol_b. Note that k may be missing (None) if there are no valid choices.

    See documentation for :py:func:`timemachine.fe.dummy.generate_dummy_group_assignments` for information on how
    candidate dummy group assignments are generated.

    Example
    -------
    >>> from rdkit import Chem
    >>> from timemachine.graph_utils import convert_to_nx
    >>> bond_graph_a = convert_to_nx(Chem.MolFromSmiles("CC"))
    >>> bond_graph_b = convert_to_nx(Chem.MolFromSmiles("c1(C)ccc1"))
    >>> core_atoms_a = [0, 1]
    >>> core_atoms_b = [2, 0]
    >>> dgas = list(generate_dummy_group_assignments(bond_graph_b, core_atoms_b))
    >>> [adgs for dgs in dgas for adgs in generate_anchored_dummy_group_assignments(dgs, bond_graph_a, bond_graph_b, core_atoms_a, core_atoms_b)]
    [{0: (2, frozenset({1, 3, 4}))}, {0: (2, frozenset({1})), 2: (0, frozenset({3, 4}))}]

    Parameters
    ----------
    dummy_groups: dict[int, frozenset[int]]
        Mapping from anchor atom to atoms in the associated dummy group. Indices refer to atoms in B.

    bond_graph_a, bond_graph_b: nx.Graph
        Bond graphs for source and target molecules of an alchemical transformation.
        Dummy atoms are added to A to transform it into a supergraph of B.

    core_atoms_a, core_atoms_b: sequence of int
        atoms in the core

    Returns
    -------
    iterator of dict
        each element is a mapping from bond anchor atom to the pair (angle anchor atom, dummy group)
    """

    core_bonds_c = get_core_bonds(bond_graph_a.edges(), bond_graph_b.edges(), core_atoms_a, core_atoms_b)
    c_to_b = {c: b for c, b in enumerate(core_atoms_b)}
    core_bonds_b = frozenset(translate_bonds(core_bonds_c, c_to_b))

    def get_angle_anchors(bond_anchor):
        valid_angle_anchors = [
            angle_anchor
            for angle_anchor in [n for n in bond_graph_b.neighbors(bond_anchor) if n in core_atoms_b]
            if canonicalize_bond((bond_anchor, angle_anchor)) in core_bonds_b
        ]
        return valid_angle_anchors or [None]

    # For each dummy group assignment, generate (possibly multiple) anchored dummy group assignments each corresponding
    # to an independent choice of an angle anchor atom for each dummy group
    anchored_dummy_group_assignments = (
        dict(anchored_dummy_group)
        for anchored_dummy_group in product(
            *[
                [(bond_anchor, (angle_anchor, dummy_group)) for angle_anchor in get_angle_anchors(bond_anchor)]
                for bond_anchor, dummy_group in dummy_groups.items()
            ]
        )
    )

    return anchored_dummy_group_assignments


def canonicalize_bond(ixn: tuple[int, ...]) -> tuple[int, ...]:
    if ixn[0] > ixn[-1]:
        return tuple(ixn[::-1])
    else:
        return tuple(ixn)


def get_core_bonds(
    bonds_a: Collection[tuple[int, int]],
    bonds_b: Collection[tuple[int, int]],
    core_atoms_a: Sequence[int],
    core_atoms_b: Sequence[int],
) -> frozenset[tuple[int, ...]]:
    """Returns core-core bonds that are present in both mol_a and mol_b"""
    a_to_c = {a: c for c, a in enumerate(core_atoms_a)}
    b_to_c = {b: c for c, b in enumerate(core_atoms_b)}
    return frozenset(translate_bonds(bonds_a, a_to_c)).intersection(frozenset(translate_bonds(bonds_b, b_to_c)))


def translate_bonds(bonds: Collection[tuple[int, ...]], mapping: dict[int, int]) -> list[tuple[int, ...]]:
    """Applies the given mapping of atom indices to a collection of bonds (i.e. tuples of atom indices)

    Bonds containing indices that are missing from the input mapping are omitted in the output."""
    return [
        canonicalize_bond(tuple(mapping[idx] for idx in bond)) for bond in bonds if all(idx in mapping for idx in bond)
    ]


_K = TypeVar("_K")
_V = TypeVar("_V")


def union_by_key(ts: Iterable[tuple[_K, frozenset[_V]]]) -> dict[_K, frozenset[_V]]:
    """Given an iterable of key-value pairs where the values are sets, returns a dictionary of sets merged by key."""
    d: defaultdict[_K, frozenset[_V]] = defaultdict(frozenset)
    for k, xs in ts:
        d[k] = d[k].union(xs)
    return dict(d)
