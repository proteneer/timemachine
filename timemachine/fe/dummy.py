import copy
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np

from timemachine.graph_utils import convert_to_nx


def _add_successors(mol, core, groups):
    # (ytz): internal utility used for bfs
    next_groups = []
    for group in groups:
        # grab the last node
        last_node = group[-1]
        for nbr in mol.GetAtomWithIdx(last_node).GetNeighbors():
            new_group = copy.deepcopy(group)
            nbr = nbr.GetIdx()
            # ensure is core_atom and not already visited
            if nbr in core and nbr not in new_group:
                new_group.append(nbr)
                next_groups.append(new_group)

    return next_groups


def identify_anchor_groups(mol, core, root_anchor):
    """
    Generate all choices for valid anchor groups. An anchor group
    is an ordered sequence of core atoms (root_anchor,b,c) that are connected by bonds.
    ie. an anchor group is a rooted subtree with three nodes spanning from the root_anchor.

    Parameters
    ----------
    mol: Chem.Mol
        rdkit molecule

    core: list or set or iterable of ints
        core atoms

    root_anchor: int
        core atom we're initializing the search over

    Returns
    -------
    3-tuple
        Returns anchor groups of size 1, 2, and 3. Size 1 group will always
        be [[root_anchor]]. Size 2 groups will be [[root_anchor, x], ...] and size 3
        groups will be [[root_anchor,x,y,], ...]

    """

    assert root_anchor in core

    # We perform a bfs of depth 3 starting from the root_anchor.
    layer_1_groups = [[root_anchor]]
    layer_2_groups = _add_successors(mol, core, layer_1_groups)
    layer_3_groups = _add_successors(mol, core, layer_2_groups)

    # sanity assertions to make sure we don't have duplicate idxs
    for g in layer_2_groups:
        assert len(set(g)) == len(g)

    for g in layer_3_groups:
        assert len(set(g)) == len(g)

    return layer_1_groups, layer_2_groups, layer_3_groups


def identify_root_anchors(mol, core, dummy_atom):
    """
    Identify the root anchor(s) for a given atom. A root anchor is defined as the starting atom
    in an anchor group (comprised of up to 3 anchor atoms). If this returns multiple root anchors
    then the dummy_atom is a bridge between multiple core atoms.

    Parameters
    ----------
    mol: Chem.Mol
        rdkit molecule

    core: list or set or iterable
        core atoms

    dummy_atom: int
        atom we're initializing the search over

    Returns
    -------
    list of int
        List of root anchors that the dummy atom is connected to.

    """

    assert len(set(core)) == len(core)

    core = set(core)

    assert dummy_atom not in core

    # first convert to a dense graph
    N = mol.GetNumAtoms()
    dense_graph = np.zeros((N, N), dtype=np.int32)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dense_graph[i, j] = 1
        dense_graph[j, i] = 1

    # sparsify to simplify and speed up traversal code
    sparse_graph = []
    for row in dense_graph:
        nbs = []
        for col_idx, col in enumerate(row):
            if col == 1:
                nbs.append(col_idx)
        sparse_graph.append(nbs)

    def dfs(i, visited):
        if i in visited:
            return
        else:
            visited.add(i)
            if i not in core:
                for nb in sparse_graph[i]:
                    dfs(nb, visited)

    visited = set()

    dfs(dummy_atom, visited)

    anchors = [a_idx for a_idx in visited if a_idx in core]

    return anchors


def enumerate_anchor_groups(mol, core, dummy_group):
    """
    An anchor group is an ordered set of only core atoms that are allowed to interact with
    atoms in a dummy group in a way that allows the partition function to be separated.
    Unlike dummy groups, anchor groups do not form disjoint partitions. Different anchor
    groups are allowed to have overlap in core atoms used. In other words, anchor groups
    for any particular dummy group is fully independent of other anchor groups for any
    other dummy group.

    While the choices of atoms in an anchor group is arbitrary, one heuristic we use
    to enumerate anchor groups is as follows. In order to determine:

    1) the 1-2 bond anchor i, we find a core atom that is directly bonded a dummy atom
    2) the 1-3 angle anchor j, we pick a core atom that is adjacent to i
    3) the 1-4 torsion anchor k, we pick a core atom adjacent to j, but not equal to i.

    Parameters
    ----------
    mol: Chem.Mol
        Molecule of interest

    core: list of int
        core atoms

    dummy_group: list of int
        dummy atoms in a given group.

    Returns
    -------
    tuple
        Anchor groups of size 1, 2, and 3 atoms

    """

    anchors = set()

    for dummy_atom in dummy_group:
        for anchor_atom in identify_root_anchors(mol, core, dummy_atom):
            anchors.add(anchor_atom)

    l1s = []
    l2s = []
    l3s = []

    for anchor in anchors:
        l1, l2, l3 = identify_anchor_groups(mol, core, anchor)
        l1s.extend(l1)
        l2s.extend(l2)
        l3s.extend(l3)

    return l1s, l2s, l3s


def identify_dummy_groups(mol, core):
    """
    A dummy group is a set of dummy atoms that are inserted or deleted in alchemical
    free energy calculations. The bonded terms that involve dummy atoms need to be
    judiciously pruned such that the partition function at the end-states remain
    factorizable, and cancelleable. Dummy groups are subject to the following constraints:

    1) They must not contain atoms in the core.
    2) Dummy groups do not interact with other dummy groups.
    3) Dummy groups interact with the core only through anchor atoms.

    While the choices of dummy groups is arbitrary, this function partitions the dummy
    atoms into multiple dummy groups using a heuristic:

    1) Identify all 1-2, 1-3 edges between dummy atoms.
    2) Generate an induced graph using edges from above set.
    3) Disconnected components of the resulting induced graph are our dummy groups.

    Code example:

    ```
    mol = Chem.MolFromSmiles("FC1CC1(F)N")
    core = [1, 2, 3]
    dg = identify_dummy_groups(mol, core)
    assert_set_equality(dg, [{0}, {4, 5}])
    ```

    Returns
    -------
    List of set of ints:
        eg: [{3,4}, {7,8,9}]

    """

    g = convert_to_nx(mol)
    N = mol.GetNumAtoms()
    induced_g = nx.Graph()

    # add nodes and edges into the induced graph.
    for i in range(N):
        if i not in core:
            induced_g.add_node(i)
            for j in range(i + 1, N):
                if j not in core:
                    dist = nx.shortest_path_length(g, source=i, target=j)
                    if dist <= 2:
                        induced_g.add_edge(i, j)

    cc = list(nx.connected_components(induced_g))

    # check that we didn't miss any dummy atoms
    assert np.sum([len(c) for c in cc]) == (N - len(core))

    # refine based on rooted-anchor, this lets us deal with ring opening and closing:
    #
    #        D----D
    #        |    |
    #        C----C
    #        |    |
    #        C----C
    #
    # in this case, we want the dummy atoms to be in different groups.

    dummy_groups = []

    for dg in cc:
        anchor_membership = []
        dg = list(dg)
        for dummy in dg:
            # identify membership of the dummy
            anchors = identify_root_anchors(mol, core, dummy)
            # get distance to anchor
            dists = []
            for a in anchors:
                dists.append(nx.shortest_path_length(g, source=dummy, target=a))
            anchor_membership.append(np.argmin(dists))

        anchor_kv = defaultdict(set)
        for idx, anchor in enumerate(anchor_membership):
            anchor_kv[anchor].add(dg[idx])

        for v in anchor_kv.values():
            dummy_groups.append(v)

    return dummy_groups


def enumerate_dummy_ixns(dg, ag):
    """
    Enumerate the allowed set of interactions between
    1) atoms within the dummy group
    2) atoms in a dummy group and atoms in anchor group

    Parameters
    ----------
    dg: set of int
        dummy atoms in a dummy group

    ag: list of int
        ordered core anchor atoms

    Returns
    -------
    set
        Set of interactions containing, 1-2, 1-3, and 1-4 interactions
        The bonds returned are ordered.

    """

    allowed_ixns = set()

    nc1 = list(itertools.combinations(dg, 1))
    nc2 = list(itertools.combinations(dg, 2))
    nc3 = list(itertools.combinations(dg, 3))
    nc4 = list(itertools.combinations(dg, 4))

    # enumerate dummy-dummy ixns
    for ijkl in nc4:
        for i, j, k, l in itertools.permutations(ijkl):
            allowed_ixns.add((i, j, k, l))
    for ijk in nc3:
        for i, j, k in itertools.permutations(ijk):
            allowed_ixns.add((i, j, k))
    for ij in nc2:
        for i, j in itertools.permutations(ij):
            allowed_ixns.add((i, j))

    # enumerate dummy-anchor ixns
    if len(ag) > 0:
        a = ag[0]
        for ijk in nc3:
            for i, j, k in itertools.permutations(ijk):
                allowed_ixns.add((a, i, j, k))
                allowed_ixns.add((i, a, j, k))
                allowed_ixns.add((i, j, a, k))
                allowed_ixns.add((i, j, k, a))
        for ij in nc2:
            for i, j in itertools.permutations(ij):
                allowed_ixns.add((a, i, j))
                allowed_ixns.add((i, a, j))
                allowed_ixns.add((i, j, a))

        for (i,) in nc1:
            allowed_ixns.add((i, a))

        if len(ag) > 1:
            b = ag[1]
            for ij in nc2:
                for i, j in itertools.permutations(ij):
                    allowed_ixns.add((i, j, a, b))

            for (i,) in nc1:
                allowed_ixns.add((i, a, b))

            if len(ag) > 2:
                c = ag[2]
                for (i,) in nc1:
                    allowed_ixns.add((i, a, b, c))

    return make_bond_set(allowed_ixns)


def find_ags_for_dg(mol, core, dg):
    """
    Find all possible anchor groups for a given dummy group.

    Parameters
    ----------
    mol: Chem.Mol
        input molecule

    core: list of int
        core indices

    dg: iterable of int
        dummy group

    Returns
    -------
    list of list of int of up to size 3
        Returns all allowed anchor groups for a given dummy group.

    """
    anchors = []
    for dummy_atom in dg:
        anchors.extend(identify_root_anchors(mol, core, dummy_atom))
    anchors = set(anchors)

    anchor_groups = []
    for anchor in anchors:
        ag0, ag1, ag2 = identify_anchor_groups(mol, core, anchor)
        anchor_groups.extend(ag0)
        anchor_groups.extend(ag1)
        anchor_groups.extend(ag2)

    return anchor_groups


def ordered_tuple(ixn):
    if ixn[0] > ixn[-1]:
        return tuple(ixn[::-1])
    else:
        return tuple(ixn)


def make_bond_set(old_set):
    """
    Bond set has the requirement that indices are symmetrically
    reversible. i.e. ij = ji, ijk = kji, ijkl = lkji

    This is required to avoid duplication of bonds.
    """
    new_set = set()
    for idxs in old_set:
        new_set.add(ordered_tuple(idxs))
    return new_set


def flag_bonds(mol, core, bond_idxs):
    """
    Flags bonds based on minimizing the number of terms we have to
    turn off.

    Parameters
    ----------
    mol: Chem.Mol
        rdkit molecule

    core: list of int
        idxs of core atoms

    bond_idxs: list of list of int
        list of 2-tuple, 3-tuple, 4-tuple

    Returns
    -------
    boolean flags of len(bond_idxs)
        1: keep, 0: remove

    """

    # 1. process core bonds
    keep_flags = np.zeros(len(bond_idxs), dtype=np.int32)

    dummy_ixns = set()
    for b_idx, atom_idxs in enumerate(bond_idxs):
        if np.all([i in core for i in atom_idxs]):
            keep_flags[b_idx] = 1
        else:
            dummy_ixns.add(tuple(atom_idxs))

    dummy_ixns = make_bond_set(dummy_ixns)

    # 2. process dummy bonds
    dgs, ags, ag_ixns = generate_optimal_dg_ag_pairs(mol, core, bond_idxs)

    allowed_ixns = set()

    for ixns in ag_ixns:
        allowed_ixns |= ixns

    for b_idx, atom_idxs in enumerate(bond_idxs):
        if tuple(ordered_tuple(atom_idxs)) in allowed_ixns:
            keep_flags[b_idx] = 1

    return keep_flags


def generate_dg_ag_pairs(mol, core, bond_idxs):
    """
    Generate all pairings of dummy group and anchor group atoms
    such that bond_idxs is maximized.

    Parameters
    ----------
    mol: Chem.Mol
        rdkit molecule

    core: list of int
        idxs of core atoms

    bond_idxs: list of list of int
        list of 2-tuple, 3-tuple, 4-tuple used to compare

    Returns
    -------
        list of (dummy_group, anchor_group, anchor_group_ixns) triples

    """
    # 1. process core bonds
    ff_core_ixns = set()  # ff interactions that involve *only* core atoms
    ff_dummy_ixns = set()  # ff interactions that involve *any* dummy atom
    for atom_idxs in bond_idxs:
        if np.all([i in core for i in atom_idxs]):
            ff_core_ixns.add(tuple(atom_idxs))
        else:
            ff_dummy_ixns.add(tuple(atom_idxs))

    ff_core_ixns = make_bond_set(ff_core_ixns)
    ff_dummy_ixns = make_bond_set(ff_dummy_ixns)

    # 2. process dummy bonds
    dgs = identify_dummy_groups(mol, core)

    all_agcs = []
    all_agis = []

    for dg in dgs:
        anchor_group_candidates = find_ags_for_dg(mol, core, dg)
        anchor_group_ixns = []
        # enumerate over all interactions
        for ag in anchor_group_candidates:
            # set of all possible dummy interactions, without taking ff idxs into account
            allowed_dummy_ixns = enumerate_dummy_ixns(dg, ag)
            mutual_bonds = allowed_dummy_ixns.intersection(ff_dummy_ixns)
            mutual_bonds = mutual_bonds.union(ff_core_ixns)

            # do additional pruning by checking given idxs
            bonds_12 = set()
            bonds_13 = set()
            bonds_14 = set()
            for idxs in mutual_bonds:
                if len(idxs) == 2:
                    bonds_12.add(idxs)
            for idxs in mutual_bonds:
                if len(idxs) == 3:
                    i, j, k = idxs
                    if (ordered_tuple((i, j)) in bonds_12) and (ordered_tuple((j, k)) in bonds_12):
                        bonds_13.add(idxs)
            for idxs in mutual_bonds:
                if len(idxs) == 4:
                    i, j, k, l = idxs
                    if (
                        (ordered_tuple((i, j)) in bonds_12)
                        and (ordered_tuple((j, k)) in bonds_12)
                        and (ordered_tuple((k, l)) in bonds_12)
                        and (ordered_tuple((i, j, k)) in bonds_13)
                        and (ordered_tuple((j, k, l)) in bonds_13)
                    ):
                        bonds_14.add(idxs)

            mutual_bonds = bonds_12.union(bonds_13).union(bonds_14)
            anchor_group_ixns.append(mutual_bonds)

        all_agcs.append(anchor_group_candidates)
        all_agis.append(anchor_group_ixns)

    return dgs, all_agcs, all_agis


def generate_optimal_dg_ag_pairs(mol, core, bond_idxs):
    """
    Generate optimal (dummy group, anchor group) pairs given a list of bonded terms.

    The dummy groups are generated from dummy atoms defined as non-core atoms. The heuristic
    used attempts to maximize the number of 1-2 terms that can be left on, then the number
    of 1-3 terms that can be left on, and finally the number of 1-4 terms that can be left on.

    Parameters
    ----------
    mol: Chem.Mol
        Input molecule

    core: list of int
        Indices for the core atom

    bond_idxs: list of list of ints
        Input bond_idxs

    Returns
    -------
    3-tuple of dummy_groups, best_anchor_groups, best_anchor_group_ixns
        Best anchor group for each dummy group and its interactions are returned

    """
    dgs, all_agcs, all_agis = generate_dg_ag_pairs(mol, core, bond_idxs)

    picked_agcs = []
    picked_agis = []

    for agcs, agis in zip(all_agcs, all_agis):

        counts = []
        for mutual_bonds in agis:
            bond_count = np.sum([len(idxs) == 2 for idxs in mutual_bonds])
            angle_count = np.sum([len(idxs) == 3 for idxs in mutual_bonds])
            torsion_count = np.sum([len(idxs) == 4 for idxs in mutual_bonds])
            counts.append((bond_count, angle_count, torsion_count))

        best_idx = max(zip(range(len(counts)), counts), key=lambda x: x[1])[0]
        picked_agcs.append(agcs[best_idx])
        picked_agis.append(agis[best_idx])

    return dgs, picked_agcs, picked_agis
