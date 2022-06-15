import copy
import itertools

import networkx as nx
import numpy as np


def convert_bond_list_to_nx(bond_list):
    """
    Convert an ROMol into a networkx graph.
    """
    g = nx.Graph()
    for i, j in bond_list:
        assert i != j  # do not allow self-edges
        g.add_edge(i, j)

    return g


def _add_successors(bond_idxs, core, groups):

    g = convert_bond_list_to_nx(bond_idxs)
    # (ytz): internal utility used for bfs
    next_groups = []
    for group in groups:
        # grab the last node
        last_node = group[-1]
        for nbr in g.neighbors(last_node):
            new_group = copy.deepcopy(group)
            # ensure is core_atom and not already visited
            if nbr in core and nbr not in new_group:
                new_group.append(nbr)
                next_groups.append(new_group)

    return next_groups


def identify_anchor_groups(bond_idxs, core, root_anchor):
    """
    Generate all choices for valid anchor groups. An anchor group
    is an ordered sequence of core atoms (root_anchor,b,c) that are connected by bonds.
    ie. an anchor group is a rooted subtree with three nodes spanning from the root_anchor.

    Parameters
    ----------
    bond_idxs: list of 2-tuples of ints
        list of (i,j) bonds denoting the atoms in the bond

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
    layer_2_groups = _add_successors(bond_idxs, core, layer_1_groups)
    layer_3_groups = _add_successors(bond_idxs, core, layer_2_groups)

    # sanity assertions to make sure we don't have duplicate idxs
    for g in layer_2_groups:
        assert len(set(g)) == len(g)

    for g in layer_3_groups:
        assert len(set(g)) == len(g)

    return layer_1_groups, layer_2_groups, layer_3_groups


def identify_root_anchors(bond_idxs, core, dummy_atom):
    """
    Identify the root anchor(s) for a given atom. A root anchor is defined as the starting atom
    in an anchor group (comprised of up to 3 anchor atoms). If this returns multiple root anchors
    then the dummy_atom is a bridge between multiple core atoms.

    Parameters
    ----------
    bond_idxs: list of 2-tuples of ints
        list of (i,j) bonds denoting the atoms in the bond

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

    # first convert to a dense graph, and assume that bond_idxs start from zero
    flat_idxs = np.array(bond_idxs).flatten()
    N = np.amax(flat_idxs) + 1
    assert N == len(set(flat_idxs))
    dense_graph = np.zeros((N, N), dtype=np.int32)

    for i, j in bond_idxs:
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

    # conditional depth first search that terminates when
    # we encounter a core atom.
    def conditional_dfs(i, visited):
        if i in visited:
            return
        else:
            visited.add(i)
            if i not in core:
                for nb in sparse_graph[i]:
                    conditional_dfs(nb, visited)

    visited = set()

    conditional_dfs(dummy_atom, visited)

    anchors = [a_idx for a_idx in visited if a_idx in core]

    return anchors


def enumerate_anchor_groups(bond_idxs, core, dummy_group):
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
    bond_idxs: list of 2-tuples of ints
        list of (i,j) bonds denoting the atoms in the bond

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
        for anchor_atom in identify_root_anchors(bond_idxs, core, dummy_atom):
            anchors.add(anchor_atom)

    l1s = []
    l2s = []
    l3s = []

    for anchor in anchors:
        l1, l2, l3 = identify_anchor_groups(bond_idxs, core, anchor)
        l1s.extend(l1)
        l2s.extend(l2)
        l3s.extend(l3)

    return l1s, l2s, l3s


def identify_dummy_groups(bond_idxs, core):
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

    Parameters
    ----------
    bond_idxs: list of 2-tuples of ints
        list of (i,j) bonds denoting the atoms in the bond

    core: list of int
        atoms in the core

    Returns
    -------
    List of set of ints:
        eg: [{3,4}, {7,8,9}]

    """
    g = convert_bond_list_to_nx(bond_idxs)
    N = g.number_of_nodes()
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

    return cc


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


def find_ags_for_dg(bond_idxs, core, dg):
    """
    Find all possible anchor groups for a given dummy group.

    Parameters
    ----------
    bond_idxs: list of 2-tuples of ints
        list of (i,j) bonds denoting the atoms in the bond

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
        anchors.extend(identify_root_anchors(bond_idxs, core, dummy_atom))
    anchors = set(anchors)

    anchor_groups = []
    for anchor in anchors:
        ag0, ag1, ag2 = identify_anchor_groups(bond_idxs, core, anchor)
        anchor_groups.extend(ag0)
        anchor_groups.extend(ag1)
        anchor_groups.extend(ag2)

    return anchor_groups


def canonicalize_bond(ixn):
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
        new_set.add(canonicalize_bond(idxs))
    return new_set


def flag_factorizable_bonds(core, valence_idxs):
    """
    Flags bonds based on minimizing the number of terms we have to
    turn off.

    Parameters
    ----------
    core: list of int
        idxs of core atoms

    valence_idxs: list of list of int
        list of 2-tuple, 3-tuple, 4-tuple

    Returns
    -------
    boolean flags of len(valence_idxs)
        1: keep, 0: remove

    """

    # REDUNDANT - since valence_idxs already encode "mol"

    # 1. process core bonds
    keep_flags = np.zeros(len(valence_idxs), dtype=np.int32)

    dummy_ixns = set()
    for b_idx, atom_idxs in enumerate(valence_idxs):
        if np.all([i in core for i in atom_idxs]):
            keep_flags[b_idx] = 1
        else:
            dummy_ixns.add(tuple(atom_idxs))

    dummy_ixns = make_bond_set(dummy_ixns)

    # 2. process dummy bonds
    dgs, ags, ag_ixns = generate_optimal_dg_ag_pairs(core, valence_idxs)

    allowed_ixns = set()

    for ixns in ag_ixns:
        allowed_ixns |= ixns

    for b_idx, atom_idxs in enumerate(valence_idxs):
        if tuple(canonicalize_bond(atom_idxs)) in allowed_ixns:
            keep_flags[b_idx] = 1

    return keep_flags


class MissingBondError(Exception):
    def __init__(self, bad_atom):
        message = "atom " + str(bad_atom) + " is disconnected."
        super().__init__(message)


def generate_dg_ag_ixns(core, valence_idxs):
    """
    Generate all pairings of dummy group and anchor group atoms.

    Parameters
    ----------
    core: list of int
        idxs of core atoms

    valence_idxs: list of list of int
        list of 2-tuple, 3-tuple, 4-tuple used to generate

    Returns
    -------
        list of (dummy_group, anchor_group, anchor_group_ixns) triples

    """
    # sanity check that 13, 14 terms don't have extra interactions
    bond_atoms = set()
    for ij in valence_idxs:
        if len(ij) == 2:
            i, j = ij
            bond_atoms.add(i)
            bond_atoms.add(j)

    for atom_idxs in valence_idxs:
        for i in atom_idxs:
            if i not in bond_atoms:
                raise MissingBondError(i)

    # 1. process core bonds
    ff_core_ixns = set()  # ff interactions that involve *only* core atoms
    ff_dummy_ixns = set()  # ff interactions that involve *any* dummy atom
    for atom_idxs in valence_idxs:
        if np.all([i in core for i in atom_idxs]):
            ff_core_ixns.add(tuple(atom_idxs))
        else:
            ff_dummy_ixns.add(tuple(atom_idxs))

    ff_core_ixns = make_bond_set(ff_core_ixns)
    ff_dummy_ixns = make_bond_set(ff_dummy_ixns)

    # 2. process dummy bonds
    bond_12_idxs = [x for x in valence_idxs if len(x) == 2]
    for ij in bond_12_idxs:
        assert len(ij) == 2

    dgs = identify_dummy_groups(bond_12_idxs, core)

    all_agcs = []
    all_agis = []

    for dg in dgs:
        anchor_group_candidates = find_ags_for_dg(bond_12_idxs, core, dg)
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
                    if (canonicalize_bond((i, j)) in bonds_12) and (canonicalize_bond((j, k)) in bonds_12):
                        bonds_13.add(idxs)
            for idxs in mutual_bonds:
                if len(idxs) == 4:
                    i, j, k, l = idxs
                    if (
                        (canonicalize_bond((i, j)) in bonds_12)
                        and (canonicalize_bond((j, k)) in bonds_12)
                        and (canonicalize_bond((k, l)) in bonds_12)
                        and (canonicalize_bond((i, j, k)) in bonds_13)
                        and (canonicalize_bond((j, k, l)) in bonds_13)
                    ):
                        bonds_14.add(idxs)

            mutual_bonds = bonds_12.union(bonds_13).union(bonds_14)
            anchor_group_ixns.append(mutual_bonds)

        all_agcs.append(anchor_group_candidates)
        all_agis.append(anchor_group_ixns)

    return dgs, all_agcs, all_agis


def generate_optimal_dg_ag_pairs(core, valence_idxs):
    """
    Generate optimal (dummy group, anchor group) pairs given a list of bonded terms.

    The dummy groups are generated from dummy atoms defined as non-core atoms. The heuristic
    used attempts to maximize the number of 1-2 terms that can be left on, then the number
    of 1-3 terms that can be left on, and finally the number of 1-4 terms that can be left on.

    Parameters
    ----------
    mol: Chem.Mol
        Input molecule
        Indices for the core atom

    valence_idxs: list of list of int
        list of 2-tuple, 3-tuple, 4-tuple used to generate

    Returns
    -------
    3-tuple of dummy_groups, best_anchor_groups, best_anchor_group_ixns
        Best anchor group for each dummy group and its interactions are returned

    """
    dgs, all_agcs, all_agis = generate_dg_ag_ixns(core, valence_idxs)

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


def flag_stable_dummy_ixns(
    core,
    bond_idxs,
    bond_params,
    angle_idxs,
    angle_params,
    torsion_idxs,
    torsion_params,
    min_bond_k=10.0,
    min_bond_l=0.02,
    min_angle_k=10.0,
    min_angle_offset=0.05,
):
    """
    Prune extraneous dummy-core ixns that may be numerically unstable. Sources of numerical instability typically
    are due to either missing interactions, or incompatible parameters. This does not prune interactions in the
    dummy-dummy regions since they are required for correctness. While core-core interactions can be pruned, we
    choose to leave them untouched for now.

    0) All bond terms are untouched.
    1) All angles i-j-k must have bond terms i-j and j-k present, and with k > 10 and b > 0.2 angstrom on said bonds
    2) All torsions i-j-k-l must have angle terms i-j-k and j-k-l present, and with k > 10 and
        abs(angle - pi) > 0.05 rad and abs(angle - 0) > 0.05. In addition, bonded terms i-j, j-k,
        and k-l must be present, and with k > 10 and b > 0.2 angstrom

    Note that this does not *strictly* guarantee that the system is stable. For example, ring systems do not have
    angles/bonds set to their equilibrium values:

                D
               /
        0--1--2
        |  |  |
        3--4--5

    The torsion term 0-1-2-D is numerically unstable, even though the angles and bonds are well-defined. Note that the
    0-1-2 angle term is not defined in the forcefield as having an equilibrium angle of zero. One additional sanity check
    that we can probably add later on is to also check the *initial* geometry's bond/angle length parameters.

    Conversely, an interaction may still be stable even if certain sub-component interactions are missing. For example,
    improper torsions are missing bonds and angles in the trefoil definitions, but the geometry is well defined nevertheless.

    Parameters
    ----------
    core: list of int
        core indices

    bond_idxs: list of 2-tuples
        (i,j) tuples of the bonded terms

    bond_params: list of 2-tuples
        (k, b) force constant, bond length

    angle_idxs: list of 3-tuples
        (i,j,k) tuples where the angle is defined between i-j-k

    angle_params: list of 2-tuples
        (k, a) force constant, angle in radians

    torsion_idxs: list of 4-tuples
        (i,j,k,l) tuples where the torsion is defined over i-j-k-l

    torsion_idxs: list of 3-tuples
        (k, p, n) force constant, period, phase

    Returns
    -------
    keep_angle_flags, keep_torsion_flags
        boolean flags indicating which angles and torsions we keep

    """

    assert len(bond_idxs) == len(bond_params)
    assert len(angle_idxs) == len(angle_params)
    assert len(torsion_idxs) == len(torsion_params)

    bond_kv = {}
    for ij, params in zip(bond_idxs, bond_params):
        bond_kv[canonicalize_bond(ij)] = params

    keep_angle_flags = []

    def ixn_is_all_core_or_all_dummy(atom_idxs):
        all_core = all(a in core for a in atom_idxs)
        all_dummy = all(a not in core for a in atom_idxs)

        return all_core or all_dummy

    def bond_is_okay(kb):
        k, b = kb
        return (k > min_bond_k) and (b > min_bond_l)

    def angle_is_okay(ka):
        k, a = ka
        return (k > min_angle_k) and (np.abs(a - np.pi) > min_angle_offset) and (np.abs(a - 0.0) > min_angle_offset)

    for (i, j, k), params in zip(angle_idxs, angle_params):
        b0 = canonicalize_bond((i, j))
        b1 = canonicalize_bond((j, k))
        if (b0 in bond_kv) and (b1 in bond_kv):
            bonds_okay = np.all([bond_is_okay(kb) for kb in [bond_kv[b0], bond_kv[b1]]])
        else:
            # missing bonds
            bonds_okay = False

        if ixn_is_all_core_or_all_dummy((i, j, k)) or bonds_okay:
            keep_angle_flags.append(1)
        else:
            keep_angle_flags.append(0)

    angle_kv = {}
    for idx, (ijk, params) in enumerate(zip(angle_idxs, angle_params)):
        if keep_angle_flags[idx]:
            angle_kv[canonicalize_bond(ijk)] = params

    keep_torsion_flags = []
    for (i, j, k, l), params in zip(torsion_idxs, torsion_params):
        b0 = canonicalize_bond((i, j))
        b1 = canonicalize_bond((j, k))
        b2 = canonicalize_bond((k, l))

        if (b0 in bond_kv) and (b1 in bond_kv) and (b2 in bond_kv):
            bonds_okay = np.all([bond_is_okay(kb) for kb in [bond_kv[b0], bond_kv[b1], bond_kv[b2]]])
        else:
            bonds_okay = False

        a0 = canonicalize_bond((i, j, k))
        a1 = canonicalize_bond((j, k, l))

        if (a0 in angle_kv) and (a1 in angle_kv):
            angles_okay = np.all([angle_is_okay(ka) for ka in [angle_kv[a0], angle_kv[a1]]])
        else:
            angles_okay = False

        if ixn_is_all_core_or_all_dummy((i, j, k, l)) or (bonds_okay and angles_okay):
            keep_torsion_flags.append(1)
        else:
            keep_torsion_flags.append(0)

    return keep_angle_flags, keep_torsion_flags
