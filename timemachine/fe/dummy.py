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


def canonicalize_bond(ixn):
    if ixn[0] > ixn[-1]:
        return tuple(ixn[::-1])
    else:
        return tuple(ixn)
