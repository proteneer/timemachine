# maximum common subgraph routines based off of the mcgregor paper
import copy
import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray


def _arcs_left(marcs):
    num_row_edges = np.sum(np.any(marcs, 1))
    num_col_edges = np.sum(np.any(marcs, 0))
    return min(num_row_edges, num_col_edges)


# used in main recursion() loop
UNMAPPED = -1  # (UNVISITED) OR (VISITED AND DEMAPPED)

# used in inner loop when determining whether a mapping will necessarily result
# in a disconnected subgraph.
NODE_STATE_VISITED_DEMAPPED = 0  # VISITED AND DEMAPPED
NODE_STATE_VISITED_MAPPED = 1  # VISITED AND MAPPED
NODE_STATE_UNVISITED = 2  # UNVISITED


def _initialize_marcs_given_predicate(g1, g2, predicate):
    num_a_edges = g1.n_edges
    num_b_edges = g2.n_edges
    marcs = np.full((num_a_edges, num_b_edges), True, dtype=bool)
    for e_a in range(num_a_edges):
        src_a, dst_a = g1.edges[e_a]
        for e_b in range(num_b_edges):
            src_b, dst_b = g2.edges[e_b]
            # an edge mapping is allowed in two cases:
            # 1) src_a can map to src_b, and dst_a can map dst_b
            # 2) src_a can map to dst_b, and dst_a can map src_b
            # if either 1 or 2 is satisfied, we skip, otherwise
            # we can confidently reject the mapping
            if predicate[src_a][src_b] and predicate[dst_a][dst_b]:
                continue
            elif predicate[src_a][dst_b] and predicate[dst_a][src_b]:
                continue
            else:
                marcs[e_a][e_b] = False

    return marcs


def _verify_core_impl(g1, g2, new_v1, map_1_to_2):
    for e1 in g1.get_edges(new_v1):
        src, dst = g1.edges[e1]
        # both ends are mapped
        src_2, dst_2 = map_1_to_2[src], map_1_to_2[dst]
        if src_2 != UNMAPPED and dst_2 != UNMAPPED:
            # see if this edge is present in g2
            if not g2.cmat[src_2][dst_2]:
                return False
    return True


def _verify_core_is_connected(g1, g2, new_v1, new_v2, map_1_to_2, map_2_to_1):
    # incremental checks
    if _verify_core_impl(g1, g2, new_v1, map_1_to_2):
        return _verify_core_impl(g2, g1, new_v2, map_2_to_1)
    else:
        return False


def refine_marcs(g1, g2, new_v1, new_v2, marcs):
    """
    return vertices that have changed
    """
    new_marcs = copy.copy(marcs)

    if new_v2 == UNMAPPED:
        # zero out rows corresponding to the edges of new_v1
        new_marcs[g1.get_edges_as_vector(new_v1)] = False
    else:
        # mask out every row in marcs
        adj1 = g1.get_edges_as_vector(new_v1)
        adj2 = g2.get_edges_as_vector(new_v2)
        mask = np.where(adj1[:, np.newaxis], adj2, ~adj2)
        new_marcs &= mask

    return new_marcs


class MCSResult:
    def __init__(self):
        self.all_maps = []
        self.all_marcs = []
        self.num_edges = 0
        self.timed_out = False
        self.nodes_visited = 0


class Graph:
    def __init__(self, n_vertices, edges):
        self.n_vertices = n_vertices
        self.n_edges = len(edges)
        self.edges = edges
        self.nxg = nx.Graph(edges)  # assumes input graph is fully connected

        cmat = np.full((n_vertices, n_vertices), False, dtype=bool)
        for i, j in edges:
            cmat[i][j] = True
            cmat[j][i] = True

        self.cmat = cmat

        # list of lists, n_vertices x n_vertices
        self.lol_vertices = []
        for idx in range(n_vertices):
            nbs = []
            for jdx in range(n_vertices):
                if cmat[idx][jdx]:
                    nbs.append(jdx)
            self.lol_vertices.append(nbs)

        # list of lists, n_vertices x n_edges
        self.lol_edges = [[] for _ in range(n_vertices)]

        # note: lol_edges are not sorted.
        for edge_idx, (src, dst) in enumerate(edges):
            self.lol_edges[src].append(edge_idx)
            self.lol_edges[dst].append(edge_idx)

        self.ve_matrix = np.full((self.n_vertices, self.n_edges), False, dtype=bool)
        for vertex_idx, edges in enumerate(self.lol_edges):
            for edge_idx in edges:
                self.ve_matrix[vertex_idx][edge_idx] = True

    def mapping_is_disconnected(self, source, node_states, n_mapped_nodes):
        r"""
        Verify whether mapped nodes can still be connected in the graph. It is assumed that
        source is a mapped node. i.e. node_states[source] == NODE_STATE_VISITED_MAPPED. If this function
        returns True, then the resulting graph is definitely disconnected. If this function
        returns False, then the resulting graph can still be connected.

        For example, let M be mapped node, D be a demapped node, and U be an unvisited node:

        1. M-U-M # returns False
        2. M-D-M # returns True
        3. M-M-U # returns False
        4. M-M-D # returns False

        The implementation is adapted from the _plain_bfs() method in:
        https://networkx.org/documentation/stable/_modules/networkx/algorithms/components/connected.html

        """
        unseen = [True] * self.n_vertices
        unseen[source] = False
        nextlevel = [source]
        mapped_count = 1

        if mapped_count == n_mapped_nodes:
            return False

        while nextlevel:
            thislevel = nextlevel
            nextlevel = []
            for v in thislevel:
                for w in self.get_neighbors(v):
                    if unseen[w] and node_states[w] != NODE_STATE_VISITED_DEMAPPED:
                        if node_states[w] == NODE_STATE_VISITED_MAPPED:
                            mapped_count += 1
                            if mapped_count == n_mapped_nodes:
                                return False
                        unseen[w] = False
                        nextlevel.append(w)
        return True

    def get_neighbors(self, vertex):
        return self.lol_vertices[vertex]

    def get_edges(self, vertex):
        return self.lol_edges[vertex]

    def get_edges_as_vector(self, vertex):
        return self.ve_matrix[vertex]


def max_tree_size(priority_list):
    cur_layer_size = 1
    layer_sizes = [cur_layer_size]
    for neighbors in priority_list:
        cur_layer_size *= len(neighbors)
        layer_sizes.append(cur_layer_size)
    return sum(layer_sizes)


def build_predicate_matrix(n_a, n_b, priority_idxs):
    assert len(priority_idxs) == n_a
    pmat = np.full((n_a, n_b), False, dtype=bool)
    for idx, jdxs in enumerate(priority_idxs):
        for jdx in jdxs:
            pmat[idx][jdx] = True
    return pmat


class MaxVisitsWarning(UserWarning):
    pass


class NoMappingError(Exception):
    pass


@dataclass
class MCSDiagnostics:
    total_nodes_visited: int
    core_size: int
    num_cores: int


def core_to_perm(core: NDArray, num_atoms_a: int) -> Sequence[int]:
    a_to_b = {a: b for a, b in core}
    return [a_to_b.get(a, UNMAPPED) for a in range(num_atoms_a)]


def perm_to_core(perm: Sequence[int]) -> NDArray:
    core = []
    for a, b in enumerate(perm):
        if b != UNMAPPED:
            core.append((a, b))
    core_array = np.array(sorted(core))
    return core_array


def mcs(
    n_a,
    n_b,
    priority_idxs,
    bonds_a,
    bonds_b,
    max_visits,
    max_cores,
    enforce_core_core,
    connected_core,
    min_threshold,
    filter_fxn: Callable[[Sequence[int]], bool] = lambda core: True,
) -> Tuple[List[NDArray], List[NDArray], MCSDiagnostics]:
    assert n_a <= n_b

    g_a = Graph(n_a, bonds_a)
    g_b = Graph(n_b, bonds_b)

    predicate = build_predicate_matrix(n_a, n_b, priority_idxs)
    marcs = _initialize_marcs_given_predicate(g_a, g_b, predicate)

    priority_idxs = tuple(tuple(x) for x in priority_idxs)
    # Keep start time for debugging purposes below
    start_time = time.time()  # noqa

    mcs_result = None

    # run in reverse by guessing max # of edges to avoid getting stuck in minima.
    max_threshold = _arcs_left(marcs)
    total_nodes_visited = 0
    for idx in range(max_threshold):
        cur_threshold = max_threshold - idx
        if cur_threshold < min_threshold:
            raise NoMappingError(f"Unable to find mapping with at least {min_threshold} edges")
        map_a_to_b = [UNMAPPED] * n_a
        map_b_to_a = [UNMAPPED] * n_b
        mcs_result = MCSResult()
        recursion(
            g_a,
            g_b,
            map_a_to_b,
            map_b_to_a,
            0,
            marcs,
            mcs_result,
            priority_idxs,
            max_visits,
            max_cores,
            cur_threshold,
            enforce_core_core,
            connected_core,
            filter_fxn,
        )

        total_nodes_visited += mcs_result.nodes_visited

        # If timed out, either due to max_visits or max_cores, raise exception.
        if mcs_result.timed_out:
            warnings.warn(
                f"Reached max number of visits/cores: {len(mcs_result.all_maps)} cores with {mcs_result.nodes_visited} nodes visited. "
                "Cores may be suboptimal.",
                MaxVisitsWarning,
            )

        if len(mcs_result.all_maps) > 0:
            # don't remove this comment and the one below, useful for debugging!
            # print(
            # f"==SUCCESS==[NODES VISITED {mcs_result.nodes_visited} | CORE_SIZE {len([x != UNMAPPED for x in mcs_result.all_maps[0]])} | NUM_CORES {len(mcs_result.all_maps)} | NUM_EDGES {mcs_result.num_edges} | time taken: {time.time()-start_time} | time out? {mcs_result.timed_out}]====="
            # )
            break
        # else:
        # print(
        # f"==FAILED==[NODES VISITED {mcs_result.nodes_visited} | time taken: {time.time()-start_time} | time out? {mcs_result.timed_out}]====="
        # )

    assert mcs_result is not None

    if len(mcs_result.all_maps) == 0:
        raise NoMappingError("Unable to find mapping")

    all_cores = []

    for atom_map_1_to_2 in mcs_result.all_maps:
        core_array = perm_to_core(atom_map_1_to_2)
        all_cores.append(core_array)

    return (
        all_cores,
        mcs_result.all_marcs,
        MCSDiagnostics(
            total_nodes_visited=total_nodes_visited,
            core_size=len(all_cores[0]),
            num_cores=len(all_cores),
        ),
    )


def atom_map_add(map_1_to_2, map_2_to_1, idx, jdx):
    map_1_to_2[idx] = jdx
    map_2_to_1[jdx] = idx


def atom_map_pop(map_1_to_2, map_2_to_1, idx, jdx):
    map_1_to_2[idx] = UNMAPPED
    map_2_to_1[jdx] = UNMAPPED


def recursion(
    g1,
    g2,
    atom_map_1_to_2,
    atom_map_2_to_1,
    layer,
    marcs,
    mcs_result,
    priority_idxs,
    max_visits,
    max_cores,
    threshold,
    enforce_core_core,
    connected_core,
    filter_fxn,
):
    if mcs_result.nodes_visited > max_visits:
        mcs_result.timed_out = True
        return

    if len(mcs_result.all_maps) >= max_cores:
        mcs_result.timed_out = True
        return

    num_edges = _arcs_left(marcs)
    if num_edges < threshold:
        return

    if connected_core:
        # process g1 using atom_map_1_to_2_information
        g1_node_states = [NODE_STATE_VISITED_DEMAPPED] * g1.n_vertices
        g1_source = None
        g1_mapped_count = 0
        for a1, a2 in enumerate(atom_map_1_to_2):
            if a1 < layer:
                # visited nodes
                if a2 != UNMAPPED:
                    g1_node_states[a1] = NODE_STATE_VISITED_MAPPED
                    g1_source = a1
                    g1_mapped_count += 1
            else:
                g1_node_states[a1] = NODE_STATE_UNVISITED

        if g1_source and g1.mapping_is_disconnected(g1_source, g1_node_states, g1_mapped_count):
            return

        g2_node_states = [NODE_STATE_VISITED_DEMAPPED] * g2.n_vertices
        g2_source = None
        g2_mapped_count = 0

        # g2 is a little trickier to process, we need to look at the priority idxs as well
        for a2, a1 in enumerate(atom_map_2_to_1):
            if a1 != UNMAPPED:
                g2_node_states[a2] = NODE_STATE_VISITED_MAPPED
                g2_source = a2
                g2_mapped_count += 1

        # look up priority_idxs of remaining atoms
        for a2_list in priority_idxs[layer:]:
            for a2 in a2_list:
                if g2_node_states[a2] != NODE_STATE_VISITED_MAPPED:
                    g2_node_states[a2] = NODE_STATE_UNVISITED

        if g2_source and g2.mapping_is_disconnected(g2_source, g2_node_states, g2_mapped_count):
            return

    mcs_result.nodes_visited += 1
    n_a = g1.n_vertices

    # leaf-node, every atom has been mapped
    if layer == n_a:
        if num_edges == threshold:
            mcs_result.all_maps.append(copy.copy(atom_map_1_to_2))
            mcs_result.all_marcs.append(copy.copy(marcs))
            mcs_result.num_edges = num_edges
        return

    for jdx in priority_idxs[layer]:
        if atom_map_2_to_1[jdx] == UNMAPPED:  # optimize later
            atom_map_add(atom_map_1_to_2, atom_map_2_to_1, layer, jdx)
            if enforce_core_core and not _verify_core_is_connected(
                g1, g2, layer, jdx, atom_map_1_to_2, atom_map_2_to_1
            ):
                pass
            elif not filter_fxn(atom_map_1_to_2):
                pass
            else:
                new_marcs = refine_marcs(g1, g2, layer, jdx, marcs)
                recursion(
                    g1,
                    g2,
                    atom_map_1_to_2,
                    atom_map_2_to_1,
                    layer + 1,
                    new_marcs,
                    mcs_result,
                    priority_idxs,
                    max_visits,
                    max_cores,
                    threshold,
                    enforce_core_core,
                    connected_core,
                    filter_fxn,
                )
            atom_map_pop(atom_map_1_to_2, atom_map_2_to_1, layer, jdx)

    # always allow for explicitly not mapping layer atom
    # nit: don't need to check for connected core if mapping to None
    new_marcs = refine_marcs(g1, g2, layer, UNMAPPED, marcs)

    recursion(
        g1,
        g2,
        atom_map_1_to_2,
        atom_map_2_to_1,
        layer + 1,
        new_marcs,
        mcs_result,
        priority_idxs,
        max_visits,
        max_cores,
        threshold,
        enforce_core_core,
        connected_core,
        filter_fxn,
    )
