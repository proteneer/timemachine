# maximum common subgraph routines based off of the mcgregor paper
import copy
import time

import numpy as np


def _arcs_left(marcs):
    num_row_edges = np.sum(np.any(marcs, 1))
    num_col_edges = np.sum(np.any(marcs, 0))
    return min(num_row_edges, num_col_edges)


UNMAPPED = -1


def _initialize_marcs_given_predicate(g1, g2, predicate):
    num_a_edges = g1.n_edges
    num_b_edges = g2.n_edges
    marcs = np.ones((num_a_edges, num_b_edges), dtype=np.byte)
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
                marcs[e_a][e_b] = 0

    return marcs


def _verify_core_impl(g1, g2, new_v1, map_1_to_2):
    for e1 in g1.get_edges(new_v1):
        src, dst = g1.edges[e1]
        # both ends are mapped
        src_2, dst_2 = map_1_to_2[src], map_1_to_2[dst]
        if src_2 != UNMAPPED and dst_2 != UNMAPPED:
            # see if this edge is present in g2
            if g2.cmat[src_2][dst_2] == 0:
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
        for e1 in g1.get_edges(new_v1):
            # this is equivalent to new_marcs[e1] = np.zeros(marcs.shape[1])
            new_marcs[e1] = 0
    else:
        # mask out every row in marcs
        # eg. returns [0,1,0,0,1,0,1]
        mask = g2.get_edges_as_vector(new_v2)
        # eg. returns [1,0,1,1,0,1,0]
        antimask = 1 - mask
        for e1_idx, is_v1_edge in enumerate(g1.get_edges_as_vector(new_v1)):
            if is_v1_edge:
                new_marcs[e1_idx] &= mask
            else:
                new_marcs[e1_idx] &= antimask

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

        cmat = np.zeros((n_vertices, n_vertices), dtype=np.byte)
        for i, j in edges:
            cmat[i][j] = 1
            cmat[j][i] = 1

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

        self.ve_matrix = np.zeros((self.n_vertices, self.n_edges), dtype=np.byte)
        for vertex_idx, edges in enumerate(self.lol_edges):
            for edge_idx in edges:
                self.ve_matrix[vertex_idx][edge_idx] = 1

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
    pmat = np.zeros((n_a, n_b), dtype=np.int32)
    for idx, jdxs in enumerate(priority_idxs):
        for jdx in jdxs:
            pmat[idx][jdx] = 1
    return pmat


class MaxVisitsError(Exception):
    pass


def mcs(
    n_a, n_b, priority_idxs, bonds_a, bonds_b, max_visits, max_cores, enforce_core_core, filter_fxn=(lambda core: True)
):

    assert n_a <= n_b

    g_a = Graph(n_a, bonds_a)
    g_b = Graph(n_b, bonds_b)

    predicate = build_predicate_matrix(n_a, n_b, priority_idxs)
    marcs = _initialize_marcs_given_predicate(g_a, g_b, predicate)

    priority_idxs = tuple(tuple(x) for x in priority_idxs)
    start_time = time.time()

    # run in reverse by guessing max # of edges to avoid getting stuck in minima.
    max_threshold = _arcs_left(marcs)
    for idx in range(max_threshold):
        cur_threshold = max_threshold - idx
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
            start_time,
            max_visits,
            max_cores,
            cur_threshold,
            enforce_core_core,
            filter_fxn,
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

        # If timed out and no maps found, raise exception.
        if mcs_result.timed_out:
            raise MaxVisitsError(f"Reached max number of visits: {max_visits}")

    assert len(mcs_result.all_maps) > 0

    all_cores = []

    for atom_map_1_to_2 in mcs_result.all_maps:
        core = []
        for a, b in enumerate(atom_map_1_to_2):
            if b != UNMAPPED:
                core.append((a, b))
        core = np.array(sorted(core))
        all_cores.append(core)

    return all_cores, mcs_result.all_marcs


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
    start_time,
    max_visits,
    max_cores,
    threshold,
    enforce_core_core,
    filter_fxn,
):

    if mcs_result.nodes_visited > max_visits:
        mcs_result.timed_out = True
        return

    if len(mcs_result.all_maps) == max_cores:
        return

    num_edges = _arcs_left(marcs)
    if num_edges < threshold:
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
                    start_time,
                    max_visits,
                    max_cores,
                    threshold,
                    enforce_core_core,
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
        start_time,
        max_visits,
        max_cores,
        threshold,
        enforce_core_core,
        filter_fxn,
    )
