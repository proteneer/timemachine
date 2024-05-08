# maximum common subgraph routines based off of the mcgregor paper
import copy
import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def _arcs_left(marcs):
    num_row_edges = np.sum(np.any(marcs, 1))
    num_col_edges = np.sum(np.any(marcs, 0))
    return min(num_row_edges, num_col_edges)


UNMAPPED = -1


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
        self.num_atoms = 0


class Graph:
    def __init__(self, n_vertices, edges):
        self.n_vertices = n_vertices
        self.n_edges = len(edges)
        self.edges = edges

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
    min_threshold,
    initial_mapping,
    filter_fxn: Callable[[Sequence[int]], bool] = lambda core: True,
) -> Tuple[List[NDArray], List[NDArray], MCSDiagnostics]:
    predicate = build_predicate_matrix(n_a, n_b, priority_idxs)
    g_a = Graph(n_a, bonds_a)
    g_b = Graph(n_b, bonds_b)
    base_marcs = _initialize_marcs_given_predicate(g_a, g_b, predicate)

    base_map_a_to_b = [UNMAPPED] * n_a
    base_map_b_to_a = [UNMAPPED] * n_b

    assert len(initial_mapping) == 0
    # if initial_mapping is not None:
    #     for a, b in initial_mapping:
    #         base_map_a_to_b[a] = b
    #         base_map_b_to_a[b] = a
    #         base_marcs = refine_marcs(g_a, g_b, a, b, base_marcs)

    base_layer = len(initial_mapping)
    priority_idxs = tuple(tuple(x) for x in priority_idxs)
    # Keep start time for debugging purposes below
    start_time = time.time()  # noqa

    mcs_result = None

    # run in reverse by guessing max # of edges to avoid getting stuck in minima.
    max_threshold = _arcs_left(base_marcs)
    total_nodes_visited = 0
    for idx in range(max_threshold):
        cur_threshold = max_threshold - idx
        if cur_threshold < min_threshold:
            raise NoMappingError(f"Unable to find mapping with at least {min_threshold} atoms")
        mcs_result = MCSResult()
        core_dict = dict()

        print("CUR THRESHOLD", cur_threshold)
        recursion_v2(
            g_a,
            g_b,
            core_dict,
            base_marcs,
            mcs_result,
            predicate,
            max_visits,
            max_cores,
            max_threshold,
            filter_fxn,
            dict(),
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
        all_cores.append(np.array(core_array))

    # print(all_cores[0])

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


def find_neighbors(g, atom, core):
    return [x for x in g.get_neighbors(atom) if x not in core]


# def _cbc(mol_a, mol_b, marcs):
#     a_edges = get_edges(mol_a)
#     b_edges = get_edges(mol_b)
#     bond_core = {}
#     for e_a in range(len(a_edges)):
#         src_a, dst_a = a_edges[e_a]
#         for e_b in range(len(b_edges)):
#             src_b, dst_b = b_edges[e_b]
#             if marcs[e_a][e_b]:
#                 assert (src_a, dst_a) not in bond_core
#                 assert (dst_a, src_a) not in bond_core
#                 bond_core[(src_a, dst_a)] = (src_b, dst_b)
#     return bond_core

import networkx as nx


def _verify_connectivity(g1, g2, core_dict):
    # verify that the induced common-core is a single component
    nxg = nx.Graph()
    for i, j in g1.edges:
        assert g1.cmat[i][j]
        if i in core_dict and j in core_dict:
            map_i = core_dict[i]
            map_j = core_dict[j]
            if g2.cmat[map_i][map_j]:
                nxg.add_edge(i, j)

    cc_a = list(nx.connected_components(nxg))  # this can be 0 or 1
    assert len(cc_a) < 2


def recursion_v2(
    g1,
    g2,
    core_dict,
    marcs,
    mcs_result,
    pred_mat,
    max_visits,
    max_cores,
    threshold,
    filter_fxn,
    canonical_visited_core_1s,
):
    mcs_result.nodes_visited += 1

    if mcs_result.nodes_visited > max_visits:
        mcs_result.timed_out = True
        return

    if len(mcs_result.all_maps) > max_cores:
        mcs_result.timed_out = True
        return

    num_edges = _arcs_left(marcs)

    # filter on predicates matrix and connected component size?
    # pmat_copy = copy.deepcopy(pred_mat)
    # for i,j in core_dict.items():
    #     # zero out rows
    #     pmat_copy[i] = 0
    #     pmat_copy[:, j] = 0
    #     pmat_copy[i,j] = 1

    # num_vertices = _arcs_left(pmat_copy)
    # print("num_vertices", num_vertices, "core_size", len(core_dict))

    # if num_edges < threshold:
    # print("failed thresh", threshold, "num edges", num_edges)
    # return

    # debug
    rev_core_dict = {v: k for k, v in core_dict.items()}
    if len(core_dict) > 0:
        _verify_connectivity(g1, g2, core_dict)
        _verify_connectivity(g2, g1, rev_core_dict)

    if mcs_result.nodes_visited % 1000 == 0:
        print(
            "Visited",
            mcs_result.nodes_visited,
            "nodes",
            len(core_dict),
            "atoms mapped with",
            num_edges,
            "possible edges",
        )

    atom_map_1_to_2 = np.zeros(g1.n_vertices, dtype=np.int32) - 1
    for c1, c2 in core_dict.items():
        atom_map_1_to_2[c1] = c2

    def filter_candidates(g, core):
        outer_vertices = [x for x in range(g.n_vertices) if x not in core]
        keep = []
        for v in outer_vertices:
            if len(core) > 0:
                v_is_connected = False
                for nb in g.get_neighbors(v):
                    if nb in core:
                        v_is_connected = True
                        break
                if v_is_connected:
                    assert v not in keep
                    keep.append(v)
            else:
                assert v not in keep
                keep.append(v)
        return keep

    nbs_1 = filter_candidates(g1, core_dict.keys())
    nbs_2 = filter_candidates(g2, core_dict.values())
    choices = []

    for v1 in nbs_1:
        for v2 in nbs_2:
            # tbd: other filters
            if pred_mat[v1][v2] and filter_fxn(atom_map_1_to_2):
                v1_core_bond_count = 0
                for v1_nb in g1.get_neighbors(v1):
                    # v1, v1_nb -> v2, core_dict[v1_nb]
                    if v1_nb in core_dict.keys() and g2.cmat[v2][core_dict[v1_nb]]:
                        v1_core_bond_count += 1

                v2_core_bond_count = 0
                for v2_nb in g2.get_neighbors(v2):
                    # v2, v2_nb -> v1, core_dict[v2_nb]
                    if v2_nb in rev_core_dict.keys() and g1.cmat[v1][rev_core_dict[v2_nb]]:
                        v2_core_bond_count += 1

                if len(core_dict) == 0 or (v1_core_bond_count and v2_core_bond_count):
                    choices.append((v1, v2))

    # print("CHOICES", choices)
    # print("LEN", len(choices))

    # terminal node
    # wait: did this actually terminate? is this # of truly mapped edges?
    if len(choices) == 0:
        # if num_edges == threshold:
        print(
            "terminal_node with",
            len(core_dict),
            "mapped atoms, and visited",
            "num_edges",
            num_edges,
            mcs_result.nodes_visited,
            "nodes",
        )
        # if len(core_dict) > 30:
        # print(core_dict)
        mcs_result.num_atoms = len(core_dict)
        # mcs_result.all_maps.append(copy.copy(atom_map_1_to_2))
        # mcs_result.all_marcs.append(copy.copy(marcs))
        mcs_result.num_edges = num_edges
    # else:
    # print("!! terminal_node but not at threshold", core_dict, num_edges, "visited", mcs_result.nodes_visited)

    # print("number of choices", len(choices))

    for v1, v2 in choices:
        core_dict[v1] = v2
        new_marcs = refine_marcs(g1, g2, v1, v2, marcs)
        core_key = hash(frozenset(core_dict.items()))
        if core_key not in canonical_visited_core_1s:
            # print("Not Skipping", len(canonical_visited_core_1s))
            recursion_v2(
                g1,
                g2,
                core_dict,
                new_marcs,
                mcs_result,
                pred_mat,
                max_visits,
                max_cores,
                threshold,
                filter_fxn,
                canonical_visited_core_1s,
            )
            canonical_visited_core_1s[core_key] = _arcs_left(new_marcs)
        else:
            # print("Skipping", len(canonical_visited_core_1s))
            assert _arcs_left(new_marcs) == canonical_visited_core_1s[core_key]
        core_dict.pop(v1)

    return
