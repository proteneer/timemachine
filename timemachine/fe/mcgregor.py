# maximum common subgraph routines based off of the mcgregor paper
import copy
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray


def _arcs_left(marcs):
    num_row_edges = np.sum(np.any(marcs, 1))
    num_col_edges = np.sum(np.any(marcs, 0))
    return min(num_row_edges, num_col_edges)


# used in main recursion() loop
UNMAPPED = -1  # (UNVISITED) OR (VISITED AND DEMAPPED)


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
        self.leaves_visited = 0


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

    def mapping_incompatible_with_cc_constraints(
        self,
        mapped_nodes: Set[int],
        unvisited_nodes: Set[int],
        max_connected_components: Optional[int],
        min_connected_component_size: int,
    ):
        r"""Returns whether the current search state (as specified by mapped nodes and unvisited nodes) is
        _incompatible_ with either of the `max_connected_components` or `min_connected_component_size` constraints.

        For example, let M be mapped node, D be a demapped (visited but not mapped) node, and U be an unvisited node. If
        `max_connected_components=1` and `min_connected_component_size=1`:

        1. M-U-M # returns False
        2. M-D-M # returns True
        3. M-M-U # returns False
        4. M-M-D # returns False

        With `min_connected_component_size=2`, example 1 would be incompatible and we would return True (3 and 4 would
        still be False).

        This calls an optimized implementation in mapping_incompatible_with_cc_constraints_fast; see
        mapping_incompatible_with_cc_constraints_ref for a simpler-to-understand reference version.
        """

        # Uncomment to assert result of fast implementation matches the reference
        # args = (mapped_nodes, unvisited_nodes, max_connected_components, min_connected_component_size)
        # assert self.mapping_incompatible_with_cc_constraints_fast(
        #     *args
        # ) == self.mapping_incompatible_with_cc_constraints_ref(*args)

        return self.mapping_incompatible_with_cc_constraints_fast(
            mapped_nodes, unvisited_nodes, max_connected_components, min_connected_component_size
        )

    def mapping_incompatible_with_cc_constraints_fast(
        self,
        mapped_nodes: Set[int],
        unvisited_nodes: Set[int],
        max_connected_components: Optional[int],
        min_connected_component_size: int,
    ):
        """Optimized implementation of mapping_incompatible_with_cc_constraints_ref, sacrificing modularity for efficiency"""

        seen = set()
        n_ccs = 0
        for u in mapped_nodes:
            if u not in seen:
                # visit component containing v
                seen.add(u)
                cc_size = 1
                nextlevel = [u]
                while nextlevel:
                    thislevel = nextlevel
                    nextlevel = []
                    for v in thislevel:
                        for w in self.get_neighbors(v):
                            if w in mapped_nodes or w in unvisited_nodes:
                                if w not in seen:
                                    seen.add(w)
                                    cc_size += 1
                                    nextlevel.append(w)
                n_ccs += 1
                if cc_size < min_connected_component_size:
                    return True
                if max_connected_components is not None and n_ccs == max_connected_components:
                    # if we've seen the maximum number of connected components, we should have seen all of the mapped nodes
                    return not mapped_nodes.issubset(seen)

        return False

    def mapping_incompatible_with_cc_constraints_ref(
        self,
        mapped_nodes: Set[int],
        unvisited_nodes: Set[int],
        max_connected_components: Optional[int],
        min_connected_component_size: int,
    ):
        """Reference implementation of mapping_incompatible_with_cc_constraints, decomposed into standard algorithms for
        clarity"""

        g = self.to_networkx()

        # Consider the subgraph induced by mapped and unvisited nodes (ignoring visited nodes that have not been mapped)
        sg = g.subgraph(mapped_nodes | unvisited_nodes)

        n_ccs_with_mapped_nodes = 0

        for cc in nx.connected_components(sg):
            if cc.intersection(mapped_nodes):
                n_ccs_with_mapped_nodes += 1

                # Visiting the remaining nodes can only maintain or increase the number of connected components.
                if max_connected_components and n_ccs_with_mapped_nodes > max_connected_components:
                    return True

                # Visiting the remaining nodes can only maintain or shrink a connected component
                if len(cc) < min_connected_component_size:
                    return True

        return False

    def get_neighbors(self, vertex):
        return self.lol_vertices[vertex]

    def get_edges(self, vertex):
        return self.lol_edges[vertex]

    def get_edges_as_vector(self, vertex):
        return self.ve_matrix[vertex]

    def to_networkx(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.n_vertices))
        g.add_edges_from(self.edges)
        return g


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
    total_leaves_visited: int
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
    max_connected_components: Optional[int],
    min_connected_component_size: int,
    min_threshold,
    initial_mapping,
    filter_fxn: Callable[[Sequence[int]], bool] = lambda _: True,
    leaf_filter_fxn: Callable[[Sequence[int]], bool] = lambda _: True,
) -> Tuple[List[NDArray], List[NDArray], MCSDiagnostics]:
    assert n_a <= n_b
    assert max_connected_components is None or max_connected_components > 0, "Must have max_connected_components > 0"

    predicate = build_predicate_matrix(n_a, n_b, priority_idxs)
    g_a = Graph(n_a, bonds_a)
    g_b = Graph(n_b, bonds_b)
    base_marcs = _initialize_marcs_given_predicate(g_a, g_b, predicate)

    base_map_a_to_b = [UNMAPPED] * n_a
    base_map_b_to_a = [UNMAPPED] * n_b
    if initial_mapping is not None:
        for a, b in initial_mapping:
            base_map_a_to_b[a] = b
            base_map_b_to_a[b] = a
            base_marcs = refine_marcs(g_a, g_b, a, b, base_marcs)

    base_layer = len(initial_mapping)
    priority_idxs = tuple(tuple(x) for x in priority_idxs)
    # Keep start time for debugging purposes below
    # import time
    # start_time = time.time()  # noqa

    # run in reverse by guessing max # of edges to avoid getting stuck in minima.
    max_threshold = _arcs_left(base_marcs)
    if max_threshold < 1:
        raise NoMappingError("No possible mapping given the predicate matrix, verify molecules are aligned")

    total_nodes_visited = 0
    total_leaves_visited = 0
    mcs_result = MCSResult()
    for idx in range(max_threshold):
        cur_threshold = max_threshold - idx
        if cur_threshold < min_threshold:
            raise NoMappingError(f"Unable to find mapping with at least {min_threshold} edges")
        # Construct copies of the base map
        map_a_to_b = list(base_map_a_to_b)
        map_b_to_a = list(base_map_b_to_a)
        mcs_result = MCSResult()
        recursion(
            g_a,
            g_b,
            map_a_to_b,
            map_b_to_a,
            base_layer,
            base_marcs,
            mcs_result,
            priority_idxs,
            # Decrement the max visits by the number already visited to ensure constant wall clock time
            max_visits - total_nodes_visited,
            max_cores,
            cur_threshold,
            enforce_core_core,
            max_connected_components,
            min_connected_component_size,
            filter_fxn,
            leaf_filter_fxn,
        )

        total_nodes_visited += mcs_result.nodes_visited
        total_leaves_visited += mcs_result.leaves_visited

        if len(mcs_result.all_maps) > 0:
            # If we timed out but got cores, throw a warning
            if mcs_result.timed_out and len(mcs_result.all_maps) < max_cores:
                warnings.warn(
                    f"Inexhaustive search: reached max number of visits ({max_visits}) and found only "
                    + f"{len(mcs_result.all_maps)} out of {max_cores} desired cores.",
                    MaxVisitsWarning,
                )
            # don't remove this comment and the one below, useful for debugging!
            # print(
            # f"==SUCCESS==[NODES VISITED {mcs_result.nodes_visited} | CORE_SIZE {len([x != UNMAPPED for x in mcs_result.all_maps[0]])} | NUM_CORES {len(mcs_result.all_maps)} | NUM_EDGES {mcs_result.num_edges} | time taken: {time.time()-start_time} | time out? {mcs_result.timed_out}]====="
            # )
            break
        elif mcs_result.timed_out:
            # If timed out, either due to max_visits or max_cores, raise exception.
            raise NoMappingError(
                f"Exceeded max number of visits/cores - no valid cores could be found: {total_nodes_visited} nodes visited."
            )
        # else:
        # print(
        # f"==FAILED==[NODES VISITED {mcs_result.nodes_visited} | time taken: {time.time()-start_time} | time out? {mcs_result.timed_out}]====="
        # )

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
            total_leaves_visited=total_leaves_visited,
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
    g1: Graph,
    g2: Graph,
    atom_map_1_to_2,
    atom_map_2_to_1,
    layer,
    marcs,
    mcs_result: MCSResult,
    priority_idxs,
    max_visits,
    max_cores,
    threshold,
    enforce_core_core,
    max_connected_components: Optional[int],
    min_connected_component_size: int,
    filter_fxn: Callable[[Sequence[int]], bool],
    leaf_filter_fxn: Callable[[Sequence[int]], bool],
):
    if mcs_result.nodes_visited >= max_visits:
        mcs_result.timed_out = True
        return

    if len(mcs_result.all_maps) >= max_cores:
        mcs_result.timed_out = True
        return

    num_edges = _arcs_left(marcs)
    if num_edges < threshold:
        return

    if max_connected_components is not None or min_connected_component_size > 1:
        g1_mapped_nodes = {a1 for a1, a2 in enumerate(atom_map_1_to_2[:layer]) if a2 != UNMAPPED}

        if g1_mapped_nodes:
            # Nodes in g1 are visited in order, so nodes left to visit are [layer, layer + 1, ..., n - 1]
            g1_unvisited_nodes = set(range(layer, g1.n_vertices))
            if g1.mapping_incompatible_with_cc_constraints(
                g1_mapped_nodes, g1_unvisited_nodes, max_connected_components, min_connected_component_size
            ):
                return

        g2_mapped_nodes = {a2 for a2, a1 in enumerate(atom_map_2_to_1) if a1 != UNMAPPED}

        if g2_mapped_nodes:
            # Nodes in g2 are visited in the order determined by priority_idxs. Nodes may be repeated in priority_idxs,
            # but we skip over nodes that have already been mapped.
            g2_unvisited_nodes = {a2 for a2s in priority_idxs[layer:] for a2 in a2s if a2 not in g2_mapped_nodes}
            if g2.mapping_incompatible_with_cc_constraints(
                g2_mapped_nodes, g2_unvisited_nodes, max_connected_components, min_connected_component_size
            ):
                return

    mcs_result.nodes_visited += 1
    n_a = g1.n_vertices

    # leaf-node, every atom has been mapped
    if layer == n_a:
        if num_edges == threshold:
            mcs_result.leaves_visited += 1
            if not leaf_filter_fxn(atom_map_1_to_2):
                return

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
                    max_connected_components,
                    min_connected_component_size,
                    filter_fxn,
                    leaf_filter_fxn,
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
        max_connected_components,
        min_connected_component_size,
        filter_fxn,
        leaf_filter_fxn,
    )
