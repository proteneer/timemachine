# maximum common subgraph routines based off of the mcgregor paper
import warnings
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .tree_search import best_first

UNMAPPED = -1  # (UNVISITED) OR (VISITED AND DEMAPPED)


class Graph:
    def __init__(self, n_vertices, edges):
        self.n_vertices = n_vertices
        self.n_edges = len(edges)
        self.edges = np.asarray(edges)

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


@dataclass(frozen=True)
class Marcs:
    marcs: NDArray[np.bool_]

    @cached_property
    def num_edges_upper_bound(self) -> int:
        """Upper bound on the number of edges that can be put in correspondence, given the current partial mapping.

        McGregor (1982) refers to this as "arcsleft".
        """
        num_row_edges = self.marcs.any(1).sum()
        num_col_edges = self.marcs.any(0).sum()
        return min(num_row_edges, num_col_edges)

    @classmethod
    def from_predicate(cls, g1: Graph, g2: Graph, predicate: NDArray[np.bool_]) -> "Marcs":
        src_a = g1.edges[:, None, 0]
        dst_a = g1.edges[:, None, 1]
        src_b = g2.edges[None, :, 0]
        dst_b = g2.edges[None, :, 1]

        # An edge mapping is allowed in two cases:
        # 1) src_a can map to src_b, and dst_a can map dst_b
        # 2) src_a can map to dst_b, and dst_a can map src_b
        # The mapping is allowed iff (1) or (2) is true

        marcs = np.logical_or(
            predicate[src_a, src_b] & predicate[dst_a, dst_b],
            predicate[src_a, dst_b] & predicate[dst_a, src_b],
        )

        return Marcs(marcs)

    def refine(self, g1: Graph, g2: Graph, new_v1: int, new_v2: int) -> "Marcs":
        """Set to False entries in marcs corresponding to edge pairs that can no longer correspond under the addition of
        (new_v1, new_v2) to the mapping"""

        # TODO: handle case when new_v2 is demapped
        # (i.e., set to False columns corresponding to the edges incident on new_v2)
        assert new_v1 != UNMAPPED

        new_marcs = np.array(self.marcs)
        e1 = g1.get_edges_as_vector(new_v1)

        if new_v2 == UNMAPPED:
            # Set to False rows corresponding to the edges incident on new_v1
            new_marcs[e1, :] = False
        else:
            e2 = g2.get_edges_as_vector(new_v2)

            # Set to False edge pairs (e1, e2) where either
            # 1. e1 connects to new_v1 but e2 does NOT connect to new_v2
            # 2. e1 does NOT connect to new_v1 but e2 connects to new_v2
            mask = e1[:, None] == e2[None, :]
            new_marcs &= mask

        return Marcs(new_marcs)


@dataclass(frozen=True)
class AtomMap:
    a_to_b: Tuple[int, ...]
    b_to_a: Tuple[int, ...]

    @classmethod
    def init(cls, n_1: int, n_2: int) -> "AtomMap":
        return cls((UNMAPPED,) * n_1, (UNMAPPED,) * n_2)

    def add(self, new_v1: int, new_v2: int) -> "AtomMap":
        def set_at(xs: Tuple[int, ...], idx: int, val: int) -> Tuple[int, ...]:
            return xs[:idx] + (val,) + xs[idx + 1 :]

        return AtomMap(
            set_at(self.a_to_b, new_v1, new_v2),
            set_at(self.b_to_a, new_v2, new_v1),
        )

    @cached_property
    def core_size(self):
        return sum(1 for j in self.a_to_b if j != UNMAPPED)


def _verify_map_preserves_core_edges(g1: Graph, g2: Graph, new_v1: int, new_v2: int, atom_map: AtomMap) -> bool:
    def verify(g1: Graph, g2: Graph, new_v1: int, map_1_to_2: Sequence[int]):
        for e1 in g1.get_edges(new_v1):
            src, dst = g1.edges[e1]
            src_2, dst_2 = map_1_to_2[src], map_1_to_2[dst]
            if src_2 != UNMAPPED and dst_2 != UNMAPPED:
                # both ends are mapped
                # see if this edge is present in g2
                if not g2.cmat[src_2][dst_2]:
                    return False
        return True

    return verify(g1, g2, new_v1, atom_map.a_to_b) and verify(g2, g1, new_v2, atom_map.b_to_a)


@dataclass(frozen=True)
class Node:
    atom_map: AtomMap
    marcs: Marcs
    layer: int

    @classmethod
    def init(cls, g1: Graph, g2: Graph, predicate: NDArray[np.bool_]) -> "Node":
        return cls(AtomMap.init(g1.n_vertices, g2.n_vertices), Marcs.from_predicate(g1, g2, predicate), 0)

    def add(self, g1: Graph, g2: Graph, new_v2: int) -> "Node":
        # TODO: further refine the marcs matrix accounting for nodes in g2 that are implicitly demapped (by looking at
        # priority_idxs; probably requires extra bookkeeping to be fast)
        return Node(
            self.atom_map.add(self.layer, new_v2),
            self.marcs.refine(g1, g2, self.layer, new_v2),
            self.layer + 1,
        )

    def skip(self, g1: Graph, g2: Graph) -> "Node":
        # TODO: further refine the marcs matrix accounting for nodes in g2 that are implicitly demapped (by looking at
        # priority_idxs; probably requires extra bookkeeping to be fast)
        return Node(
            self.atom_map,
            self.marcs.refine(g1, g2, self.layer, UNMAPPED),
            self.layer + 1,
        )

    @cached_property
    def is_leaf(self):
        return self.layer == len(self.atom_map.a_to_b)

    @cached_property
    def priority(self):
        """Compute the priority of this node. By convention, lowest numerical value is highest priority"""
        return (-self.marcs.num_edges_upper_bound, -self.layer)

    def __lt__(self, other: "Node") -> bool:
        return self.priority < other.priority


@dataclass(frozen=True)
class MCSResult:
    all_maps: Tuple[Tuple[int, ...], ...]
    all_marcs: Tuple[NDArray, ...]
    num_edges: int
    timed_out: bool
    nodes_visited: int
    leaves_visited: int

    @classmethod
    def from_nodes(
        cls, nodes: Iterable[Node], leaf_filter_fxn: Callable[[Tuple[int, ...]], bool], max_nodes: int, max_leaves: int
    ) -> "MCSResult":
        all_maps: List[Tuple[int, ...]] = []
        all_marcs: List[NDArray[np.bool_]] = []

        node = None
        nodes_visited = 0
        leaves_visited = 0
        timed_out = False

        for nodes_visited, node in enumerate(nodes, 1):
            if node.is_leaf and node.atom_map.core_size > 0:
                if leaf_filter_fxn(node.atom_map.a_to_b):
                    all_maps.append(node.atom_map.a_to_b)
                    all_marcs.append(node.marcs.marcs)

                leaves_visited += 1

                if leaves_visited == max_leaves:
                    timed_out = True
                    break

            if nodes_visited == max_nodes:
                timed_out = True
                break

        assert node is not None, "found no valid mappings"

        return MCSResult(
            tuple(all_maps),
            tuple(all_marcs),
            node.marcs.num_edges_upper_bound,
            timed_out=timed_out,
            nodes_visited=nodes_visited,
            leaves_visited=leaves_visited,
        )


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


@dataclass(frozen=True)
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
    min_num_edges: int,
    initial_mapping,
    filter_fxn: Callable[[Sequence[int]], bool] = lambda _: True,
    leaf_filter_fxn: Callable[[Sequence[int]], bool] = lambda _: True,
) -> Tuple[List[NDArray], List[NDArray], MCSDiagnostics]:
    assert n_a <= n_b
    assert max_connected_components is None or max_connected_components > 0, "Must have max_connected_components > 0"

    predicate = build_predicate_matrix(n_a, n_b, priority_idxs)
    g_a = Graph(n_a, bonds_a)
    g_b = Graph(n_b, bonds_b)

    init_node = Node.init(g_a, g_b, predicate)

    if initial_mapping is not None:
        initial_mapping_dict = {a: b for a, b in initial_mapping}
        for a in range(len(initial_mapping)):
            init_node = init_node.add(g_a, g_b, initial_mapping_dict.get(a, UNMAPPED))

    if init_node.marcs.num_edges_upper_bound == 0:
        raise NoMappingError("No possible mapping given the predicate matrix")

    priority_idxs = tuple(tuple(x) for x in priority_idxs)
    # Keep start time for debugging purposes below
    # import time
    # start_time = time.time()

    cached_leaf_filter_fxn = cache(leaf_filter_fxn)

    expand = make_expand(
        g_a,
        g_b,
        priority_idxs,
        enforce_core_core,
        max_connected_components,
        min_connected_component_size,
        filter_fxn,
        cached_leaf_filter_fxn,
    )

    nodes = best_first(expand, init_node, min_num_edges)

    mcs_result = MCSResult.from_nodes(nodes, cached_leaf_filter_fxn, max_visits, max_cores)

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

    elif mcs_result.timed_out:
        # If timed out, either due to max_visits or max_cores, raise exception.
        raise NoMappingError(
            f"Exceeded max number of visits/cores - no valid cores could be found: {mcs_result.nodes_visited} nodes visited."
        )
    # else:
    # print(
    # f"==FAILED==[NODES VISITED {mcs_result.nodes_visited} | time taken: {time.time()-start_time} | time out? {mcs_result.timed_out}]====="
    # )

    if len(mcs_result.all_maps) == 0:
        raise NoMappingError(f"Unable to find mapping with at least {min_num_edges} edges")

    all_cores = []

    for a_to_b in mcs_result.all_maps:
        core_array = perm_to_core(a_to_b)
        all_cores.append(core_array)

    return (
        all_cores,
        list(mcs_result.all_marcs),
        MCSDiagnostics(
            total_nodes_visited=mcs_result.nodes_visited,
            total_leaves_visited=mcs_result.leaves_visited,
            core_size=len(all_cores[0]),
            num_cores=len(all_cores),
        ),
    )


def make_expand(
    g1: Graph,
    g2: Graph,
    priority_idxs,
    enforce_core_core,
    max_connected_components: Optional[int],
    min_connected_component_size: int,
    filter_fxn: Callable[[Sequence[int]], bool],
    leaf_filter_fxn: Callable[[Sequence[int]], bool],
) -> Callable[[Node, int], Tuple[Sequence[Node], int]]:
    def satisfies_connected_components_constraints(node: Node) -> bool:
        if max_connected_components is not None or min_connected_component_size > 1:
            g1_mapped_nodes = {a1 for a1, a2 in enumerate(node.atom_map.a_to_b[: node.layer]) if a2 != UNMAPPED}

            if g1_mapped_nodes:
                # Nodes in g1 are visited in order, so nodes left to visit are [layer, layer + 1, ..., n - 1]
                g1_unvisited_nodes = set(range(node.layer, g1.n_vertices))
                if g1.mapping_incompatible_with_cc_constraints(
                    g1_mapped_nodes, g1_unvisited_nodes, max_connected_components, min_connected_component_size
                ):
                    return False

            g2_mapped_nodes = {a2 for a2, a1 in enumerate(node.atom_map.b_to_a) if a1 != UNMAPPED}

            if g2_mapped_nodes:
                # Nodes in g2 are visited in the order determined by priority_idxs. Nodes may be repeated in priority_idxs,
                # but we skip over nodes that have already been mapped.
                g2_unvisited_nodes = {
                    a2 for a2s in priority_idxs[node.layer :] for a2 in a2s if a2 not in g2_mapped_nodes
                }
                if g2.mapping_incompatible_with_cc_constraints(
                    g2_mapped_nodes, g2_unvisited_nodes, max_connected_components, min_connected_component_size
                ):
                    return False

        return True

    def expand(node: Node, best_num_edges: int) -> Tuple[List[Node], int]:
        if node.marcs.num_edges_upper_bound < best_num_edges:
            return [], best_num_edges

        if node.is_leaf:
            new_best_num_edges = (
                max(best_num_edges, node.marcs.num_edges_upper_bound)
                if leaf_filter_fxn(node.atom_map.a_to_b)
                else best_num_edges
            )
            return [], new_best_num_edges

        mapped_children = [
            child
            for new_v2 in priority_idxs[node.layer]
            if node.atom_map.b_to_a[new_v2] == UNMAPPED
            for child in [node.add(g1, g2, new_v2)]
            if (not enforce_core_core or _verify_map_preserves_core_edges(g1, g2, node.layer, new_v2, child.atom_map))
        ]

        unmapped_child = node.skip(g1, g2)

        children = [
            child
            for child in mapped_children + [unmapped_child]
            if child.marcs.num_edges_upper_bound >= best_num_edges
            if satisfies_connected_components_constraints(child)
            if filter_fxn(child.atom_map.a_to_b)
        ]

        return children, best_num_edges

    return expand
