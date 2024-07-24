import hypothesis.strategies as st
import networkx as nx
import numpy as np
from hypothesis import event, example, given, seed, settings

from timemachine.fe import mcgregor
from timemachine.fe.mcgregor import UNMAPPED, core_to_perm, perm_to_core


@st.composite
def permutations(draw):
    num_atoms_a = draw(st.integers(1, 30))
    num_atoms_b = draw(st.integers(1, 30))
    num_atoms_c = draw(st.integers(1, min(num_atoms_a, num_atoms_b)))

    indices_a = st.integers(0, num_atoms_a - 1)
    indices_b = st.integers(0, num_atoms_b - 1)
    mapping_a = draw(st.lists(indices_a, min_size=num_atoms_c, max_size=num_atoms_c, unique=True))
    mapping_b = draw(st.lists(indices_b, min_size=num_atoms_c, max_size=num_atoms_c, unique=True))

    mapping = {a: b for a, b in zip(mapping_a, mapping_b)}
    perm = [mapping.get(a, UNMAPPED) for a in range(num_atoms_a)]
    return perm


@given(permutations())
def test_core_to_perm_inv(perm):
    num_atoms_a = len(perm)
    assert core_to_perm(perm_to_core(perm), num_atoms_a) == perm
    np.testing.assert_array_equal(perm_to_core(core_to_perm(perm_to_core(perm), num_atoms_a)), perm_to_core(perm))


@st.composite
def graphs(draw, number_of_nodes_min, number_of_nodes_max):
    n = draw(st.integers(min_value=number_of_nodes_min, max_value=number_of_nodes_max))

    # for positive (negative) eps, graph is likely to be (dis)connected
    # https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model
    eps = -0.01
    p = (1.0 + eps) * np.log(n) / n

    seed = draw(st.integers())
    g = nx.fast_gnp_random_graph(n, p, seed=seed)

    # for reporting statistics with --hypothesis-show-statistics flag to pytest
    n_ccs = len(list(nx.connected_components(g)))
    if n_ccs == 1:
        event("n_ccs = 1")
    elif 1 < n_ccs < g.number_of_nodes():
        event("1 < n_ccs < n_nodes")
    elif n_ccs == g.number_of_nodes():
        event("n_ccs = n_nodes")
    else:
        assert False, "broken assumptions"

    return g


@st.composite
def graph_and_search_state(draw, number_of_nodes_max):
    g = draw(graphs(number_of_nodes_min=1, number_of_nodes_max=number_of_nodes_max))
    nodes = list(g.nodes)
    mapped_nodes = draw(st.lists(st.sampled_from(nodes), min_size=0, max_size=len(nodes), unique=True))
    unmapped_nodes = set(nodes) - set(mapped_nodes)
    unvisited_nodes = (
        draw(st.lists(st.sampled_from(list(unmapped_nodes)), min_size=0, max_size=len(unmapped_nodes), unique=True))
        if unmapped_nodes
        else set()
    )
    return g.number_of_nodes(), list(g.edges), set(mapped_nodes), set(unvisited_nodes)


@given(graph_and_search_state(30), st.one_of(st.integers(1, 30), st.none()), st.integers(1, 30))
@settings(max_examples=1_000)
@seed(2024)
@example((1, [], {0}, set()), None, 2)
def test_mapping_incompatible_with_cc_constraints(
    graph_and_search_state, max_connected_components, min_connected_component_size
):
    n_nodes, edges, mapped_nodes, unvisited_nodes = graph_and_search_state
    graph = mcgregor.Graph(n_nodes, edges)

    args = (mapped_nodes, unvisited_nodes, max_connected_components, min_connected_component_size)
    result_fast = graph.mapping_incompatible_with_cc_constraints_fast(*args)
    result_ref = graph.mapping_incompatible_with_cc_constraints_ref(*args)

    event(f"reference returned {result_ref}")

    assert result_fast == result_ref
