import hypothesis.strategies as st
import networkx as nx
import numpy as np
import pytest
from hypothesis import given

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


@pytest.mark.parametrize("seed", [2024, 2025])
@pytest.mark.parametrize("n", [1, 3, 10, 30, 100])
def test_connected_components(n, seed):
    """Ensure that the implementation of connected_components is correct by comparing it to the networkx reference
    implementation on $G_{np}$ random graphs in the regime with many connected components."""
    p = 1 / (n + 1)
    G = nx.fast_gnp_random_graph(n, p, seed, directed=False)

    rng = np.random.default_rng(seed)
    k = rng.choice(n)
    sources = rng.choice(n, size=(k,), replace=False)

    def connected_components_ref(G, sources):
        """Generate connected components.

        Adapted from the networkx version to additionally accept an iterable of sources, and only generate connected
        components containing at least one of the specified sources.
        """
        seen = set()
        for v in sources:
            if v not in seen:
                c = nx.node_connected_component(G, v)
                seen.update(c)
                yield c

    ccs = mcgregor.connected_components(G.neighbors, G.number_of_nodes(), sources)
    ccs_ref = connected_components_ref(G, sources)

    # Uncomment to ensure we get a good diversity of numbers of connected components
    # print(len(list(ccs)))
    # print(len(list(ccs_ref)))

    assert set(map(frozenset, ccs)) == set(map(frozenset, ccs_ref))
