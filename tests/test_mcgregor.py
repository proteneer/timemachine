import hypothesis.strategies as st
import numpy as np
from hypothesis import given

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
