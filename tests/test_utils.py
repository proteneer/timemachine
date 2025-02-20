import numpy as np
import pytest
from hypothesis import given, seed
from hypothesis.strategies import integers, permutations
from numpy.typing import NDArray

from timemachine.utils import batches, invert_permutation

pytestmark = [pytest.mark.nocuda]


@given(integers(min_value=1))
@seed(2023)
def test_batches_of_nothing(batch_size):
    assert list(batches(0, batch_size)) == []


@given(integers(min_value=1, max_value=1000), integers(min_value=1))
@seed(2023)
def test_batches(n, batch_size):
    assert sum(batches(n, batch_size)) == n
    assert all(batch == batch_size for batch in list(batches(n, batch_size))[:-1])
    *_, last = batches(n, batch_size)
    assert 0 < last <= batch_size


@given(integers(min_value=1, max_value=100).flatmap(lambda n: permutations(range(n))).map(np.asarray))
@seed(2025)
def test_invert_permutation_property(p: NDArray):
    q = invert_permutation(p)
    arr = np.arange(len(p))
    np.testing.assert_array_equal(arr[p][q], arr)
