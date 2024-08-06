from typing import Sequence

import hypothesis.strategies as st
from hypothesis import given, seed

from timemachine.utils import fair_product_2


@given(
    st.integers().map(lambda n: [n]),
    st.integers().map(lambda n: [n]),
)
@seed(2024)
def test_fair_product_singletons(xs, ys):
    assert list(fair_product_2(xs, ys)) == [(xs[0], ys[0])]


@given(
    st.lists(st.just(()), max_size=5),
    st.lists(st.just(()), max_size=5),
)
@seed(2024)
def test_fair_product_length(xs, ys):
    assert len(list(fair_product_2(xs, ys))) == len(xs) * len(ys)


def is_nondecreasing(xs: Sequence) -> bool:
    return all(x1 <= x2 for x1, x2 in zip(xs, xs[1:]))


@given(
    st.integers(0, 5).map(lambda n: list(range(n))),
    st.integers(0, 5).map(lambda n: list(range(n))),
)
@seed(2024)
def test_fair_product_ordering(xs, ys):
    manhattan_dists = [x + y for x, y in fair_product_2(xs, ys)]
    assert is_nondecreasing(manhattan_dists)
