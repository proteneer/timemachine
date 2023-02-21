from typing import Callable, List, TypeVar

_T = TypeVar("_T")


def greedy_bisection_step(
    protocol: List[_T],
    local_cost: Callable[[_T, _T], float],
    make_intermediate: Callable[[_T, _T], _T],
) -> List[_T]:
    assert len(protocol) >= 2
    adjacent_pairs = zip(protocol, protocol[1:])
    _, s1_idx, (s1, s2) = max((local_cost(s1, s2), idx, (s1, s2)) for idx, (s1, s2) in enumerate(adjacent_pairs))
    new_state = make_intermediate(s1, s2)
    refined_protocol = insert(protocol, s1_idx + 1, new_state)  # insert to the right of s1
    return refined_protocol


def insert(xs: List[_T], idx: int, x: _T) -> List[_T]:
    assert idx <= len(xs)
    xs_ = xs.copy()
    xs_.insert(idx, x)
    return xs_
