from typing import Callable, List, Tuple, TypeVar

_T = TypeVar("_T")


def greedy_bisection_step(
    protocol: List[_T],
    local_cost: Callable[[_T, _T], float],
    make_intermediate: Callable[[_T, _T], _T],
) -> Tuple[List[_T], Tuple[List[float], int, _T]]:
    assert len(protocol) >= 2

    pairs = zip(protocol, protocol[1:])
    costs = [local_cost(left, right) for left, right in pairs]
    pairs_by_cost = [(cost, left_idx, pair) for left_idx, (pair, cost) in enumerate(zip(pairs, costs))]
    _, left_idx, (left, right) = max(pairs_by_cost)
    new_state = make_intermediate(left, right)
    refined_protocol = insert(protocol, left_idx + 1, new_state)

    return refined_protocol, (costs, left_idx, new_state)


def insert(xs: List[_T], idx: int, x: _T) -> List[_T]:
    assert idx <= len(xs)
    xs_ = xs.copy()
    xs_.insert(idx, x)
    return xs_
