from typing import Callable, TypeVar

_T = TypeVar("_T")


def greedy_bisection_step(
    protocol: list[_T],
    local_cost: Callable[[_T, _T], float],
    make_intermediate: Callable[[_T, _T], _T],
) -> tuple[list[_T], tuple[list[float], int, _T]]:
    r"""Perform a single step of greedy bisection.

    Parameters
    ----------
    protocol: list
        Initial list of states

    local_cost: callable
        Function to use for computing the cost of adjacent pairs of states. The pair with the largest cost will be
        bisected. For free energy calculations using pair BAR, reasonable choices include BAR :math:`\Delta G` error and
        the inverse of the BAR overlap.

    make_intermediate: callable
        Function to use for instantiating a new state "between" a given pair. For states parameterized by a scalar lambda,
        a reasonable choice is the midpoint function. Note: since this function is polymorphic in the type of the state,
        this can in principle make use of other data attached to the state, e.g. samples for reweighting.

    Returns
    -------
    tuple of (list, tuple)
        Pair of (refined list of states, diagnostic info).
    """
    assert len(protocol) >= 2

    pairs = list(zip(protocol, protocol[1:]))
    costs = [local_cost(left, right) for left, right in pairs]
    pairs_by_cost = [(cost, left_idx, pair) for left_idx, (pair, cost) in enumerate(zip(pairs, costs))]
    _, left_idx, (left, right) = max(pairs_by_cost)
    new_state = make_intermediate(left, right)
    refined_protocol = copy_and_insert(protocol, left_idx + 1, new_state)

    return refined_protocol, (costs, left_idx, new_state)


def copy_and_insert(xs: list[_T], idx: int, x: _T) -> list[_T]:
    assert idx <= len(xs)
    xs_ = xs.copy()
    xs_.insert(idx, x)
    return xs_
