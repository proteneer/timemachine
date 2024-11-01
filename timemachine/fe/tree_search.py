import heapq
from typing import Callable, Iterator, Sequence, TypeVar

Node = TypeVar("Node")
State = TypeVar("State")


def best_first(expand: Callable[[Node], Sequence[Node]], root: Node) -> Iterator[Node]:
    """Generic search algorithm returning an iterator over nodes.

    The best-first strategy proceeds by maintaining a priority queue of active search nodes, and at each iteration
    yielding the best (minimal) node and adding its children to the queue.

    Parameters
    ----------
    expand : Callable
       Function returning the children of a given search node

    root : Node
       Starting node
    """
    queue = [root]
    while queue:
        node = heapq.heappop(queue)
        children = expand(node)
        yield node
        for child in children:
            heapq.heappush(queue, child)


def best_first_stateful(
    expand: Callable[[Node, State], tuple[Sequence[Node], State]], root: Node, initial_state: State
) -> Iterator[Node]:
    """Stateful variant of :py:func:`best_first`.

    Compared with the latter, this accepts an augmented `expand` function that updates the global search state in
    addition to producing the children of a given node, and an initial state.

    Parameters
    ----------
    expand : Callable
       Function returning children and updated state, given a node and initial state

    root : Node
       Starting node

    initial_state : State
       Initial value of the global search state
    """
    state = initial_state
    queue = [root]
    while queue:
        node = heapq.heappop(queue)
        children, state = expand(node, state)
        yield node
        for child in children:
            heapq.heappush(queue, child)
