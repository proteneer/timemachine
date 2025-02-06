import heapq
from collections.abc import Iterator, Sequence
from typing import Callable, TypeVar

Node = TypeVar("Node")
State = TypeVar("State")


def best_first(
    expand: Callable[[Node, State], tuple[Sequence[Node], State]],
    root: Node,
    initial_state: State,
) -> Iterator[Node]:
    """Generic search algorithm returning an iterator over nodes.

    The best-first strategy proceeds by maintaining a priority queue of active search nodes, and at each iteration
    yielding the best (minimal) node and adding its children to the queue.

    Parameters
    ----------
    expand : Callable
       Function from node and initial state to children and updated state. If the search is stateless, this function may
       ignore its second argument and return an arbitrary second element (e.g. None).

    root : Node
       Starting node

    initial_state : State
       Initial value of the global search state. If the search is stateless, can be an arbitrary value (e.g. None)
    """
    state = initial_state
    queue = [root]
    while queue:
        node = heapq.heappop(queue)
        children, state = expand(node, state)
        yield node
        for child in children:
            heapq.heappush(queue, child)
