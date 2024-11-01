import heapq
from typing import Callable, Iterator, Sequence, TypeVar

Node = TypeVar("Node")
State = TypeVar("State")


def best_first(
    expand: Callable[[Node, State], tuple[Sequence[Node], State]], root: Node, initial_state: State
) -> Iterator[Node]:
    state = initial_state
    queue = [root]
    while queue:
        node = heapq.heappop(queue)
        children, state = expand(node, state)
        yield node
        for child in children:
            heapq.heappush(queue, child)
