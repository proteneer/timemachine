import heapq
from typing import Callable, Iterator, Sequence, Tuple, TypeAlias, TypeVar

Node = TypeVar("Node")
State = TypeVar("State")

ExpandFn: TypeAlias = Callable[[Node, State], Tuple[Sequence[Node], State]]
SearchAlgorithm: TypeAlias = Callable[[ExpandFn, Node, State], Iterator[Tuple[Node, State]]]
SearchAlgorithm_: TypeAlias = Callable[[ExpandFn, Node, State], Iterator[Node]]


def dfs(expand: ExpandFn[Node, State], root: Node, initial_state: State) -> Iterator[Tuple[Node, State]]:
    state = initial_state

    def go(node: Node) -> Iterator[Tuple[Node, State]]:
        nonlocal state
        children, state = expand(node, state)
        yield (node, state)
        for child in children:
            yield from go(child)

    return go(root)


def bfs(expand: ExpandFn[Node, State], root: Node, initial_state: State) -> Iterator[Tuple[Node, State]]:
    state = initial_state
    queue = [root]
    while queue:
        node = queue.pop(0)
        children, state = expand(node, state)
        yield node, state
        queue.extend(children)


def best_first(expand: ExpandFn[Node, State], root: Node, initial_state: State) -> Iterator[Tuple[Node, State]]:
    state = initial_state
    queue = [root]
    while queue:
        node = heapq.heappop(queue)
        children, state = expand(node, state)
        yield node, state
        for child in children:
            heapq.heappush(queue, child)


def drop_state(alg: SearchAlgorithm) -> SearchAlgorithm_:
    def sa(expand, root, initial_state):
        return (node for node, _ in alg(expand, root, initial_state))

    return sa


dfs_: SearchAlgorithm_ = drop_state(dfs)
bfs_: SearchAlgorithm_ = drop_state(bfs)
best_first_: SearchAlgorithm_ = drop_state(best_first)
