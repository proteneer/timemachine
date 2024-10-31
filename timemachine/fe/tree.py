import heapq
from typing import Callable, Iterator, Sequence, Tuple, TypeVar

N = TypeVar("N")
S = TypeVar("S")


def dfs(expand: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[Tuple[N, S]]:
    state = initial_state

    def go(node: N) -> Iterator[Tuple[N, S]]:
        nonlocal state
        children, state = expand(node, state)
        yield (node, state)
        for child in children:
            yield from go(child)

    return go(root)


def dfs_(expand: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[N]:
    return (node for node, _ in dfs(expand, root, initial_state))


def dfs_pure(expand: Callable[[N], Sequence[N]], root: N) -> Iterator[N]:
    def expand_(node: N, _: None) -> Tuple[Sequence[N], None]:
        return expand(node), None

    return dfs_(expand_, root, None)


def bfs(expand: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[Tuple[N, S]]:
    state = initial_state
    queue = [root]
    while queue:
        node = queue.pop(0)
        children, state = expand(node, state)
        yield node, state
        queue.extend(children)


def bfs_(expand: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[N]:
    return (node for node, _ in bfs(expand, root, initial_state))


def best_first(expand: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[Tuple[N, S]]:
    state = initial_state
    queue = [root]
    while queue:
        node = heapq.heappop(queue)
        children, state = expand(node, state)
        yield node, state
        for child in children:
            heapq.heappush(queue, child)


def best_first_(expand: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[N]:
    return (node for node, _ in best_first(expand, root, initial_state))
