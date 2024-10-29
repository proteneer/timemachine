import heapq
from typing import Callable, Iterator, Sequence, Tuple, TypeVar

N = TypeVar("N")
S = TypeVar("S")


def dfs(get_children: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[Tuple[N, S]]:
    state = initial_state

    def go(node: N) -> Iterator[Tuple[N, S]]:
        nonlocal state
        children, state = get_children(node, state)
        yield (node, state)
        for child in children:
            yield from go(child)

    return go(root)


def dfs_(get_children: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[N]:
    return (node for node, _ in dfs(get_children, root, initial_state))


def dfs_pure(get_children: Callable[[N], Sequence[N]], root: N) -> Iterator[N]:
    def get_children_(node: N, _: None) -> Tuple[Sequence[N], None]:
        return get_children(node), None

    return dfs_(get_children_, root, None)


def bfs(get_children: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[Tuple[N, S]]:
    state = initial_state
    queue = [root]
    while queue:
        node = queue.pop(0)
        children, state = get_children(node, state)
        yield node, state
        queue.extend(children)


def bfs_(get_children: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[N]:
    return (node for node, _ in bfs(get_children, root, initial_state))


def best_first(
    get_children: Callable[[N, S], Tuple[Sequence[N], S]],
    root: N,
    initial_state: S,
) -> Iterator[Tuple[N, S]]:
    state = initial_state
    queue = [root]
    while queue:
        node = heapq.heappop(queue)
        children, state = get_children(node, state)
        yield node, state
        for child in children:
            heapq.heappush(queue, child)


def best_first_(get_children: Callable[[N, S], Tuple[Sequence[N], S]], root: N, initial_state: S) -> Iterator[N]:
    return (node for node, _ in best_first(get_children, root, initial_state))
