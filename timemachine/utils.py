from collections.abc import Iterable, Iterator, Sequence
from typing import Callable, Optional, TypeVar


def batches(n: int, batch_size: int) -> Iterator[int]:
    assert n >= 0
    assert batch_size > 0
    quot, rem = divmod(n, batch_size)
    for _ in range(quot):
        yield batch_size
    if rem:
        yield rem


A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def not_ragged(xss: Sequence[Sequence]) -> bool:
    return all(len(xs) == len(xss[0]) for xs in xss)


def pairwise_transform_and_combine(xs: Iterable[A], f: Callable[[A], B], g: Callable[[B, B], C]) -> Iterator[C]:
    """Transforms an iterable using function `f` and combines adjacent transformed elements using function `g`."""

    prev_b: Optional[B] = None

    for a in xs:
        b = f(a)

        if prev_b is not None:
            yield g(prev_b, b)

        prev_b = b
