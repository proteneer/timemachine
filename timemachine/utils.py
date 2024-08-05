from typing import Callable, Iterable, Iterator, Optional, Sequence, Tuple, TypeVar


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


def fair_product_2(xs: Sequence[A], ys: Sequence[B]) -> Iterable[Tuple[A, B]]:
    """Like py:func:`itertools.product`, except iterates over arguments in an unbiased ordering.

    Examples
    --------
    >>> import itertools
    >>> len(list(itertools.product([1, 2, 3], [1, 2, 3])))
    9

    >>> list(itertools.product([1, 2, 3], [1, 2, 3]))
    [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]

    >>> len(list(fair_product_2([1, 2, 3], [1, 2, 3])))
    9

    >>> list(fair_product_2([1, 2, 3], [1, 2, 3]))
    [(1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (1, 3), (3, 2), (2, 3), (3, 3)]
    """

    return (
        (xs[n - k], ys[k])
        for n in range(0, len(xs) + len(ys))
        for k in range(n + 1)
        if 0 <= n - k < len(xs) and 0 <= k < len(ys)
    )
