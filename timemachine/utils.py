from collections.abc import Iterator, Sequence


def batches(n: int, batch_size: int) -> Iterator[int]:
    assert n >= 0
    assert batch_size > 0
    quot, rem = divmod(n, batch_size)
    for _ in range(quot):
        yield batch_size
    if rem:
        yield rem


def not_ragged(xss: Sequence[Sequence]) -> bool:
    return all(len(xs) == len(xss[0]) for xs in xss)
