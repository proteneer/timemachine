from typing import Iterable


def batches(n: int, batch_size: int) -> Iterable[int]:
    assert n >= 0
    assert batch_size > 0
    quot, rem = divmod(n, batch_size)
    for _ in range(quot):
        yield batch_size
    if rem:
        yield rem
