from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


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


@contextmanager
def path_to_internal_file(module: str, file_name: str):
    with resources.as_file(resources.files(module).joinpath(file_name)) as path:
        yield path


def invert_permutation(p: ArrayLike) -> NDArray[Any]:
    """Given a permutation p of the integers 0..len(p)-1, returns an array q such that np.array_equal(arr[p][q], arr) is True."""
    p = np.asarray(p)
    q = np.empty_like(p)
    q[p] = np.arange(len(p))
    return q
