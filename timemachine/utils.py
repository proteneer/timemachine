from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from warnings import warn


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
        import os
        from pathlib import Path

        warn(f"DEBUG_PATH: {module} {file_name} {path} {os.getcwd()} {list(Path('.').rglob(file_name))}")

        yield path
