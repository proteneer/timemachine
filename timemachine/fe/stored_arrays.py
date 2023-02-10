import io
import shutil
import tempfile
import weakref
from itertools import count
from pathlib import Path
from typing import Iterator, List, NoReturn, Optional, Sequence, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from timemachine.parallel.client import AbstractFileClient


class StoredArrays(Sequence[NDArray]):
    def __init__(self):
        self._chunk_sizes: List[int] = []
        self._dir = Path(tempfile.mkdtemp())
        self._finalizer = weakref.finalize(self, shutil.rmtree, self._dir)

    def __iter__(self) -> Iterator[NDArray]:
        for chunk in self._chunks():
            for array in chunk:
                yield array

    def __len__(self) -> int:
        return sum(self._chunk_sizes)

    @overload
    def __getitem__(self, key: int) -> NDArray:
        ...

    @overload
    def __getitem__(self, key: slice) -> NoReturn:
        ...

    def __getitem__(self, key) -> NDArray:
        if isinstance(key, int):
            if key < -len(self) or key >= len(self):
                raise IndexError(f"index {key} out of range for sequence of length {len(self)}")
            if key < 0:
                key += len(self)
            for idx, size in enumerate(self._chunk_sizes):
                if key < size:
                    return np.load(self._get_chunk_path(idx))[key]
                key -= size
            assert False, "should not get here"
        elif isinstance(key, slice):
            raise NotImplementedError("slices are not implemented")
        else:
            raise ValueError("invalid subscript")

    def _get_chunk_path(self, idx: int) -> Path:
        return StoredArrays.get_chunk_path(self._dir, idx)

    def __eq__(self, other) -> bool:
        return self._chunk_sizes == other._chunk_sizes and all(
            np.array_equal(a, b, equal_nan=True) for a, b in zip(self, other)
        )

    def _chunks(self) -> Iterator[List[NDArray]]:
        for idx, _ in enumerate(self._chunk_sizes):
            yield np.load(self._get_chunk_path(idx))

    def extend(self, xs: Sequence[ArrayLike]):
        np.save(self._get_chunk_path(len(self._chunk_sizes)), np.array(xs))
        self._chunk_sizes.append(len(xs))

    @staticmethod
    def get_chunk_path(path: Path, idx: int) -> Path:
        return (path / str(idx)).with_suffix(".npy")

    def store(self, client: AbstractFileClient, prefix: Optional[Path] = None):
        for idx, chunk in enumerate(self._chunks()):
            serialized_array = serialize_array(np.array(chunk))
            path = StoredArrays.get_chunk_path(prefix or Path("."), idx)
            if client.exists(str(path)):
                raise FileExistsError(f"file already exists: {path}")
            client.store(str(path), serialized_array)

    @classmethod
    def load(cls, client: AbstractFileClient, prefix: Optional[Path] = None) -> "StoredArrays":
        sa = cls()
        for idx in count():
            path = cls.get_chunk_path(prefix or Path("."), idx)
            if client.exists(str(path)):
                bs = client.load(str(path))
                chunk = list(deserialize_array(bs))
                sa.extend(chunk)
            else:
                break
        return sa


def serialize_array(array: NDArray) -> bytes:
    fp = io.BytesIO()
    np.save(fp, array)
    fp.seek(0)
    return fp.read()


def deserialize_array(bs: bytes) -> NDArray:
    fp = io.BytesIO(bs)
    array = np.load(fp)
    return array
