import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Iterator, NoReturn, Sequence, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray


class StoredArrays(Sequence[NDArray]):
    def __init__(self):
        self._block_sizes = []
        self._path = Path(tempfile.mkdtemp())
        self._finalizer = weakref.finalize(self, shutil.rmtree, self._path)

    def extend(self, xs: Sequence[ArrayLike]):
        np.save(self._get_path(len(self._block_sizes)), np.array(xs))
        self._block_sizes.append(len(xs))

    def __iter__(self) -> Iterator[NDArray]:
        for block, _ in enumerate(self._block_sizes):
            for x in np.load(self._get_path(block)):
                yield x

    def __len__(self) -> int:
        return sum(self._block_sizes)

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
            for block, size in enumerate(self._block_sizes):
                if key < size:
                    return np.load(self._get_path(block))[key]
                key -= size
            assert False, "should not get here"
        elif isinstance(key, slice):
            raise NotImplementedError("slices are not implemented")
        else:
            raise ValueError("invalid subscript")

    def _get_path(self, block: int):
        return (self._path / f"{block}").with_suffix(".npy")
