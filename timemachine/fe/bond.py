from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from timemachine.fe.single_topology import canonicalize_bonds


@dataclass(frozen=True)
class CanonicalBond:
    src_idx: int
    dst_idx: int

    _unsafe: bool = field(default=False, init=True, repr=False)

    def __post_init__(self):
        if not self._unsafe:
            if self.src_idx >= self.dst_idx:
                raise ValueError(f"({self.src_idx}, {self.dst_idx}) is not canonical")

    @classmethod
    def from_tuple(cls, idxs: Tuple[int, int]):
        src_idx, dst_idx = idxs
        return cls(src_idx, dst_idx, _unsafe=True) if src_idx < dst_idx else cls(dst_idx, src_idx, _unsafe=True)


@dataclass(frozen=True)
class CanonicalAngle:
    i: int
    j: int
    k: int

    _unsafe: bool = field(default=False, init=True)

    def __post_init__(self):
        if not self._unsafe:
            if self.i >= self.k:
                raise ValueError(f"({self.i}, {self.j}, {self.k}) is not canonical")

    @classmethod
    def from_idxs(cls, i, j, k):
        return cls(i, j, k, _unsafe=True) if i < k else cls(k, j, i, _unsafe=True)


@dataclass(frozen=True)
class CanonicalBonds:
    idxs: NDArray[np.int32]

    _unsafe: bool = field(default=False, init=True)

    def __post_init__(self):
        if np.any(self.idxs[:, 0] >= self.idxs[:, -1]):
            raise ValueError("bonds are not canonical")

    @classmethod
    def from_idxs(cls, bonds: ArrayLike, dim: int):
        bonds = np.asarray(bonds)
        if len(bonds) == 0:
            bonds = bonds.reshape(0, dim)
        return cls(canonicalize_bonds(bonds), _unsafe=True)
