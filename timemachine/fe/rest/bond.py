from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CanonicalBond:
    i: int
    j: int

    _unsafe: bool = field(default=False, init=True, compare=False, repr=False)

    def __post_init__(self):
        if self._unsafe:
            return
        if self.i >= self.j:
            raise ValueError(f"{(self.i, self.j)} is not canonical")

    @classmethod
    def from_idxs(cls, i: int, j: int):
        return cls(i, j, _unsafe=True) if i < j else cls(j, i, _unsafe=True)

    def translate(self, a_to_b: Mapping[int, int] | Sequence[int] | NDArray[np.int32]) -> "CanonicalBond":
        return CanonicalBond.from_idxs(a_to_b[self.i], a_to_b[self.j])


@dataclass(frozen=True)
class CanonicalAngle:
    i: int
    j: int
    k: int

    _unsafe: bool = field(default=False, init=True, compare=False, repr=False)

    def __post_init__(self):
        if self._unsafe:
            return
        if self.i >= self.k:
            raise ValueError(f"{(self.i, self.j, self.k)} is not canonical")

    @classmethod
    def from_idxs(cls, i: int, j: int, k: int):
        return cls(i, j, k, _unsafe=True) if i < k else cls(k, j, i, _unsafe=True)

    def translate(self, a_to_b: Mapping[int, int] | Sequence[int] | NDArray[np.int32]) -> "CanonicalAngle":
        return CanonicalAngle.from_idxs(a_to_b[self.i], a_to_b[self.j], a_to_b[self.k])


@dataclass(frozen=True)
class CanonicalProper:
    i: int
    j: int
    k: int
    l: int

    _unsafe: bool = field(default=False, init=True, compare=False, repr=False)

    def __post_init__(self):
        if self._unsafe:
            return
        if self.i >= self.l:
            raise ValueError(f"{(self.i, self.j, self.k, self.l)} is not canonical")

    @classmethod
    def from_idxs(cls, i: int, j: int, k: int, l: int):
        return cls(i, j, k, l, _unsafe=True) if i < l else cls(l, k, j, i, _unsafe=True)

    def translate(self, a_to_b: Mapping[int, int] | Sequence[int] | NDArray[np.int32]) -> "CanonicalProper":
        return CanonicalProper.from_idxs(a_to_b[self.i], a_to_b[self.j], a_to_b[self.k], a_to_b[self.l])


def mkbond(i: int, j: int) -> CanonicalBond:
    return CanonicalBond.from_idxs(i, j)


def mkangle(i: int, j: int, k: int) -> CanonicalAngle:
    return CanonicalAngle.from_idxs(i, j, k)


def mkproper(i: int, j: int, k: int, l: int) -> CanonicalProper:
    return CanonicalProper.from_idxs(i, j, k, l)
