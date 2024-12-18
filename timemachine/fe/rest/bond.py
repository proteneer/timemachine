from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

TCanonicalIxn = TypeVar("TCanonicalIxn", bound="CanonicalIxn")  # in Python 3.11, can remove this in favor of Self


@dataclass(frozen=True)
class CanonicalIxn(ABC):
    @abstractmethod
    def map(self: TCanonicalIxn, f: Callable[[int], int]) -> TCanonicalIxn:
        ...

    def translate(self: TCanonicalIxn, a_to_b: Mapping[int, int] | Sequence[int] | NDArray[np.int32]) -> TCanonicalIxn:
        return self.map(lambda i: a_to_b[i])


@dataclass(frozen=True)
class CanonicalBond(CanonicalIxn):
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

    def map(self, f: Callable[[int], int]) -> "CanonicalBond":
        return CanonicalBond.from_idxs(f(self.i), f(self.j))


@dataclass(frozen=True)
class CanonicalAngle(CanonicalIxn):
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

    def map(self, f: Callable[[int], int]) -> "CanonicalAngle":
        return CanonicalAngle.from_idxs(f(self.i), f(self.j), f(self.k))


@dataclass(frozen=True)
class CanonicalProper(CanonicalIxn):
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

    def map(self, f: Callable[[int], int]) -> "CanonicalProper":
        return CanonicalProper.from_idxs(f(self.i), f(self.j), f(self.k), f(self.l))


def mkbond(i: int, j: int) -> CanonicalBond:
    return CanonicalBond.from_idxs(i, j)


def mkangle(i: int, j: int, k: int) -> CanonicalAngle:
    return CanonicalAngle.from_idxs(i, j, k)


def mkproper(i: int, j: int, k: int, l: int) -> CanonicalProper:
    return CanonicalProper.from_idxs(i, j, k, l)
