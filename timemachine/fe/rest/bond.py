from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from timemachine.fe.single_topology import canonicalize_bonds


@dataclass(frozen=True)
class CanonicalBond:
    i: int
    j: int

    _unsafe: bool = field(default=False, init=True, compare=False, repr=False)

    def __post_init__(self):
        if not self._unsafe:
            if self.i >= self.j:
                raise ValueError(f"{(self.i, self.j)} is not canonical")

    @classmethod
    def from_idxs(cls, i: int, j: int):
        return cls(i, j, _unsafe=True) if i < j else cls(j, i, _unsafe=True)

    def translate(self, a_to_b: Mapping[int, int] | Sequence[int] | NDArray[np.int_]):
        return CanonicalBond.from_idxs(a_to_b[self.i], a_to_b[self.j])


@dataclass(frozen=True)
class CanonicalAngle:
    i: int
    j: int
    k: int

    _unsafe: bool = field(default=False, init=True, compare=False, repr=False)

    def __post_init__(self):
        if not self._unsafe:
            if self.i >= self.k:
                raise ValueError(f"{(self.i, self.j, self.k)} is not canonical")

    @classmethod
    def from_idxs(cls, i: int, j: int, k: int):
        return cls(i, j, k, _unsafe=True) if i < k else cls(k, j, i, _unsafe=True)

    def translate(self, a_to_b: Mapping[int, int] | Sequence[int] | NDArray[np.int_]):
        return CanonicalAngle.from_idxs(a_to_b[self.i], a_to_b[self.j], a_to_b[self.k])


@dataclass(frozen=True)
class CanonicalTorsion:
    i: int
    j: int
    k: int
    l: int

    _unsafe: bool = field(default=False, init=True, compare=False, repr=False)

    def __post_init__(self):
        if not self._unsafe:
            if self.i >= self.l:
                raise ValueError(f"{(self.i, self.j, self.k, self.l)} is not canonical")

    @classmethod
    def from_idxs(cls, i: int, j: int, k: int, l: int):
        return cls(i, j, k, l, _unsafe=True)

    def translate(self, a_to_b: Mapping[int, int] | Sequence[int] | NDArray[np.int_]):
        return CanonicalTorsion.from_idxs(a_to_b[self.i], a_to_b[self.j], a_to_b[self.k], a_to_b[self.l])


@dataclass(frozen=True)
class CanonicalIxns:
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
        else:
            assert bonds.shape[1] == dim
        return cls(canonicalize_bonds(bonds), _unsafe=True)

    def translate(self, a_to_b: NDArray[np.int_]):
        return CanonicalIxns.from_idxs(a_to_b[self.idxs], self.idxs.shape[1])


def mkbond(i: int, j: int) -> CanonicalBond:
    return CanonicalBond.from_idxs(i, j)


def mkangle(i: int, j: int, k: int) -> CanonicalAngle:
    return CanonicalAngle.from_idxs(i, j, k)


def mktorsion(i: int, j: int, k: int, l: int) -> CanonicalTorsion:
    return CanonicalTorsion.from_idxs(i, j, k, l)


def mkbonds(idxs: ArrayLike) -> CanonicalIxns:
    return CanonicalIxns.from_idxs(idxs, 2)


def mkangles(idxs: ArrayLike) -> CanonicalIxns:
    return CanonicalIxns.from_idxs(idxs, 3)


def mktorsions(idxs: ArrayLike) -> CanonicalIxns:
    return CanonicalIxns.from_idxs(idxs, 4)
