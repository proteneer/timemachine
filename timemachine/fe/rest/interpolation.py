"""This module defines interpolation functions specialized to the domain [0, 1]"""

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray


class InterpolationFxn(Protocol):
    @property
    def src(self) -> ArrayLike:
        ...

    @property
    def dst(self) -> ArrayLike:
        ...

    def __call__(self, x: ArrayLike) -> NDArray:
        ...


@dataclass(frozen=True)
class Linear:
    """Linear interpolation on [0, 1].

    Parameters
    ----------
    src : ArrayLike
        value at 0
    dst : ArrayLike
        value at 1

    Returns
    -------
    NDArray
    """

    src: ArrayLike
    dst: ArrayLike

    def __call__(self, x: ArrayLike) -> NDArray:
        src = np.asarray(self.src)
        dst = np.asarray(self.dst)
        x = np.asarray(x)

        a = dst - src
        b = src
        return a * x + b


@dataclass(frozen=True)
class Quadratic:
    """Quadratic interpolation on [0, 1], specialized to assume b^2 - 4 a c = 0 (i.e. a single root).

    Parameters
    ----------
    src : ArrayLike
        value at 0
    dst : ArrayLike
        value at 1

    Returns
    -------
    NDArray
    """

    src: ArrayLike
    dst: ArrayLike

    def __post_init__(self):
        assert np.all(self.src != self.dst)

    def __call__(self, x: ArrayLike) -> NDArray:
        src = np.asarray(self.src)
        dst = np.asarray(self.dst)
        x = np.asarray(x)

        a = src + dst - 2.0 * np.sqrt(src * dst)
        b = (src + np.sqrt(src * dst)) / (dst - src)

        # special-case 0 and 1 to avoid issues with roundoff error
        return np.where(
            x == 0.0,
            self.src,
            np.where(
                x == 1.0,
                self.dst,
                a * (x + b) ** 2,
            ),
        )


@dataclass(frozen=True)
class Exponential:
    """Exponential interpolation on [0, 1], specialized to assume f(x -> -inf) = 0.

    Parameters
    ----------
    src : ArrayLike
        value at 0
    dst : ArrayLike
        value at 1

    Returns
    -------
    NDArray
    """

    src: ArrayLike
    dst: ArrayLike

    def __call__(self, x: ArrayLike) -> NDArray:
        src = np.asarray(self.src)
        dst = np.asarray(self.dst)
        x = np.asarray(x)

        a = src
        b = np.log(dst / src)
        return a * np.exp(b * x)


F = TypeVar("F", bound=InterpolationFxn)


@dataclass(frozen=True)
class Symmetric(Generic[F]):
    """Symmetric interpolation on [0, 1] derived from an arbitrary interpolation function f.

    The symmetry property is Symmetric(f)(x) == Symmetric(f)(1 - x).

    Parameters
    ----------
    src : ArrayLike
        value at 0
    dst : ArrayLike
        value at 1

    Returns
    -------
    NDArray
    """

    f: F

    @property
    def src(self):
        return self.f.src

    @property
    def dst(self):
        return self.f.src

    def get_value(self, x: ArrayLike) -> NDArray:
        x = np.asarray(x)
        return np.where(
            x < 0.5,
            self.f(2.0 * x),
            self.f(2.0 * (1.0 - x)),
        )


def plot_interpolation_fxn(f: InterpolationFxn):
    x = np.linspace(0.0, 1.0, 100)
    return plt.plot(x, f(x), label=str(f))


InterpolationFxnName = Literal["linear", "quadratic", "exponential"]


def get_interpolation_fxn(name: InterpolationFxnName, src: ArrayLike, dst: ArrayLike) -> InterpolationFxn:
    match name:
        case "linear":
            return Linear(src, dst)
        case "quadratic":
            return Quadratic(src, dst)
        case "exponential":
            return Exponential(src, dst)
