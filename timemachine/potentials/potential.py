from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass
from typing import Any, Generic, Optional, Sequence, TypeVar, cast

import numpy as np
from jax import Array
from numpy.typing import NDArray

from timemachine.lib import custom_ops

from . import jax_interface
from .types import Box, Conf, Params

Precision = Any

_P = TypeVar("_P", bound="Potential")


@dataclass
class Potential(ABC):
    @abstractmethod
    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array: ...

    def bind(self: _P, params: Params) -> "BoundPotential[_P]":
        return BoundPotential(self, params)

    def to_gpu(self, precision: Precision) -> "GpuImplWrapper":
        ctor = getattr(custom_ops, self._custom_ops_class_name(precision))
        args = astuple(self)
        impl = ctor(*args)
        return GpuImplWrapper(impl)

    @classmethod
    def _custom_ops_class_name(cls, precision: Precision) -> str:
        suffix = get_custom_ops_class_name_suffix(precision)
        return f"{cls.__name__}_{suffix}"


@dataclass
class BoundPotential(Generic[_P]):
    potential: _P
    params: Params

    def __call__(self, conf: Conf, box: Optional[Box]) -> float | Array:
        return self.potential(conf, self.params, box)

    def to_gpu(self, precision: Precision) -> "BoundGpuImplWrapper":
        return self.potential.to_gpu(precision).bind(np.asarray(self.params))


@dataclass
class GpuImplWrapper:
    unbound_impl: custom_ops.Potential

    def __call__(self, conf: NDArray, params: NDArray, box: NDArray) -> float:
        res = jax_interface.call_unbound_impl(self.unbound_impl, conf, params, box)
        return cast(float, res)

    def bind(self, params: NDArray) -> "BoundGpuImplWrapper":
        return BoundGpuImplWrapper(custom_ops.BoundPotential(self.unbound_impl, params))


@dataclass
class BoundGpuImplWrapper:
    bound_impl: custom_ops.BoundPotential

    def __call__(self, conf: NDArray, box: NDArray) -> float:
        res = jax_interface.call_bound_impl(self.bound_impl, conf, box)
        return cast(float, res)


def get_custom_ops_class_name_suffix(precision: Precision):
    if precision == np.float32:
        return "f32"
    elif precision == np.float64:
        return "f64"
    else:
        raise ValueError("invalid precision")


def get_bound_potential_by_type(bps: Sequence[BoundPotential[_P]], pot_type: type[_P]) -> BoundPotential[_P]:
    """Given a list of bound potentials return the first bound potential with the matching potential type.

    Raises
    ------
        ValueError:
            Unable to find potential with the expected type
    """
    result: Optional[BoundPotential[_P]] = None
    for bp in bps:
        if isinstance(bp.potential, pot_type):
            result = bp
            break
    if result is None:
        raise ValueError(f"Unable to find potential of type: {pot_type}")
    return result


def get_potential_by_type(pots: Sequence[Potential], pot_type: type[_P]) -> _P:
    """Given a list of potentials return the first potential with the matching type.

    Raises
    ------
        ValueError:
            Unable to find potential with the expected type
    """
    result: Optional[_P] = None
    for pot in pots:
        if isinstance(pot, pot_type):
            result = pot
            break
    if result is None:
        raise ValueError(f"Unable to find potential of type: {pot_type}")
    return result
