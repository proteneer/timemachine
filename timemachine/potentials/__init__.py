from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, Protocol, Sequence

import importlib_resources as resources
import jax

from timemachine.lib import custom_ops
from timemachine.potentials import nonbonded

Array = Any  # see https://github.com/google/jax/issues/943


class Precision(Enum):
    F64 = 64
    F32 = 32

    def __str__(self):
        return "f64" if self == Precision.F64 else "f32"


class EnergyAndGradients(NamedTuple):
    du_dx: Array
    du_dp: Array
    du_dl: float
    u: float


class UnboundImpl(Protocol):
    def execute(self, coords: Array, params: Array, box: Array, lam: float) -> EnergyAndGradients:
        ...

    def execute_selective(
        self,
        coords: Array,
        params: Array,
        box: Array,
        lam: float,
        compute_du_dx: bool,
        compute_du_dp: bool,
        compute_du_dl: bool,
        compute_u: bool,
    ) -> EnergyAndGradients:
        ...


class BoundImpl(Protocol):
    def execute(self, coords: Array, box: Array, lam: float) -> EnergyAndGradients:
        ...


class Potential(Protocol):
    def impl_reference(self) -> UnboundImpl:
        ...

    def impl_cuda(self, precision: Precision) -> UnboundImpl:
        ...

    def impl_cuda_bound(self, precision: Precision, params: Array) -> BoundImpl:
        ...


class HasCustomOp(ABC):
    @classmethod
    def _custom_op_ctor(cls, precision: Precision, suffix: Sequence[str]):
        class_name = "_".join([f"{cls.__name__}", str(precision)] + list(suffix))
        return getattr(custom_ops, class_name)

    @abstractmethod
    def impl_cuda(self, precision: Precision) -> UnboundImpl:
        ...

    def impl_cuda_bound(self, precision: Precision, params: Array) -> BoundImpl:
        return custom_ops.BoundPotential(self.impl_cuda(precision), params)


@dataclass(frozen=True)
class ReferenceUnboundImpl:

    u_fxn: Callable[[Array, Array, Array, float], float]

    def execute(self, coords: Array, params: Array, box: Array, lam: float) -> EnergyAndGradients:
        u, u_grad = jax.value_and_grad(self.u_fxn, argnums=(0, 1, 3))(coords, params, box, lam)
        return EnergyAndGradients(du_dx=u_grad[0], du_dp=u_grad[1], du_dl=u_grad[2], u=u)

    def execute_selective(
        self,
        coords: Array,
        params: Array,
        box: Array,
        lam: float,
        compute_du_dx: bool,
        compute_du_dp: bool,
        compute_du_dl: bool,
        compute_u: bool,
    ):
        del coords, params, box, lam, compute_du_dx, compute_du_dp, compute_du_dl, compute_u
        raise NotImplementedError("execute_selective not implemented on reference potentials")


@dataclass
class NonbondedAllPairs(HasCustomOp):
    lambda_plane_idxs: Array
    lambda_offset_idxs: Array
    beta: float
    cutoff: float
    atom_idxs: Optional[Array] = None
    interpolated: bool = False

    def impl_reference(self) -> UnboundImpl:
        subset_slice = self.atom_idxs or slice(None)

        def reference_impl(coords: Array, params: Array, box: Array, lam: float) -> EnergyAndGradients:
            coords_subset = coords[subset_slice]
            num_atoms, _ = coords_subset.shape
            rescale_mask = nonbonded.convert_exclusions_to_rescale_masks([], [], num_atoms)

            return nonbonded.nonbonded_v3(
                coords_subset,
                params[subset_slice, :],
                box,
                lam,
                charge_rescale_mask=rescale_mask,
                lj_rescale_mask=rescale_mask,
                beta=self.beta,
                cutoff=self.cutoff,
                lambda_plane_idxs=self.lambda_plane_idxs[subset_slice],
                lambda_offset_idxs=self.lambda_offset_idxs[subset_slice],
            )

        return ReferenceUnboundImpl(nonbonded.interpolated(reference_impl) if self.interpolated else reference_impl)

    def impl_cuda(self, precision: Precision) -> UnboundImpl:
        with resources.files("timemachine.cpp.src.kernels") as path:
            return self._custom_op_ctor(precision, ["interpolated"] if self.interpolated else [])(
                str(path),
                self.lambda_plane_idxs,
                self.lambda_offset_idxs,
                self.beta,
                self.cutoff,
                self.atom_idxs,
            )
