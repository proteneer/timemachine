from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Sequence, Type, TypeVar

from typing_extensions import Protocol

import timemachine.lib.potentials as gpu
import timemachine.potentials.chiral_restraints as ref_chiral
from timemachine.potentials import bonded as ref_bonded
from timemachine.potentials import nonbonded as ref_nonbonded
from timemachine.potentials import summed as ref_summed

Array = Any
Conf = Array
Params = Array
Box = Array
Lambda = float
PotentialFxn = Callable[[Conf, Params, Box, Lambda], float]


GenericPotential = TypeVar("GenericPotential", bound="Potential")
GpuPotential = TypeVar("GpuPotential", bound=gpu.CustomOpWrapper)
BondedGpuPotential = TypeVar("BondedGpuPotential", bound=gpu.BondedWrapper)


class Potential(Protocol[GpuPotential]):
    @classmethod
    def from_gpu(cls: Type[GenericPotential], p: GpuPotential) -> GenericPotential:
        ...

    def to_reference(self) -> PotentialFxn:
        ...

    def to_gpu(self) -> GpuPotential:
        ...


@dataclass
class Bonded(Generic[BondedGpuPotential]):
    idxs: Array
    lambda_mult: Optional[Array] = None
    lambda_offset: Optional[Array] = None

    @classmethod
    def from_gpu(cls, p: BondedGpuPotential):
        return cls(p.get_idxs(), p.get_lambda_mult(), p.get_lambda_offset())


@dataclass
class HarmonicBond(Bonded):
    def to_reference(self):
        def U(conf, params, box, lam):
            return ref_bonded.harmonic_bond(
                conf,
                params,
                box,
                lam,
                self.idxs,
                self.lambda_mult,
                self.lambda_offset,
            )

        return U

    def to_gpu(self):
        return gpu.HarmonicBond(self.idxs, self.lambda_mult, self.lambda_offset)


@dataclass
class HarmonicAngle(Bonded):
    def to_reference(self):
        def U(conf, params, box, lam):
            return ref_bonded.harmonic_angle(
                conf,
                params,
                box,
                lam,
                self.idxs,
                self.lambda_mult,
                self.lambda_offset,
            )

        return U

    def to_gpu(self):
        return gpu.HarmonicAngle(self.idxs, self.lambda_mult, self.lambda_offset)


@dataclass
class CentroidRestraint:
    group_a_idxs: Array
    group_b_idxs: Array
    kb: float
    b0: float

    @classmethod
    def from_gpu(cls, p: gpu.CentroidRestraint):
        return cls(p.get_a_idxs(), p.get_b_idxs(), p.get_kb(), p.get_b0())

    def to_reference(self):
        def U(conf, params, box, lam):
            return ref_bonded.centroid_restraint(
                conf, params, box, lam, self.group_a_idxs, self.group_b_idxs, self.kb, self.b0
            )

        return U

    def to_gpu(self):
        return gpu.CentroidRestraint(self.group_a_idxs, self.group_b_idxs, self.kb, self.b0)


@dataclass
class ChiralAtomRestraint:
    idxs: Array

    @classmethod
    def from_gpu(cls, p: gpu.ChiralAtomRestraint):
        return cls(p.get_idxs())

    def to_reference(self):
        def U(conf, params, box, lam):
            return ref_chiral.chiral_atom_restraint(conf, params, box, lam, self.idxs)

        return U

    def to_gpu(self):
        return gpu.ChiralAtomRestraint(self.idxs)


@dataclass
class ChiralBondRestraint:
    idxs: Array
    signs: Array

    @classmethod
    def from_gpu(cls, p: gpu.ChiralBondRestraint):
        return cls(p.get_idxs(), p.get_signs())

    def to_reference(self):
        def U(conf, params, box, lam):
            return ref_chiral.chiral_bond_restraint(conf, params, box, lam, self.idxs, self.signs)

        return U

    def to_gpu(self):
        return gpu.ChiralBondRestraint(self.idxs, self.signs)


@dataclass
class FlatBottomBond:
    idxs: Array

    @classmethod
    def from_gpu(cls, p: gpu.FlatBottomBond):
        return cls(p.get_idxs())

    def to_reference(self):
        def U(conf, params, box, _):
            return ref_bonded.flat_bottom_bond(conf, params, box, self.idxs)

        return U

    def to_gpu(self):
        return gpu.FlatBottomBond(self.idxs)


@dataclass
class PeriodicTorsion(Bonded):
    def to_reference(self):
        def U(conf, params, box, lam):
            return ref_bonded.periodic_torsion(
                conf,
                params,
                box,
                lam,
                self.idxs,
                self.lambda_mult,
                self.lambda_offset,
            )

        return U

    def to_gpu(self):
        return gpu.PeriodicTorsion(self.idxs, self.lambda_mult, self.lambda_offset)


@dataclass
class Nonbonded:
    exclusion_idxs: Array
    scale_factors: Array
    lambda_plane_idxs: Array
    lambda_offset_idxs: Array
    beta: float
    cutoff: float

    @classmethod
    def from_gpu(cls, p: gpu.Nonbonded):
        return cls(
            p.get_exclusion_idxs(),
            p.get_scale_factors(),
            p.get_lambda_plane_idxs(),
            p.get_lambda_offset_idxs(),
            p.get_beta(),
            p.get_cutoff(),
        )

    def to_reference(self):
        charge_rescale_mask, lj_rescale_mask = ref_nonbonded.convert_exclusions_to_rescale_masks(
            self.exclusion_idxs, self.scale_factors, len(self.lambda_plane_idxs)
        )

        def U(conf, params, box, lam):
            return ref_nonbonded.nonbonded_v3(
                conf,
                params,
                box,
                lam,
                charge_rescale_mask,
                lj_rescale_mask,
                self.beta,
                self.cutoff,
                self.lambda_plane_idxs,
                self.lambda_offset_idxs,
            )

        return U

    def to_gpu(self):
        return gpu.Nonbonded(
            self.exclusion_idxs,
            self.scale_factors,
            self.lambda_plane_idxs,
            self.lambda_offset_idxs,
            self.beta,
            self.cutoff,
        )


@dataclass
class SummedPotential:
    potentials: Sequence[Potential]
    params_init: Sequence[Array]

    @classmethod
    def from_gpu(cls, p: gpu.SummedPotential):
        potentials = [from_gpu(p_) for p_ in p._potentials]
        assert all(p_ is not None for p_ in potentials)
        return SummedPotential([p_ for p_ in potentials if p_ is not None], p._params_init)

    def to_reference(self):
        U_fns = [p.to_reference() for p in self.potentials]
        shapes = [ps.shape for ps in self.params_init]

        def U(conf, params, box, lam):
            return ref_summed.summed_potential(conf, params, box, lam, U_fns, shapes)

        return U

    def to_gpu(self):
        gpu_potentials = [p.to_gpu() for p in self.potentials]
        return gpu.SummedPotential(gpu_potentials, self.params_init)


def from_gpu(p: gpu.CustomOpWrapper) -> Optional[Potential]:
    if isinstance(p, gpu.HarmonicBond):
        return HarmonicBond.from_gpu(p)

    if isinstance(p, gpu.HarmonicAngle):
        return HarmonicAngle.from_gpu(p)

    if isinstance(p, gpu.CentroidRestraint):
        return CentroidRestraint.from_gpu(p)

    if isinstance(p, gpu.ChiralAtomRestraint):
        return ChiralAtomRestraint.from_gpu(p)

    if isinstance(p, gpu.ChiralBondRestraint):
        return ChiralBondRestraint.from_gpu(p)

    if isinstance(p, gpu.FlatBottomBond):
        return FlatBottomBond.from_gpu(p)

    if isinstance(p, gpu.PeriodicTorsion):
        return PeriodicTorsion.from_gpu(p)

    if isinstance(p, gpu.Nonbonded):
        return Nonbonded.from_gpu(p)
