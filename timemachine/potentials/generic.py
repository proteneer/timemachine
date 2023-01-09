from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Sequence, Type, TypeVar

import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol, runtime_checkable

import timemachine.lib.potentials as gpu
import timemachine.potentials.chiral_restraints as ref_chiral
from timemachine.potentials import bonded as ref_bonded
from timemachine.potentials import nonbonded as ref_nonbonded
from timemachine.potentials import summed as ref_summed

Array = Any
Conf = Array
Params = Array
Box = Array
PotentialFxn = Callable[[Conf, Params, Box], float]


GenericPotential = TypeVar("GenericPotential", bound="Potential")
GpuPotential = TypeVar("GpuPotential", bound=gpu.CustomOpWrapper)
BondedGpuPotential = TypeVar("BondedGpuPotential", bound=gpu.BondedWrapper)


@runtime_checkable
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

    @classmethod
    def from_gpu(cls, p: BondedGpuPotential):
        return cls(p.get_idxs())


@dataclass
class HarmonicBond(Bonded):
    def to_reference(self):
        def U(conf, params, box):
            return ref_bonded.harmonic_bond(
                conf,
                params,
                box,
                self.idxs,
            )

        return U

    def to_gpu(self):
        return gpu.HarmonicBond(self.idxs)


@dataclass
class HarmonicAngle(Bonded):
    def to_reference(self):
        def U(conf, params, box):
            return ref_bonded.harmonic_angle(
                conf,
                params,
                box,
                self.idxs,
            )

        return U

    def to_gpu(self):
        return gpu.HarmonicAngle(self.idxs)


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
        def U(conf, params, box):
            return ref_bonded.centroid_restraint(
                conf, params, box, self.group_a_idxs, self.group_b_idxs, self.kb, self.b0
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
        def U(conf, params, box):
            return ref_chiral.chiral_atom_restraint(conf, params, box, self.idxs)

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
        def U(conf, params, box):
            return ref_chiral.chiral_bond_restraint(conf, params, box, self.idxs, self.signs)

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
        def U(conf, params, box):
            return ref_bonded.flat_bottom_bond(conf, params, box, self.idxs)

        return U

    def to_gpu(self):
        return gpu.FlatBottomBond(self.idxs)


@dataclass
class PeriodicTorsion(Bonded):
    def to_reference(self):
        def U(conf, params, box):
            return ref_bonded.periodic_torsion(
                conf,
                params,
                box,
                self.idxs,
            )

        return U

    def to_gpu(self):
        return gpu.PeriodicTorsion(self.idxs)


@dataclass
class Nonbonded:
    num_atoms: int
    exclusion_idxs: Array
    scale_factors: Array
    beta: float
    cutoff: float

    @classmethod
    def from_gpu(cls, p: gpu.Nonbonded):
        return cls(
            p.get_num_atoms(),
            p.get_exclusion_idxs(),
            p.get_scale_factors(),
            p.get_beta(),
            p.get_cutoff(),
        )

    def to_reference(self):
        charge_rescale_mask, lj_rescale_mask = ref_nonbonded.convert_exclusions_to_rescale_masks(
            self.exclusion_idxs, self.scale_factors, self.num_atoms
        )

        def U(conf, params, box):
            return ref_nonbonded.nonbonded(
                conf,
                params,
                box,
                charge_rescale_mask,
                lj_rescale_mask,
                self.beta,
                self.cutoff,
                runtime_validate=False,  # needed for this to be JAX-transformable
            )

        return U

    def to_gpu(self):
        return gpu.Nonbonded(
            self.num_atoms,
            self.exclusion_idxs,
            self.scale_factors,
            self.beta,
            self.cutoff,
        )


@dataclass
class NonbondedAllPairs:
    num_atoms: int
    beta: float
    cutoff: float
    atom_idxs: Optional[Array] = None

    @classmethod
    def from_gpu(cls, p: gpu.NonbondedAllPairs):
        return cls(
            p.get_num_atoms(),
            p.get_beta(),
            p.get_cutoff(),
            p.get_atom_idxs(),
        )

    def to_reference(self):
        s = self.atom_idxs if self.atom_idxs is not None else slice(None)

        def U(conf, params, box):
            conf = conf[s, :]
            num_atoms, _ = conf.shape
            no_rescale = jnp.ones((num_atoms, num_atoms))
            return ref_nonbonded.nonbonded(
                conf,
                params[s, :],
                box,
                no_rescale,
                no_rescale,
                self.beta,
                self.cutoff,
                runtime_validate=False,  # needed for this to be JAX-transformable
            )

        return U

    def to_gpu(self):
        return gpu.NonbondedAllPairs(
            self.num_atoms,
            self.beta,
            self.cutoff,
            self.atom_idxs,
        )


@dataclass
class NonbondedInteractionGroup:
    num_atoms: int
    row_atom_idxs: Array
    beta: float
    cutoff: float

    @classmethod
    def from_gpu(cls, p: gpu.NonbondedInteractionGroup):
        return cls(
            p.get_num_atoms(),
            p.get_row_atom_idxs(),
            p.get_beta(),
            p.get_cutoff(),
        )

    def to_reference(self):
        def U(conf, params, box):
            num_atoms, _ = conf.shape

            vdW, electrostatics = ref_nonbonded.nonbonded_interaction_groups(
                conf,
                params,
                box,
                self.row_atom_idxs,
                np.setdiff1d(jnp.arange(num_atoms), self.row_atom_idxs),
                self.beta,
                self.cutoff,
            )
            return jnp.sum(vdW + electrostatics)

        return U

    def to_gpu(self):
        return gpu.NonbondedInteractionGroup(
            self.num_atoms,
            self.row_atom_idxs,
            self.beta,
            self.cutoff,
        )


@dataclass
class NonbondedPairList:
    idxs: Array
    rescale_mask: Array
    beta: float
    cutoff: float

    @classmethod
    def from_gpu(cls, p: gpu.NonbondedPairList):
        return cls(
            p.get_idxs(),
            p.get_rescale_mask(),
            p.get_beta(),
            p.get_cutoff(),
        )

    def to_reference(self):
        def U(conf, params, box):
            vdW, electrostatics = ref_nonbonded.nonbonded_on_specific_pairs(
                conf,
                params,
                box,
                self.idxs,
                self.beta,
                self.cutoff,
                self.rescale_mask,
            )
            return vdW.sum() + electrostatics.sum()

        return U

    def to_gpu(self):
        return gpu.NonbondedPairList(
            self.idxs,
            self.rescale_mask,
            self.beta,
            self.cutoff,
        )


@dataclass
class NonbondedPairListPrecomputed:
    idxs: Array
    beta: float
    cutoff: float

    @classmethod
    def from_gpu(cls, p: gpu.NonbondedPairListPrecomputed):
        return cls(p.get_idxs(), p.get_beta(), p.get_cutoff())

    def to_reference(self):
        def U(conf, params, box):
            vdW, electrostatics = ref_nonbonded.nonbonded_on_precomputed_pairs(
                conf,
                params,
                box,
                self.idxs,
                self.beta,
                self.cutoff,
            )
            return vdW.sum() + electrostatics.sum()

        return U

    def to_gpu(self):
        return gpu.NonbondedPairListPrecomputed(self.idxs, self.beta, self.cutoff)


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

        def U(conf, params, box):
            return ref_summed.summed_potential(conf, params, box, U_fns, shapes)

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

    if isinstance(p, gpu.NonbondedPairListPrecomputed):
        return NonbondedPairListPrecomputed.from_gpu(p)

    if isinstance(p, gpu.SummedPotential):
        return SummedPotential.from_gpu(p)

    return None
