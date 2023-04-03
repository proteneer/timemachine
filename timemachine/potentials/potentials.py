from dataclasses import dataclass
from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from timemachine.lib import custom_ops

from . import bonded, bonded_stable, chiral_restraints, nonbonded, summed
from .potential import GpuImplWrapper, Potential, Precision, get_custom_ops_class_name_suffix
from .types import Box, Conf, Params


@dataclass
class HarmonicBond(Potential):
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        return bonded.harmonic_bond(conf, params, box, self.idxs)


@dataclass
class HarmonicAngle(Potential):
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        return bonded.harmonic_angle(conf, params, box, self.idxs)


@dataclass
class HarmonicAngleStable(Potential):
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, _: Optional[Box]) -> float | Array:
        return bonded_stable.harmonic_angle_stable(conf, params, self.idxs)


@dataclass
class CentroidRestraint(Potential):
    group_a_idxs: NDArray[np.int32]
    group_b_idxs: NDArray[np.int32]
    kb: float
    b0: float

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        return bonded.centroid_restraint(conf, params, box, self.group_a_idxs, self.group_b_idxs, self.kb, self.b0)


@dataclass
class ChiralAtomRestraint(Potential):
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        return chiral_restraints.chiral_atom_restraint(conf, params, box, self.idxs)


@dataclass
class ChiralBondRestraint(Potential):
    idxs: NDArray[np.int32]
    signs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        return chiral_restraints.chiral_bond_restraint(conf, params, box, self.idxs, self.signs)


@dataclass
class FlatBottomBond(Potential):
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        return bonded.flat_bottom_bond(conf, params, box, self.idxs)


@dataclass
class PeriodicTorsion(Potential):
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        return bonded.periodic_torsion(conf, params, box, self.idxs)


@dataclass
class Nonbonded(Potential):
    num_atoms: int
    exclusion_idxs: NDArray[np.int32]
    scale_factors: NDArray[np.float64]
    beta: float
    cutoff: float

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        charge_rescale_mask, lj_rescale_mask = nonbonded.convert_exclusions_to_rescale_masks(
            self.exclusion_idxs, self.scale_factors, self.num_atoms
        )

        return nonbonded.nonbonded(
            conf,
            params,
            box,
            charge_rescale_mask,
            lj_rescale_mask,
            self.beta,
            self.cutoff,
            runtime_validate=False,  # needed for this to be JAX-transformable
        )

    def to_gpu(self, precision: Precision) -> GpuImplWrapper:
        all_pairs = NonbondedAllPairs(self.num_atoms, self.beta, self.cutoff)
        exclusions = NonbondedPairListNegated(self.exclusion_idxs, self.scale_factors, self.beta, self.cutoff)
        return FanoutSummedPotential([all_pairs, exclusions]).to_gpu(precision)


@dataclass
class NonbondedAllPairs(Potential):
    num_atoms: int
    beta: float
    cutoff: float
    atom_idxs: Optional[Array] = None

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        s = self.atom_idxs if self.atom_idxs is not None else slice(None)
        conf = jnp.array(conf)[s, :]
        num_atoms, _ = conf.shape
        no_rescale = jnp.ones((num_atoms, num_atoms))

        return nonbonded.nonbonded(
            conf,
            jnp.array(params)[s, :],
            box,
            no_rescale,
            no_rescale,
            self.beta,
            self.cutoff,
            runtime_validate=False,  # needed for this to be JAX-transformable
        )


@dataclass
class NonbondedInteractionGroup(Potential):
    num_atoms: int
    row_atom_idxs: NDArray[np.int32]
    beta: float
    cutoff: float

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        num_atoms, _ = jnp.array(conf).shape

        vdW, electrostatics = nonbonded.nonbonded_interaction_groups(
            conf,
            params,
            box,
            self.row_atom_idxs,
            np.setdiff1d(jnp.arange(num_atoms), self.row_atom_idxs),
            self.beta,
            self.cutoff,
        )
        return jnp.sum(vdW) + jnp.sum(electrostatics)


@dataclass
class NonbondedPairList(Potential):
    idxs: NDArray[np.int32]
    rescale_mask: NDArray[np.float64]
    beta: float
    cutoff: float

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        vdW, electrostatics = nonbonded.nonbonded_on_specific_pairs(
            conf, params, box, self.idxs, self.beta, self.cutoff, self.rescale_mask
        )
        return jnp.sum(vdW) + jnp.sum(electrostatics)


@dataclass
class NonbondedPairListNegated(Potential):
    idxs: NDArray[np.int32]
    rescale_mask: NDArray[np.float64]
    beta: float
    cutoff: float

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        vdW, electrostatics = nonbonded.nonbonded_on_specific_pairs(
            conf, params, box, self.idxs, self.beta, self.cutoff, self.rescale_mask
        )
        return -jnp.sum(vdW) - jnp.sum(electrostatics)

    @classmethod
    def _custom_ops_class_name(cls, precision):
        return f"NonbondedPairList_{get_custom_ops_class_name_suffix(precision)}_negated"


@dataclass
class NonbondedPairListPrecomputed(Potential):
    idxs: NDArray[np.int32]
    beta: float
    cutoff: float

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        vdW, electrostatics = nonbonded.nonbonded_on_precomputed_pairs(
            conf, params, box, self.idxs, self.beta, self.cutoff
        )
        return jnp.sum(vdW) + jnp.sum(electrostatics)


@dataclass
class SummedPotential(Potential):
    potentials: Sequence[Potential]
    params_init: Sequence[NDArray]

    def __post_init__(self):
        if len(self.potentials) != len(self.params_init):
            raise ValueError("number of potentials != number of parameter arrays")

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        shapes = [ps.shape for ps in self.params_init]
        return summed.summed_potential(conf, params, box, self.potentials, shapes)

    def to_gpu(self, precision: Precision) -> GpuImplWrapper:
        impls = [p.to_gpu(precision).unbound_impl for p in self.potentials]
        sizes = [ps.size for ps in self.params_init]
        return GpuImplWrapper(custom_ops.SummedPotential(impls, sizes))


@dataclass
class FanoutSummedPotential(Potential):
    potentials: Sequence[Potential]

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        return summed.fanout_summed_potential(conf, params, box, self.potentials)

    def to_gpu(self, precision: Precision) -> GpuImplWrapper:
        impls = [p.to_gpu(precision).unbound_impl for p in self.potentials]
        return GpuImplWrapper(custom_ops.FanoutSummedPotential(impls))
