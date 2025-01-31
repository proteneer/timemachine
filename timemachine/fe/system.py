import multiprocessing
from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, Tuple, TypeVar, Union, cast

import jax
import numpy as np
import scipy

from timemachine import potentials
from timemachine.integrator import simulate
from timemachine.potentials import (
    BoundPotential,
    ChiralAtomRestraint,
    ChiralBondRestraint,
    HarmonicAngle,
    HarmonicAngleStable,
    HarmonicBond,
    Nonbonded,
    NonbondedInteractionGroup,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
    Potential,
    SummedPotential,
)

# Chiral bond restraints are disabled until checks are added (see GH #815)
# from timemachine.potentials import bonded, chiral_restraints, nonbonded


def minimize_scipy(U_fn, x0, return_traj=False, seed=2024):
    shape = x0.shape

    @jax.jit
    def U_flat(x_flat):
        x_full = x_flat.reshape(*shape)
        return U_fn(x_full)

    grad_bfgs_fn = jax.jit(jax.grad(U_flat))

    traj = []

    def callback_fn(x):
        traj.append(x.reshape(*shape))

    minimizer_kwargs = {"jac": grad_bfgs_fn, "callback": callback_fn}
    res = scipy.optimize.basinhopping(U_flat, x0.reshape(-1), minimizer_kwargs=minimizer_kwargs, seed=seed)
    xi = res.x.reshape(*shape)

    if return_traj:
        return traj
    else:
        return xi


def simulate_system(U_fn, x0, num_samples=20000, steps_per_batch=500, num_workers=None, minimize=True):
    num_atoms = x0.shape[0]

    seed = 2023

    if minimize:
        x_min = minimize_scipy(U_fn, x0, seed=seed)
    else:
        x_min = x0

    num_workers = num_workers or multiprocessing.cpu_count()
    samples_per_worker = int(np.ceil(num_samples / num_workers))

    burn_in_batches = num_samples // 10
    frames, _ = simulate(
        x_min,
        U_fn,
        300.0,
        np.ones(num_atoms) * 4.0,
        steps_per_batch,
        samples_per_worker + burn_in_batches,
        num_workers,
        seed=seed,
    )
    # (ytz): discard burn in batches
    frames = frames[:, burn_in_batches:, :, :]
    # collect over all workers
    frames = frames.reshape(-1, num_atoms, 3)[:num_samples]
    # sanity check that we didn't undersample
    assert len(frames) == num_samples
    return frames


def convert_bps_into_system(bps: Sequence[potentials.BoundPotential]):
    assert isinstance(bps[0].potential, potentials.HarmonicBond)
    assert isinstance(bps[1].potential, potentials.HarmonicAngle)
    assert isinstance(bps[2].potential, potentials.PeriodicTorsion)  # proper
    assert isinstance(bps[3].potential, potentials.PeriodicTorsion)  # improper
    assert isinstance(bps[4].potential, potentials.Nonbonded)  # NB all pairs
    return HostSystem(
        bond=bps[0],
        angle=bps[1],
        proper=bps[2],
        improper=bps[3],
        nonbonded_all_pairs=bps[4],
    )


def convert_omm_system(omm_system) -> Tuple["HostSystem", List[float]]:
    """Convert an openmm.System to a HostSystem object, also returning the masses"""
    from timemachine.ff.handlers import openmm_deserializer

    bps, masses = openmm_deserializer.deserialize_system(omm_system, cutoff=1.2)
    system = convert_bps_into_system(bps)
    return system, masses


_Nonbonded = TypeVar("_Nonbonded", bound=Union[Nonbonded, NonbondedPairListPrecomputed, SummedPotential])
_HarmonicAngle = TypeVar("_HarmonicAngle", bound=Union[HarmonicAngle, HarmonicAngleStable])

from abc import ABC, abstractmethod


class AbstractSystem(ABC):
    def get_U_fn(self):
        """
        Return a jax function that evaluates the potential energy of a set of coordinates.
        """
        U_fns = self.get_U_fns()

        def U_fn(x):
            return sum(U(x, box=None) for U in U_fns)

        return U_fn

    @abstractmethod
    def get_U_fns(self):
        raise NotImplementedError()


@dataclass  # mcwitt: Generic can be removed in python 3.12
class HostSystem(Generic[_HarmonicAngle], AbstractSystem):
    # utility system container
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[_HarmonicAngle]
    proper: BoundPotential[PeriodicTorsion]
    improper: BoundPotential[PeriodicTorsion]
    nonbonded_all_pairs: BoundPotential[Nonbonded]

    def get_U_fns(self) -> List[BoundPotential[Potential]]:
        potentials: List[BoundPotential] = [
            self.bond,
            self.angle,
            self.proper,
            self.improper,
            self.nonbonded_all_pairs,
        ]
        # (TODO): len(p.params) > 0 is dangerous if we later on have potentials
        # that do *not* have "free" parameters defined but should still be left on.
        # (eg. if the force constants for something like virtual sites are defined by
        # properties/attributes on the class, rather than attributes)
        return [p for p in potentials if len(p.params) > 0]


@dataclass  # mcwitt: Generic can be removed in python 3.12
class GuestSystem(Generic[_HarmonicAngle], AbstractSystem):
    # utility system container
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[_HarmonicAngle]
    proper: BoundPotential[PeriodicTorsion]
    improper: BoundPotential[PeriodicTorsion]
    chiral_atom: BoundPotential[ChiralAtomRestraint]
    chiral_bond: BoundPotential[ChiralBondRestraint]
    nonbonded_pair_list: BoundPotential[NonbondedPairListPrecomputed]

    def get_U_fns(self) -> List[BoundPotential[Potential]]:
        # For molecules too small for to have certain terms,
        # skip when no params are present
        # Chiral bond restraints are disabled until checks are added (see GH #815)
        potentials: List[BoundPotential] = [
            self.bond,
            self.angle,
            self.proper,
            self.improper,
            self.chiral_atom,
            self.nonbonded_pair_list,
        ]
        # (TODO): len(p.params) > 0 is dangerous if we later on have potentials
        # that do *not* have "free" parameters defined but should still be left on.
        # (eg. if the force constants for something like virtual sites are defined by
        # properties/attributes on the class, rather than attributes)
        return [p for p in potentials if len(p.params) > 0]


@dataclass  # mcwitt: Generic can be removed in python 3.12
class HostGuestSystem(Generic[_HarmonicAngle], AbstractSystem):
    # utility system container
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[_HarmonicAngle]
    proper: BoundPotential[PeriodicTorsion]
    improper: BoundPotential[PeriodicTorsion]
    chiral_atom: BoundPotential[ChiralAtomRestraint]
    chiral_bond: BoundPotential[ChiralBondRestraint]
    nonbonded_pair_list: BoundPotential[NonbondedPairListPrecomputed]
    nonbonded_all_pairs: BoundPotential[Nonbonded]
    nonbonded_ixn_group: BoundPotential[NonbondedInteractionGroup]

    def get_U_fns(self) -> List[BoundPotential[Potential]]:
        # For molecules too small for to have certain terms,
        # skip when no params are present
        # Chiral bond restraints are disabled until checks are added (see GH #815)
        potentials: List[BoundPotential] = [
            self.bond,
            self.angle,
            self.proper,
            self.improper,
            self.chiral_atom,
            self.nonbonded_pair_list,
            self.nonbonded_all_pairs,
            self.nonbonded_ixn_group,
        ]
        # (TODO): len(p.params) > 0 is dangerous if we later on have potentials
        # that do *not* have "free" parameters defined but should still be left on.
        # (eg. if the force constants for something like virtual sites are defined by
        # properties/attributes on the class, rather than attributes)
        return [p for p in potentials if len(p.params) > 0]
