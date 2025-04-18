import multiprocessing
from abc import ABC
from dataclasses import dataclass, fields

import jax
import numpy as np
import scipy

from timemachine.integrator import simulate
from timemachine.potentials import (
    BoundPotential,
    ChiralAtomRestraint,
    ChiralBondRestraint,
    HarmonicAngle,
    HarmonicBond,
    Nonbonded,
    NonbondedInteractionGroup,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
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


@dataclass
class AbstractSystem(ABC):
    def get_U_fn(self):
        """
        Return a jax function that evaluates the potential energy of a set of coordinates.
        """
        U_fns = self.get_U_fns()

        def U_fn(x):
            return sum(U(x, box=None) for U in U_fns)

        return U_fn

    def get_U_fns(self) -> list[BoundPotential]:
        """
        Return a list of bound potential"""
        potentials: list[BoundPotential] = []
        for f in fields(self):
            bp = getattr(self, f.name)
            # (TODO): chiral_bonds currently disabled
            if f.name != "chiral_bond":
                potentials.append(bp)

        return potentials


@dataclass
class HostSystem(AbstractSystem):
    # utility system container
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[HarmonicAngle]
    proper: BoundPotential[PeriodicTorsion]
    improper: BoundPotential[PeriodicTorsion]
    nonbonded_all_pairs: BoundPotential[Nonbonded]


@dataclass
class GuestSystem(AbstractSystem):
    # utility system container
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[HarmonicAngle]
    proper: BoundPotential[PeriodicTorsion]
    improper: BoundPotential[PeriodicTorsion]
    chiral_atom: BoundPotential[ChiralAtomRestraint]
    chiral_bond: BoundPotential[ChiralBondRestraint]
    nonbonded_pair_list: BoundPotential[NonbondedPairListPrecomputed]


@dataclass
class HostGuestSystem(AbstractSystem):
    # utility system container
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[HarmonicAngle]
    proper: BoundPotential[PeriodicTorsion]
    improper: BoundPotential[PeriodicTorsion]
    chiral_atom: BoundPotential[ChiralAtomRestraint]
    chiral_bond: BoundPotential[ChiralBondRestraint]
    nonbonded_pair_list: BoundPotential[NonbondedPairListPrecomputed]
    nonbonded_all_pairs: BoundPotential[Nonbonded]
    nonbonded_ixn_group: BoundPotential[NonbondedInteractionGroup]
