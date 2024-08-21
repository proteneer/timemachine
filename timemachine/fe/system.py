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
from timemachine.ff.nonbonded.handlers import EnvironmentBCCHandler

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
    bond = angle = torsion = nonbonded = chiral_atom = chiral_bond = None

    for bp in bps:
        if isinstance(bp.potential, potentials.HarmonicBond):
            bond = bp
        elif isinstance(bp.potential, potentials.HarmonicAngle):
            angle = bp
        elif isinstance(bp.potential, potentials.PeriodicTorsion):
            torsion = bp
        elif isinstance(bp.potential, potentials.Nonbonded):
            nonbonded = bp
        elif isinstance(bp.potential, potentials.ChiralAtomRestraint):
            chiral_atom = bp
        # TODO: uncomment when re-enabling chiral_bond
        # elif isinstance(bp.potential, potentials.ChiralBondRestraint):
        #     chiral_bond = bp
        else:
            assert 0, "Unknown potential"

    assert bond
    assert angle
    assert nonbonded

    return VacuumSystem(bond, angle, torsion, nonbonded, chiral_atom, chiral_bond)


def convert_omm_system(omm_system, omm_topology, ff: Forcefield) -> Tuple["VacuumSystem", List[float]]:
    """Convert an openmm.System to a VacuumSystem object, also returning the masses"""
    from timemachine.ff.handlers import openmm_deserializer

    bps, masses = openmm_deserializer.deserialize_system(omm_system, cutoff=1.2)
    env_bcc_h = get_env_bcc_h(ff)
    if env_bcc_h is not None:
        env_bcc_h.
    system = convert_bps_into_system(bps)
    return system, masses


_Nonbonded = TypeVar("_Nonbonded", bound=Union[Nonbonded, NonbondedPairListPrecomputed, SummedPotential])
_HarmonicAngle = TypeVar("_HarmonicAngle", bound=Union[HarmonicAngle, HarmonicAngleStable])


@dataclass
class VacuumSystem(Generic[_Nonbonded, _HarmonicAngle]):
    # utility system container
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[_HarmonicAngle]
    torsion: Optional[BoundPotential[PeriodicTorsion]]
    nonbonded: BoundPotential[_Nonbonded]
    chiral_atom: Optional[BoundPotential[ChiralAtomRestraint]]
    chiral_bond: Optional[BoundPotential[ChiralBondRestraint]]

    def get_U_fn(self):
        """
        Return a jax function that evaluates the potential energy of a set of coordinates.
        """
        assert self.torsion
        U_fns = self.get_U_fns()

        def U_fn(x):
            return sum(U(x, box=None) for U in U_fns)

        return U_fn

    def get_U_fns(self) -> List[BoundPotential[Potential]]:
        # For molecules too small for to have certain terms,
        # skip when no params are present
        # Chiral bond restraints are disabled until checks are added (see GH #815)
        potentials = [self.bond, self.angle, self.torsion, self.chiral_atom, self.nonbonded]
        terms = cast(
            List[BoundPotential[Potential]],
            [p for p in potentials if p],
        )
        return [p for p in terms if p and len(p.params) > 0]


@dataclass
class HostGuestSystem:
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[HarmonicAngleStable]
    torsion: BoundPotential[PeriodicTorsion]
    chiral_atom: BoundPotential[ChiralAtomRestraint]
    chiral_bond: BoundPotential[ChiralBondRestraint]
    nonbonded_guest_pairs: BoundPotential[NonbondedPairListPrecomputed]
    nonbonded_host: BoundPotential[Nonbonded]
    nonbonded_host_guest_ixn: BoundPotential[NonbondedInteractionGroup]

    def get_U_fns(self):
        return [
            self.bond,
            self.angle,
            self.torsion,
            # Chiral bond restraints are disabled until checks are added
            # for consistency.
            self.chiral_atom,
            # self.chiral_bond,
            self.nonbonded_guest_pairs,
            self.nonbonded_host,
            self.nonbonded_host_guest_ixn,
        ]
