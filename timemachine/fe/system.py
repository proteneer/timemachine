import functools
import multiprocessing

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from timemachine.integrator import simulate
from timemachine.lib import potentials
from timemachine.potentials import bonded, chiral_restraints, nonbonded


def minimize_scipy(U_fn, x0, return_traj=False):
    shape = x0.shape

    def U_flat(x_flat):
        x_full = x_flat.reshape(*shape)
        return U_fn(x_full)

    grad_bfgs_fn = jax.jit(jax.grad(U_flat))

    traj = []

    def callback_fn(x):
        traj.append(x.reshape(*shape))

    res = scipy.optimize.minimize(U_flat, x0.reshape(-1), jac=grad_bfgs_fn, callback=callback_fn)
    xi = res.x.reshape(*shape)

    if return_traj:
        return traj
    else:
        return xi


def simulate_system(U_fn, x0, num_samples=20000, steps_per_batch=500, num_workers=None, minimize=True):
    num_atoms = x0.shape[0]
    if minimize:
        x_min = minimize_scipy(U_fn, x0)
    else:
        x_min = x0

    seed = 2023

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


def convert_bps_into_system(bps):

    system = VacuumSystem(None, None, None, None, None, None)

    for pot in bps:
        if isinstance(pot, potentials.HarmonicBond):
            system.bond = pot
        elif isinstance(pot, potentials.HarmonicAngle):
            system.angle = pot
        elif isinstance(pot, potentials.PeriodicTorsion):
            system.torsion = pot
        elif isinstance(pot, potentials.Nonbonded):
            system.nonbonded = pot
        else:
            assert 0, "Unknown potential"

    return system


class VacuumSystem:

    # utility system container

    def __init__(self, bond, angle, torsion, nonbonded, chiral_atom, chiral_bond):
        self.bond = bond
        self.angle = angle
        self.torsion = torsion
        self.nonbonded = nonbonded
        self.chiral_atom = chiral_atom
        self.chiral_bond = chiral_bond

    def get_U_fn(self):
        """
        Return a jax function that evaluates the potential energy of a set of coordinates.
        """
        bond_U = functools.partial(
            bonded.harmonic_bond,
            params=np.array(self.bond.params),
            box=None,
            lamb=0.0,
            bond_idxs=np.array(self.bond.get_idxs()),
        )
        angle_U = functools.partial(
            bonded.harmonic_angle,
            params=np.array(self.angle.params),
            box=None,
            lamb=0.0,
            angle_idxs=np.array(self.angle.get_idxs()),
        )
        torsion_U = functools.partial(
            bonded.periodic_torsion,
            params=np.array(self.torsion.params),
            box=None,
            lamb=0.0,
            torsion_idxs=np.array(self.torsion.get_idxs()),
        )
        nbpl_U = functools.partial(
            nonbonded.nonbonded_v3_on_precomputed_pairs,
            pairs=np.array(self.nonbonded.get_idxs()),
            offsets=np.array(self.nonbonded.get_offsets()),
            params=np.array(self.nonbonded.params),
            box=None,
            beta=self.nonbonded.get_beta(),
            cutoff=self.nonbonded.get_cutoff(),
        )

        if self.chiral_atom:
            chiral_atom_U = functools.partial(
                chiral_restraints.chiral_atom_restraint,
                params=np.array(self.chiral_atom.params),
                box=None,
                idxs=np.array(self.chiral_atom.get_idxs()),
                lamb=0.0,
            )
        else:
            chiral_atom_U = lambda _: 0

        if self.chiral_bond:
            chiral_bond_U = functools.partial(
                chiral_restraints.chiral_bond_restraint,
                params=np.array(self.chiral_bond.params),
                box=None,
                idxs=np.array(self.chiral_bond.get_idxs()),
                signs=np.array(self.chiral_bond.get_signs()),
                lamb=0.0,
            )
        else:
            chiral_bond_U = lambda _: 0

        def U_fn(x):
            Us_vdw, Us_coulomb = nbpl_U(x)
            chiral_U = chiral_atom_U(x) + chiral_bond_U(x)

            return bond_U(x) + angle_U(x) + torsion_U(x) + jnp.sum(Us_vdw) + jnp.sum(Us_coulomb) + chiral_U

        return U_fn


class HostGuestSystem:

    # utility system container

    def __init__(self, bond, angle, torsion, chiral_atom, chiral_bond, nonbonded_guest_pairs, nonbonded_host_guest):
        self.bond = bond
        self.angle = angle
        self.torsion = torsion
        self.chiral_atom = chiral_atom
        self.chiral_bond = chiral_bond
        self.nonbonded_guest_pairs = nonbonded_guest_pairs
        self.nonbonded_host_guest = nonbonded_host_guest

    def get_U_fns(self):

        return [
            self.bond,
            self.angle,
            self.torsion,
            self.chiral_atom,
            self.chiral_bond,
            self.nonbonded_guest_pairs,
            self.nonbonded_host_guest,
        ]
