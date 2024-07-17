# Enhanced sampling protocols

# This file contains utility functions to generate samples in the gas-phase.

import logging
import multiprocessing
import os

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax.scipy.special import logsumexp as jlogsumexp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.special import logsumexp

from timemachine import lib
from timemachine.constants import BOLTZ
from timemachine.fe import free_energy, topology
from timemachine.fe.free_energy import HostConfig
from timemachine.fe.utils import get_mol_masses, get_romol_conf
from timemachine.integrator import simulate
from timemachine.lib import custom_ops
from timemachine.md import builders, minimizer
from timemachine.md.barostat.moves import NPTMove
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import bonded, rmsd

logger = logging.getLogger(__name__)


def identify_rotatable_bonds(mol):
    """
    Identify rotatable bonds in a molecule.

    Right now this is an extremely crude and inaccurate method that should *not* be used for production.
    This misses simple cases like benzoic acids, amides, etc.

    Parameters
    ----------
    mol: ROMol
        Input molecule

    Returns
    -------
    set of 2-tuples
        Set of bonds identified as rotatable.

    """
    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(pattern, uniquify=1)

    # sanity check
    assert len(matches) >= rdMolDescriptors.CalcNumRotatableBonds(mol)

    sorted_matches = set()

    for i, j in matches:
        if j < i:
            i, j = j, i
        sorted_matches.add((i, j))

    return sorted_matches


class VacuumState:
    def __init__(self, mol, ff):
        """
        VacuumState allows us to enable/disable various parts of a forcefield so that
        we can more easily sample across rotational barriers in the vacuum.

        Parameters
        ----------
        mol: Chem.ROMol
            rdkit molecule

        ff: Forcefield
            forcefield
        """

        self.mol = mol
        bt = topology.BaseTopology(mol, ff)
        self.bond_params, self.hb_potential = bt.parameterize_harmonic_bond(ff.hb_handle.params)
        self.angle_params, self.ha_potential = bt.parameterize_harmonic_angle(ff.ha_handle.params)
        self.proper_torsion_params, self.pt_potential = bt.parameterize_proper_torsion(ff.pt_handle.params)
        (
            self.improper_torsion_params,
            self.it_potential,
        ) = bt.parameterize_improper_torsion(ff.it_handle.params)

        self.lamb = 0.0
        self.nb_params, self.nb_potential = bt.parameterize_nonbonded(
            ff.q_handle.params,
            ff.q_handle_intra.params,
            ff.q_handle_solv.params,
            ff.lj_handle.params,
            ff.lj_handle_intra.params,
            ff.lj_handle_solv.params,
            self.lamb,
        )

        self.box = None

    def _harmonic_bond_nrg(self, x):
        return self.hb_potential(x, self.bond_params, self.box)

    def _harmonic_angle_nrg(self, x):
        return self.ha_potential(x, self.angle_params, self.box)

    def _proper_torsion_nrg(self, x):
        return self.pt_potential(x, self.proper_torsion_params, self.box)

    def _improper_torsion_nrg(self, x):
        return self.it_potential(x, self.improper_torsion_params, self.box)

    def _nonbonded_nrg(self, x, decharge):
        if decharge:
            charge_indices = jnp.index_exp[:, 0]
            nb_params = jnp.asarray(self.nb_params).at[charge_indices].set(0)
        else:
            nb_params = self.nb_params

        # tbd: set to None
        box = np.eye(3) * 1000

        return self.nb_potential(x, nb_params, box)

    def U_easy(self, x):
        """
        Vacuum potential energy function typically used for the proposal distribution.
        This state has rotatable torsions fully turned off, and nonbonded terms completely
        disabled. Note that this may at times cross atropisomerism barriers in unphysical ways.

        Parameters
        ----------
        x: np.ndarray(N,3)
            Conformation of the input ligand

        Returns
        -------
        float
            Potential energy

        """
        easy_proper_torsion_idxs = []
        easy_proper_torsion_params = []

        rotatable_bonds = identify_rotatable_bonds(self.mol)

        for idxs, params in zip(self.pt_potential.idxs, self.proper_torsion_params):
            _, j, k, _ = idxs
            if (j, k) in rotatable_bonds:
                logger.debug("turning off torsion %s", idxs)
                continue
            else:
                easy_proper_torsion_idxs.append(idxs)
                easy_proper_torsion_params.append(params)

        easy_proper_torsion_idxs = np.array(easy_proper_torsion_idxs, dtype=np.int32)
        easy_proper_torsion_params = np.array(easy_proper_torsion_params, dtype=np.float64)

        proper_torsion_nrg = bonded.periodic_torsion(x, easy_proper_torsion_params, self.box, easy_proper_torsion_idxs)

        return (
            self._harmonic_bond_nrg(x)
            + self._harmonic_angle_nrg(x)
            + proper_torsion_nrg
            + self._improper_torsion_nrg(x)
        )

    def U_full(self, x):
        """
        Fully interacting vacuum potential energy.

        Parameters
        ----------
        x: np.ndarray(N,3)
            Conformation of the input ligand

        Returns
        -------
        float
            Potential energy

        """
        return (
            self._harmonic_bond_nrg(x)
            + self._harmonic_angle_nrg(x)
            + self._proper_torsion_nrg(x)
            + self._improper_torsion_nrg(x)
            + self._nonbonded_nrg(x, decharge=False)
        )

    def U_decharged(self, x):
        """
        Fully interacting, but decharged, vacuum potential energy. Samples from distributions
        under this potential energy tend to have better phase space overlap with condensed
        states.

        Parameters
        ----------
        x: np.ndarray(N,3)
            Conformation

        Returns
        -------
        float
            Potential energy

        """
        return (
            self._harmonic_bond_nrg(x)
            + self._harmonic_angle_nrg(x)
            + self._proper_torsion_nrg(x)
            + self._improper_torsion_nrg(x)
            + self._nonbonded_nrg(x, decharge=True)
        )


# (ytz): in order for XLA's pmap to properly parallelize over multiple CPU devices, we need to
# set this explicitly via a magical command line arg. This should be ran in a subprocess
def _wrap_simulate(args):
    (
        mol,
        U_proposal,
        U_target,
        temperature,
        masses,
        steps_per_batch,
        num_batches,
        burn_in_batches,
        num_workers,
        seed,
    ) = args

    assert multiprocessing.get_start_method() == "spawn"
    kT = temperature * BOLTZ
    x0 = get_romol_conf(mol)
    batches_per_worker = int(np.ceil(num_batches / num_workers))

    xs_proposal, vs_proposal = simulate(
        x0,
        U_proposal,
        temperature,
        masses,
        steps_per_batch,
        batches_per_worker + burn_in_batches,
        num_workers,
        seed,
    )

    # discard burn-in batches and reshape into a single flat array
    xs_proposal = xs_proposal[:, burn_in_batches:, :, :]
    vs_proposal = vs_proposal[:, burn_in_batches:, :, :]

    batch_U_proposal_fn = jax.pmap(jax.vmap(U_proposal))
    batch_U_target_fn = jax.pmap(jax.vmap(U_target))

    Us_target = batch_U_target_fn(xs_proposal)
    Us_proposal = batch_U_proposal_fn(xs_proposal)

    log_numerator = -Us_target.reshape(-1) / kT
    log_denominator = -Us_proposal.reshape(-1) / kT

    log_weights = log_numerator - log_denominator

    # reshape into flat array by removing num_workers dimension
    num_atoms = mol.GetNumAtoms()
    xs_proposal = xs_proposal.reshape(-1, num_atoms, 3)
    vs_proposal = vs_proposal.reshape(-1, num_atoms, 3)

    xvs_proposal = np.stack([xs_proposal, vs_proposal], axis=1)

    # truncate to user requested num_batches
    xvs_proposal = xvs_proposal[:num_batches, ...]
    log_weights = log_weights[:num_batches]

    return xvs_proposal, log_weights


def init_env(num_workers):
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(num_workers)


def generate_log_weighted_samples(
    mol, temperature, U_proposal, U_target, seed, steps_per_batch=250, num_batches=24000, num_workers=None
):
    """
    Generate log_weighted samples from a proposal distribution using U_proposal,
    with log_weights defined by the difference relative to U_target

    Parameters
    ----------
    mol: Chem.Mol

    temperature: float
        Temperature in Kelvin

    U_proposal: fn
        Potential energy function for the proposal distribution

    U_target: fn
        Potential energy function for the target distribution

    seed: int
        Random number seed

    steps_per_batch: int
        Number of steps per batch.

    num_batches: int
        Number of batches

    num_workers: Optional, int
        Number of parallel computations

    Returns
    -------
    (num_batches*num_workers, num_atoms, 3) np.ndarray
        Samples generated from p_target

    """
    masses = get_mol_masses(mol)

    if num_workers is None:
        num_workers = os.cpu_count()

    burn_in_batches = 2000

    # wraps a callable fn so it runs in a subprocess with the device_count set explicitly
    with multiprocessing.get_context("spawn").Pool(1, init_env, [num_workers]) as pool:
        args = (
            mol,
            U_proposal,
            U_target,
            temperature,
            masses,
            steps_per_batch,
            num_batches,
            burn_in_batches,
            num_workers,
            seed,
        )
        xvs_proposal, log_weights = pool.map(_wrap_simulate, (args,))[0]

        assert xvs_proposal.shape[1] == 2
        assert xvs_proposal.shape[0] == num_batches
        assert log_weights.shape[0] == num_batches

        return xvs_proposal, log_weights


def sample_from_log_weights(weighted_samples, log_weights, size):
    """
    Given a collection of weighted samples with log_weights, resample them
    into an unweighted collection of size samples.

    Parameters
    ----------
    weighted_samples: list
        List of arbitrary objects

    log_weights: np.array
        Log weights

    size: int
        number of samples we'd like to return

    Returns
    -------
    array with size elements

    """
    weights = np.exp(log_weights - logsumexp(log_weights))
    assert len(weights) == len(weighted_samples)
    assert np.abs(np.sum(weights) - 1) < 1e-5
    idxs = np.random.choice(np.arange(len(weights)), size=size, p=weights)
    return [weighted_samples[i] for i in idxs]


def jax_sample_from_log_weights(weighted_samples, log_weights, size, key):
    """
    Given a collection of weighted samples with log_weights, resample them
    into an unweighted collection of size samples.

    Parameters
    ----------
    weighted_samples: list
        List of arbitrary objects

    log_weights: np.array
        Log weights

    size: int
        number of samples we'd like to return

    Returns
    -------
    array with size elements

    """
    weights = jnp.exp(log_weights - jlogsumexp(log_weights))
    idxs = jrandom.choice(key, jnp.arange(len(weights)), shape=(size,), p=weights)
    return weighted_samples[idxs]


def get_solvent_phase_system(mol, ff, lamb: float, box_width=3.0, margin=0.5, minimize_energy=True):
    """
    Given a mol and forcefield return a solvated system where the
    solvent has (optionally) been minimized.

    Parameters
    ----------
    mol: Chem.Mol

    ff: Forcefield

    lamb: float

    box_width: float
        water box initial width in nm

    margin: Optional, float
        Box margin in nm, default is 0.5 nm.

    minimize_energy: bool
        whether to apply minimize_host_4d
    """

    # construct water box
    water_system, water_coords, water_box, water_topology = builders.build_water_system(box_width, ff.water_ff)
    water_box = water_box + np.eye(3) * margin  # add a small margin around the box for stability
    host_config = HostConfig(water_system, water_coords, water_box, water_coords.shape[0])

    # construct alchemical system
    bt = topology.BaseTopology(mol, ff)
    afe = free_energy.AbsoluteFreeEnergy(mol, bt)
    ff_params = ff.get_params()
    potentials, params, masses = afe.prepare_host_edge(ff_params, host_config, lamb)

    # concatenate (optionally minimized) water_coords and ligand_coords
    ligand_coords = get_romol_conf(mol)
    if minimize_energy:
        new_water_coords = minimizer.minimize_host_4d([mol], host_config, ff)
        coords = np.concatenate([new_water_coords, ligand_coords])
    else:
        coords = np.concatenate([water_coords, ligand_coords])

    return potentials, params, masses, coords, water_box


def equilibrate_solvent_phase(
    potentials,
    params,
    masses,
    coords,  # minimized_coords
    box,
    temperature,
    pressure,
    num_steps,
    seed=None,
):
    """
    Generate samples in the solvent phase.
    """

    dt = 1e-4
    friction = 1.0

    bps = []
    for p, pot in zip(params, potentials):
        bps.append(pot.bind(p))

    all_impls = [bp.to_gpu(np.float32).bound_impl for bp in bps]

    intg_equil = lib.LangevinIntegrator(temperature, dt, friction, masses, seed)
    intg_equil_impl = intg_equil.impl()

    bond_list = get_bond_list(potentials[0])
    group_idxs = get_group_indices(bond_list, len(masses))
    barostat_interval = 5

    barostat = lib.MonteCarloBarostat(len(masses), pressure, temperature, group_idxs, barostat_interval, seed + 1)
    barostat_impl = barostat.impl(all_impls)

    # equilibration/minimization doesn't need a barostat
    equil_ctxt = custom_ops.Context(
        coords, np.zeros_like(coords), box, intg_equil_impl, all_impls, movers=[barostat_impl]
    )

    # TODO: revert to 50k
    equil_ctxt.multiple_steps(num_steps)

    x0 = equil_ctxt.get_x_t()

    # (ytz): This has to be zeros_like for now since if we freeze ligand
    # coordinates it would start to move during rejected moves.
    v0 = np.zeros_like(x0)

    return CoordsVelBox(x0, v0, equil_ctxt.get_box())


def align_sample(x_vacuum, x_solvent):
    """
    Return a rigidly transformed x_vacuum that is maximally aligned to x_solvent.
    """
    num_atoms = len(x_vacuum)

    xa = x_solvent[-num_atoms:]
    xb = x_vacuum

    assert xa.shape == xb.shape

    xb_new = rmsd.align_x2_unto_x1(xa, xb)
    return xb_new


def align_and_replace(x_vacuum, x_solvent):
    num_ligand_atoms = len(x_vacuum)
    aligned_x_vacuum = align_sample(x_vacuum, x_solvent)
    ligand_idxs = jnp.index_exp[-num_ligand_atoms:]
    return jnp.asarray(x_solvent).at[ligand_idxs].set(aligned_x_vacuum)


batch_align_and_replace = jax.jit(jax.vmap(align_and_replace, in_axes=(0, None)))


def aligned_batch_propose(xvb, K, key, vacuum_samples, vacuum_log_weights):
    vacuum_samples = jax_sample_from_log_weights(vacuum_samples, vacuum_log_weights, K, key)

    x_solvent = xvb.coords
    v_solvent = xvb.velocities
    b_solvent = xvb.box

    replaced_samples = batch_align_and_replace(vacuum_samples, x_solvent)

    new_xvbs = []

    # modify only ligand coordinates in the proposal
    for x_r in replaced_samples:
        new_xvbs.append(CoordsVelBox(x_r, v_solvent, b_solvent))

    return new_xvbs


def jax_aligned_batch_propose_coords(x, K, key, vacuum_samples, vacuum_log_weights):
    vacuum_samples = jax_sample_from_log_weights(vacuum_samples, vacuum_log_weights, K, key)
    return batch_align_and_replace(vacuum_samples, x)


def pregenerate_samples(
    mol,
    ff,
    lamb,
    seed,
    n_solvent_samples=1000,
    n_ligand_batches=30000,
    temperature=300.0,
    pressure=1.0,
    num_workers=None,
):
    potentials, params, masses, coords, box = get_solvent_phase_system(mol, ff, lamb)
    print(f"Generating {n_solvent_samples} solvent samples")
    solvent_xvbs = generate_solvent_samples(
        coords, box, masses, potentials, params, temperature, pressure, seed, n_solvent_samples, num_workers
    )

    print("Generating ligand samples")
    ligand_samples, ligand_log_weights = generate_ligand_samples(
        n_ligand_batches, mol, ff, temperature, seed, num_workers=num_workers
    )

    return solvent_xvbs, ligand_samples, ligand_log_weights


def generate_solvent_samples(
    coords,
    box,
    masses,
    potentials,
    params,
    temperature,
    pressure,
    seed,
    n_samples,
    num_equil_steps=50000,
    md_steps_per_move=1000,
):
    """Discard num_equil_steps of MD, then return n_samples each separated by md_steps_per_move"""
    xvb0 = equilibrate_solvent_phase(
        potentials, params, masses, coords, box, temperature, pressure, num_equil_steps, seed
    )

    npt_mover = NPTMove(potentials, masses, temperature, pressure, n_steps=md_steps_per_move, seed=seed)

    xvbs = [xvb0]
    for _ in range(n_samples):
        xvbs.append(npt_mover.move(xvbs[-1]))
    return xvbs


def generate_ligand_samples(num_batches, mol, ff, temperature, seed, num_workers=None):
    """Generate (weighted) samples of the ligand in vacuum, by importance sampling from a less-hindered state where
    torsions and intramolecular nonbonded terms are disabled"""
    state = VacuumState(mol, ff)
    vacuum_samples, vacuum_log_weights = generate_log_weighted_samples(
        mol, temperature, state.U_easy, state.U_full, num_batches=num_batches, seed=seed, num_workers=num_workers
    )

    return vacuum_samples, vacuum_log_weights
