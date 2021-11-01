# Enhanced sampling protocols

# This file contains utility functions to generate samples in the gas-phase.

import jax
import numpy as np
from scipy.special import logsumexp

from fe import topology
from fe.utils import get_romol_conf
from fe import free_energy

from timemachine.integrator import simulate
from timemachine.potentials import bonded, nonbonded, rmsd
from timemachine.constants import BOLTZ
from timemachine import lib
from timemachine.lib import custom_ops

from md.states import CoordsVelBox
from md import minimizer
from md import builders
from md.barostat.utils import get_group_indices, get_bond_list

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


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
    matches = mol.GetSubstructMatches(pattern)

    # sanity check
    assert len(matches) == rdMolDescriptors.CalcNumRotatableBonds(mol)

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
        self.nb_params, self.nb_potential = bt.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

        self.box = None
        self.lamb = 0.0

    def _harmonic_bond_nrg(self, x):
        return bonded.harmonic_bond(x, self.bond_params, self.box, self.lamb, self.hb_potential.get_idxs())

    def _harmonic_angle_nrg(self, x):
        return bonded.harmonic_angle(x, self.angle_params, self.box, self.lamb, self.ha_potential.get_idxs())

    def _proper_torsion_nrg(self, x):
        return bonded.periodic_torsion(
            x,
            self.proper_torsion_params,
            self.box,
            self.lamb,
            self.pt_potential.get_idxs(),
        )

    def _improper_torsion_nrg(self, x):
        return bonded.periodic_torsion(
            x,
            self.improper_torsion_params,
            self.box,
            self.lamb,
            self.it_potential.get_idxs(),
        )

    def _nonbonded_nrg(self, x, decharge):
        exclusions = self.nb_potential.get_exclusion_idxs()
        scales = self.nb_potential.get_scale_factors()

        N = x.shape[0]
        charge_rescale_mask = np.ones((N, N))
        lj_rescale_mask = np.ones((N, N))

        if decharge:
            nb_params = jax.ops.index_update(self.nb_params, jax.ops.index[:, 0], 0)
        else:
            nb_params = self.nb_params

        for (i, j), (lj_scale, q_scale) in zip(exclusions, scales):
            charge_rescale_mask[i][j] = 1 - q_scale
            charge_rescale_mask[j][i] = 1 - q_scale
            lj_rescale_mask[i][j] = 1 - lj_scale
            lj_rescale_mask[j][i] = 1 - lj_scale

        beta = self.nb_potential.get_beta()
        cutoff = self.nb_potential.get_cutoff()
        lambda_plane_idxs = np.zeros(N)
        lambda_offset_idxs = np.zeros(N)

        # tbd: set to None
        box = np.eye(3) * 1000

        return nonbonded.nonbonded_v3(
            x,
            nb_params,
            box,
            self.lamb,
            charge_rescale_mask,
            lj_rescale_mask,
            beta,
            cutoff,
            lambda_plane_idxs,
            lambda_offset_idxs,
            runtime_validate=False,
        )

    def U_easy(self, x):
        """
        Gas-phase potential energy function typically used for the proposal distribution.
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

        for idxs, params in zip(self.pt_potential.get_idxs(), self.proper_torsion_params):
            _, j, k, _ = idxs
            if (j, k) in rotatable_bonds:
                print("turning off torsion", idxs)
                continue
            else:
                easy_proper_torsion_idxs.append(idxs)
                easy_proper_torsion_params.append(params)

        easy_proper_torsion_idxs = np.array(easy_proper_torsion_idxs, dtype=np.int32)
        easy_proper_torsion_params = np.array(easy_proper_torsion_params, dtype=np.float64)

        proper_torsion_nrg = bonded.periodic_torsion(
            x, easy_proper_torsion_params, self.box, self.lamb, easy_proper_torsion_idxs
        )

        return (
            self._harmonic_bond_nrg(x)
            + self._harmonic_angle_nrg(x)
            + proper_torsion_nrg
            + self._improper_torsion_nrg(x)
        )

    def U_full(self, x):
        """
        Fully interacting gas-phase potential energy.

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
        Fully interacting, but decharged, gas-phase potential energy. Samples from distributions
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


def generate_log_weighted_samples(
    mol,
    temperature,
    U_proposal,
    U_target,
    steps_per_batch=250,
    num_batches=20000,
    num_workers=None,
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

    size: int
        Number of samples return

    steps_per_batch: int
        Number of steps per batch

    num_batches: int
        Number of batches

    num_workers: int
        Number of parallel computations

    Returns
    -------
    (num_batches*num_workers, num_atoms, 3) np.ndarray
        Samples generated from p_target

    """
    x0 = get_romol_conf(mol)

    kT = temperature * BOLTZ
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])

    if num_workers is None:
        num_workers = jax.device_count()

    burn_in_batches = 1000

    # xs_proposal has shape [num_workers, num_batches+burn_in_batches, num_atoms, 3]
    xs_proposal = simulate(
        x0,
        U_proposal,
        temperature,
        masses,
        steps_per_batch,
        num_batches + burn_in_batches,
        num_workers,
    )

    num_atoms = mol.GetNumAtoms()

    # discard burn-in batches and reshape into a single flat array
    xs_proposal = xs_proposal[:, burn_in_batches:, :, :]

    batch_U_proposal_fn = jax.pmap(jax.vmap(U_proposal))
    batch_U_target_fn = jax.pmap(jax.vmap(U_target))

    Us_target = batch_U_target_fn(xs_proposal)
    Us_proposal = batch_U_proposal_fn(xs_proposal)

    log_numerator = -Us_target.reshape(-1) / kT
    log_denominator = -Us_proposal.reshape(-1) / kT

    log_weights = log_numerator - log_denominator

    # reshape into flat array by removing num_workers dimension
    xs_proposal = xs_proposal.reshape(-1, num_atoms, 3)

    return xs_proposal, log_weights


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
    return weighted_samples[idxs]


def get_solvent_phase_system(mol, ff):
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    water_system, water_coords, water_box, water_topology = builders.build_water_system(3.0)
    water_box = water_box + np.eye(3) * 0.5  # add a small margin around the box for stability
    num_water_atoms = len(water_coords)
    afe = free_energy.AbsoluteFreeEnergy(mol, ff)
    ff_params = ff.get_ordered_params()
    ubps, params, masses, coords = afe.prepare_host_edge(ff_params, water_system, water_coords)

    host_coords = coords[:num_water_atoms]
    new_host_coords = minimizer.minimize_host_4d([mol], water_system, host_coords, ff, water_box)
    coords[:num_water_atoms] = new_host_coords

    return ubps, params, masses, coords, water_box


def generate_solvent_phase_samples(
    ubps,
    params,
    masses,
    coords,  # minimized_coords
    box,
    temperature,
    pressure=1.0,
    steps_per_batch=500,
    num_batches=10000,
    seed=None,
):
    """
    Generate samples in the solvent phase.
    """

    dt = 1.5e-3
    friction = 1.0
    interval = 5

    bps = []
    for p, bp in zip(params, ubps):
        bps.append(bp.bind(p))

    all_impls = [bp.bound_impl(np.float32) for bp in bps]

    intg_equil = lib.LangevinIntegrator(temperature, 1e-4, friction, masses, seed)
    intg_equil_impl = intg_equil.impl()

    # equilibration/minimization doesn't need a barostat
    equil_ctxt = custom_ops.Context(coords, np.zeros_like(coords), box, intg_equil_impl, all_impls, None)

    lamb = 0.0
    equil_schedule = np.ones(50000) * lamb
    equil_ctxt.multiple_steps(equil_schedule)

    x0 = equil_ctxt.get_x_t()
    v0 = np.zeros_like(x0)

    # production
    intg = lib.LangevinIntegrator(temperature, dt, friction, masses, seed)
    intg_impl = intg.impl()

    # reset impls
    all_impls = [bp.bound_impl(np.float32) for bp in bps]

    bond_list = get_bond_list(ubps[0])
    group_idxs = get_group_indices(bond_list)

    barostat = lib.MonteCarloBarostat(len(masses), pressure, temperature, group_idxs, interval, seed + 1)
    barostat_impl = barostat.impl(all_impls)

    ctxt = custom_ops.Context(x0, v0, box, intg_impl, all_impls, barostat_impl)

    lamb = 0.0
    lambda_windows = np.array([0.0])

    burn_in = 50000

    ctxt.multiple_steps_U(lamb, burn_in, lambda_windows, 0, 0)

    num_steps = steps_per_batch
    u_interval = steps_per_batch
    x_interval = steps_per_batch

    for _ in range(num_batches):
        _, _, _ = ctxt.multiple_steps_U(lamb, num_steps, lambda_windows, u_interval, x_interval)
        old_x_t = ctxt.get_x_t()
        old_v_t = ctxt.get_v_t()
        old_box = ctxt.get_box()
        new_xvb = yield ctxt.get_x_t(), ctxt.get_v_t(), ctxt.get_box()
        if new_xvb is not None:
            # tbd fix later
            np.testing.assert_array_equal(old_v_t, new_xvb.velocities)
            np.testing.assert_array_equal(old_box, new_xvb.box)
            ctxt.set_x_t(new_xvb.coords)


def align_sample(x_gas, x_solvent):
    """
    Return a rigidly transformed x_gas that is maximally aligned to x_solvent.
    """
    num_atoms = len(x_gas)

    xa = x_solvent[-num_atoms:]
    xb = x_gas

    assert xa.shape == xb.shape

    xb_new = rmsd.align_x2_unto_x1(xa, xb)
    return xb_new


def aligned_batch_propose(xvb, K, vacuum_samples, vacuum_log_weights):
    ligand_samples = sample_from_log_weights(vacuum_samples, vacuum_log_weights, K)

    x_solvent = xvb.coords
    v_solvent = xvb.velocities
    b_solvent = xvb.box

    new_xvbs = []

    # modify only ligand coordinates in the proposal
    for x_l in ligand_samples:
        x_l_aligned = align_sample(x_l, x_solvent)
        x_solvent_copy = x_solvent.copy()
        num_ligand_atoms = len(x_l)
        x_solvent_copy[-num_ligand_atoms:] = x_l_aligned
        new_xvbs.append(CoordsVelBox(x_solvent_copy, v_solvent, b_solvent))

    return new_xvbs
