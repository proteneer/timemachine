# Enhanced sampling protocols

# This file contains utility functions to generate samples in the gas-phase.

import numpy as np
from fe import topology
from fe.utils import get_romol_conf
import jax

from scipy.special import logsumexp
from timemachine.integrator import simulate
from timemachine.potentials import bonded, nonbonded
from timemachine.constants import BOLTZ

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
        self.bond_params, self.hb_potential = bt.parameterize_harmonic_bond(
            ff.hb_handle.params
        )
        self.angle_params, self.ha_potential = bt.parameterize_harmonic_angle(
            ff.ha_handle.params
        )
        self.proper_torsion_params, self.pt_potential = bt.parameterize_proper_torsion(
            ff.pt_handle.params
        )
        (
            self.improper_torsion_params,
            self.it_potential,
        ) = bt.parameterize_improper_torsion(ff.it_handle.params)
        self.nb_params, self.nb_potential = bt.parameterize_nonbonded(
            ff.q_handle.params, ff.lj_handle.params
        )

        self.box = None
        self.lamb = 0.0

    def _harmonic_bond_nrg(self, x):
        return bonded.harmonic_bond(
            x, self.bond_params, self.box, self.lamb, self.hb_potential.get_idxs()
        )

    def _harmonic_angle_nrg(self, x):
        return bonded.harmonic_angle(
            x, self.angle_params, self.box, self.lamb, self.ha_potential.get_idxs()
        )

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

        for idxs, params in zip(
            self.pt_potential.get_idxs(), self.proper_torsion_params
        ):
            _, j, k, _ = idxs
            if (j, k) in rotatable_bonds:
                print("turning off torsion", idxs)
                continue
            else:
                easy_proper_torsion_idxs.append(idxs)
                easy_proper_torsion_params.append(params)

        easy_proper_torsion_idxs = np.array(easy_proper_torsion_idxs, dtype=np.int32)
        easy_proper_torsion_params = np.array(
            easy_proper_torsion_params, dtype=np.float64
        )

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
    num_workers=None):
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
        num_batches+burn_in_batches,
        num_workers
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

    return xs_proposal.reshape(-1, num_atoms, 3,), log_weights

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