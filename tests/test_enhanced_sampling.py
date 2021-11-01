# set environment variable when running this file.
import os

# (ytz): not pretty, but this is needed to get XLA to be less stupid
# see https://github.com/google/jax/issues/1408 for more information
# needs to be set before xla/jax is initialized, and is set to a number
# suitable for running on CI
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"

from jax.config import config

config.update("jax_enable_x64", True)
import jax


from md.moves import MultipleTryMetropolis
from timemachine.potentials import bonded, nonbonded, rmsd
from timemachine.constants import BOLTZ
import numpy as np
from md import enhanced
import test_ligands

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
import pickle


from scipy.special import logsumexp


def get_ff_simple_charge():
    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)
    return ff


def get_ff_decharged():
    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_test_sampling.py").read())
    ff = Forcefield(ff_handlers)
    return ff


def test_gas_phase_importance_sampling():
    """
    This tests importance sampling in the gas-phase, where samples generated
    from a proposal distribution p_easy are reweighted into the target p_decharged.

    The intuition is that we can easily overcome various intramolecular barriers
    such as torsions, steric clashes, by turning off specific torsions and nonbonded
    terms.
    """

    mol = test_ligands.get_benzene()
    ff = get_ff_simple_charge()
    torsion_idxs = np.array([5, 6, 7, 8])
    temperature = 300

    state = enhanced.VacuumState(mol, ff)

    # samples are weighted by counts
    weighted_samples, log_weights = enhanced.generate_log_weighted_samples(
        mol,
        temperature,
        state.U_easy,
        state.U_decharged,
    )

    enhanced_samples = enhanced.sample_from_log_weights(weighted_samples, log_weights, 100000)

    @jax.jit
    def get_torsion(x_t):
        cijkl = x_t[torsion_idxs]
        return bonded.signed_torsion_angle(*cijkl)

    batch_torsion_fn = jax.vmap(get_torsion)
    enhanced_torsions = batch_torsion_fn(enhanced_samples)

    num_negative = np.sum(enhanced_torsions < 0)
    num_positive = np.sum(enhanced_torsions >= 0)

    # should be roughly 50/50
    assert np.abs(num_negative / (num_negative + num_positive) - 0.5) < 0.05

    # check that the distributions on the lhs look roughly identical

    enhanced_torsions_lhs, _ = np.histogram(enhanced_torsions, bins=50, range=(-np.pi, 0), density=True)
    enhanced_torsions_rhs, _ = np.histogram(enhanced_torsions, bins=50, range=(0, np.pi), density=True)

    # check for symmetry about theta=0
    assert np.mean((enhanced_torsions_lhs - enhanced_torsions_rhs[::-1]) ** 2) < 5e-2

    weighted_samples, log_weights = enhanced.generate_log_weighted_samples(
        mol,
        temperature,
        state.U_decharged,
        state.U_decharged,
        num_batches=5000,  # don't need as many batches since we don't have to prune
    )

    vanilla_samples = enhanced.sample_from_log_weights(weighted_samples, log_weights, 100000)

    vanilla_torsions = batch_torsion_fn(vanilla_samples)
    vanilla_samples_lhs, _ = np.histogram(vanilla_torsions, bins=50, range=(-np.pi, 0), density=True)

    # check for consistency with vanilla samples
    assert np.mean((enhanced_torsions_lhs - vanilla_samples_lhs) ** 2) < 5e-2


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


from typing import List
from md.states import CoordsVelBox


class AlignedMover(MultipleTryMetropolis):
    def __init__(self, K, vacuum_samples, vacuum_log_weights, log_prob_fn):

        self.vacuum_samples = vacuum_samples
        self.vacuum_log_weights = vacuum_log_weights
        self.batch_log_prob_fn = jax.vmap(log_prob_fn)

        super().__init__(K)

    def batch_propose(self, x: CoordsVelBox) -> List[CoordsVelBox]:

        ligand_samples = enhanced.sample_from_log_weights(self.vacuum_samples, self.vacuum_log_weights, self.K)

        x_solvent = x.coords
        v_solvent = x.velocities
        b_solvent = x.box

        new_xvbs = []

        # modify only ligand coordinates in the proposal
        for x_l in ligand_samples:
            x_l_aligned = align_sample(x_l, x_solvent)
            x_solvent_copy = x_solvent.copy()
            num_ligand_atoms = len(x_l)
            x_solvent_copy[-num_ligand_atoms:] = x_l_aligned
            new_xvbs.append(CoordsVelBox(x_solvent_copy, v_solvent, b_solvent))

        return new_xvbs

    def batch_log_prob(self, xvbs: List[CoordsVelBox]) -> List[float]:
        batch_coords = np.array([xvb.coords for xvb in xvbs])
        batch_boxes = np.array([xvb.box for xvb in xvbs])
        return self.batch_log_prob_fn(batch_coords, batch_boxes)


def test_condensed_phase_mtm():
    """
    Tests multiple-try metropolis in the condensed phase.
    """

    mol = test_ligands.get_benzene()
    ff = get_ff_decharged()

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_ligand_atoms = len(masses)

    cache_path = "cache.pkl"

    temperature = 300.0

    state = enhanced.VacuumState(mol, ff)

    if not os.path.exists(cache_path):
        print("Generating cache")

        vacuum_samples, vacuum_log_weights = enhanced.generate_log_weighted_samples(
            mol,
            temperature,
            state.U_easy,
            state.U_decharged,
        )

        with open(cache_path, "wb") as fh:
            pickle.dump([vacuum_samples, vacuum_log_weights], fh)

    with open(cache_path, "rb") as fh:
        vacuum_samples, vacuum_log_weights = pickle.load(fh)

    ubps, params, masses, coords, box = enhanced.get_solvent_phase_system(mol, ff)

    nb_potential = ubps[-1]
    beta = nb_potential.get_beta()
    cutoff = nb_potential.get_cutoff()
    nb_params = params[-1]

    params_i = nb_params[-num_ligand_atoms:]  # ligand params
    params_j = nb_params[:-num_ligand_atoms]  # water params

    # sanity check that charges on ligand is zero
    assert np.all(params_i[:, 0] == 0.0)

    kT = temperature * BOLTZ

    # (ytz): This only works for the decharged case since samples are generated from a decharged state.
    @jax.jit
    def log_prob_fn(x_solvent, box_solvent):
        x_water = x_solvent[:-num_ligand_atoms]  # water coords
        x_original = x_solvent[-num_ligand_atoms:]  # ligand coords
        U_k = nonbonded.nonbonded_block(x_original, x_water, box_solvent, params_i, params_j, beta, cutoff)
        return -U_k / kT

    @jax.jit
    def get_torsion(x_t):
        torsion_idxs = np.array([5, 6, 7, 8])
        cijkl = x_t[torsion_idxs]
        return bonded.signed_torsion_angle(*cijkl)

    batch_get_torsion = jax.jit(jax.vmap(get_torsion))
    vacuum_weights = np.exp(vacuum_log_weights - logsumexp(vacuum_log_weights))
    assert np.abs(np.sum(vacuum_weights) - 1) < 1e-5
    vacuum_torsions = batch_get_torsion(vacuum_samples)

    # verify that these are both bimodal
    # plt.hist(vacuum_torsions, density=True, weights=vacuum_weights, bins=50)
    # plt.show()

    assert np.abs(np.average(vacuum_torsions, weights=vacuum_weights)) < 0.1

    num_batches = 10000

    frozen_masses = np.copy(masses)
    frozen_masses[-num_ligand_atoms:] = np.inf

    all_torsions = []

    # test with both frozen masses and free masses
    for test_masses in [frozen_masses, masses]:

        sample_generator = enhanced.generate_solvent_phase_samples(
            ubps, params, test_masses, coords, box, temperature, num_batches=num_batches
        )

        next_x = None

        K = 200

        mover = AlignedMover(K, vacuum_samples, vacuum_weights, log_prob_fn)

        enhanced_torsions = []
        for iteration in range(num_batches):

            x_solvent, v_solvent, box_solvent = sample_generator.send(next_x)
            solvent_torsion = get_torsion(x_solvent[-num_ligand_atoms:])
            enhanced_torsions.append(solvent_torsion)

            next_x = mover.move(CoordsVelBox(x_solvent, v_solvent, box_solvent))

            print(
                f"frame {iteration} accepted {mover.n_accepted} proposed {mover.n_proposed} solvent_torsion {solvent_torsion}"
            )

        all_torsions.append(enhanced_torsions)

    all_torsions = np.asarray(all_torsions)

    # (ytz): useful for visualization, so please leave this comment here.
    from matplotlib import pyplot as plt

    # plt.hist(vanilla_solvent_torsions, bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.5)

    enhanced_torsions_lhs, _ = np.histogram(enhanced_torsions, bins=50, range=(-np.pi, 0), density=True)
    enhanced_torsions_rhs, _ = np.histogram(enhanced_torsions, bins=50, range=(0, np.pi), density=True)

    # check for symmetry about theta=0
    assert np.mean((enhanced_torsions_lhs - enhanced_torsions_rhs[::-1]) ** 2) < 5e-2

    vanilla_torsions = []
    # generate reference samples:
    for x_solvent, box_solvent in enhanced.generate_solvent_phase_samples(
        ubps, params, masses, coords, box, temperature, num_batches=num_batches
    ):
        solvent_torsion = get_torsion(x_solvent[-num_ligand_atoms:])
        vanilla_torsions.append(solvent_torsion)

    vanilla_torsions = np.asarray(vanilla_torsions)

    plt.hist(all_torsions[0], bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.5, label="enhanced_frozen")
    plt.hist(all_torsions[1], bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.5, label="enhanced_free")
    plt.hist(vanilla_torsions, bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.25, label="vanilla")
    plt.legend()
    plt.xlabel("torsion angle")
    plt.ylabel("density")
    plt.show()

    vanilla_samples_lhs, _ = np.histogram(vanilla_torsions, bins=50, range=(-np.pi, 0), density=True)

    # check for consistency with vanilla samples
    assert np.mean((enhanced_torsions_lhs - vanilla_samples_lhs) ** 2) < 5e-2


if __name__ == "__main__":
    test_condensed_phase_mtm()
