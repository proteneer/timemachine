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
    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_dc.py").read())
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

        weighted_vacuum_samples, log_weights = enhanced.generate_log_weighted_samples(
            mol,
            temperature,
            state.U_easy,
            state.U_decharged,
        )

        with open(cache_path, "wb") as fh:
            pickle.dump([weighted_vacuum_samples, log_weights], fh)

    with open(cache_path, "rb") as fh:
        weighted_vacuum_samples, log_weights = pickle.load(fh)

    ubps, params, masses, coords, box = enhanced.get_solvent_phase_system(mol, ff)

    nb_potential = ubps[-1]
    beta = nb_potential.get_beta()
    cutoff = nb_potential.get_cutoff()
    nb_params = params[-1]

    params_i = nb_params[-num_ligand_atoms:]  # ligand params
    params_j = nb_params[:-num_ligand_atoms]  # water params

    # sanity check that charges on ligand is zero
    assert np.all(params_i[:, 0] == 0.0)

    @jax.jit
    def before_U_k(x_solvent, box_solvent):
        x_water = x_solvent[:-num_ligand_atoms]  # water coords
        x_original = x_solvent[-num_ligand_atoms:]  # ligand coords
        U_k = nonbonded.nonbonded_block(x_original, x_water, box_solvent, params_i, params_j, beta, cutoff)
        return U_k

    # batch align a set of gas molecules to the coordinates of the ligand in solvent
    # then compute the energy
    def after_U_k(x_gas, x_solvent, box_solvent):
        x_gas_aligned = align_sample(x_gas, x_solvent)  # align gas phase conformer
        x_water = x_solvent[:-num_ligand_atoms]  # water coords
        U_k = nonbonded.nonbonded_block(x_gas_aligned, x_water, box_solvent, params_i, params_j, beta, cutoff)
        return U_k

    batch_after_U_k_fn = jax.jit(jax.vmap(after_U_k, in_axes=(0, None, None)))
    kT = temperature * BOLTZ

    all_delta_us_unique = []

    @jax.jit
    def get_torsion(x_t):
        torsion_idxs = np.array([5, 6, 7, 8])
        cijkl = x_t[torsion_idxs]
        return bonded.signed_torsion_angle(*cijkl)

    batch_get_torsion = jax.jit(jax.vmap(get_torsion))
    vacuum_weights = np.exp(log_weights - logsumexp(log_weights))
    assert np.abs(np.sum(vacuum_weights) - 1) < 1e-5
    vacuum_torsions = batch_get_torsion(weighted_vacuum_samples)

    # verify that these are both bimodal
    # plt.hist(vacuum_torsions, density=True, weights=vacuum_weights, bins=50)
    # plt.show()

    assert np.abs(np.average(vacuum_torsions, weights=vacuum_weights)) < 0.1

    all_weights = []

    num_batches = 2000
    sample_generator = enhanced.generate_solvent_phase_samples(
        ubps, params, masses, coords, box, temperature, num_batches=num_batches
    )

    next_x = None

    K = 100
    num_accept = 0
    num_reject = 0

    enhanced_torsions = []
    for iteration in range(num_batches):

        x_solvent, box_solvent = sample_generator.send(next_x)
        before_U_k_sample = before_U_k(x_solvent, box_solvent)
        log_pi_x = -before_U_k_sample / kT

        solvent_torsion = get_torsion(x_solvent[-num_ligand_atoms:])
        enhanced_torsions.append(solvent_torsion)

        # Multiple Try Metropolis (MTM) on K samples

        # 1) generate K samples from vacuum
        gas_samples_yi = enhanced.sample_from_log_weights(weighted_vacuum_samples, log_weights, K)
        # gas_samples_yi = weighted_vacuum_samples[
        # np.random.choice(np.arange(num_vacuum_samples), size=K, p=vacuum_weights)
        # ]
        # 2) compute energies of vacuum solvents aligned to ligand in x_solvent
        after_U_y_i_samples = batch_after_U_k_fn(gas_samples_yi, x_solvent, box_solvent)

        # 3) compute log weights of the numerator
        log_pi_yi = -after_U_y_i_samples / kT
        normalized_pi_yi = np.exp(log_pi_yi - logsumexp(log_pi_yi))

        # 4) using log weights, pick one of the yi as the candidate
        new_y = gas_samples_yi[np.random.choice(np.arange(K), p=normalized_pi_yi)]

        # 5) let y_solvent be replacement of x_solvent ligand with new_y
        new_y_aligned = align_sample(new_y, x_solvent)
        y_solvent = np.copy(x_solvent)
        y_solvent[-num_ligand_atoms:] = new_y_aligned

        # 6) Generate another random set of samples from vacuum_samples
        gas_samples_x_i_sub_1 = enhanced.sample_from_log_weights(weighted_vacuum_samples, log_weights, K - 1)

        # 7) Compute log weights for the denominator
        after_U_x_i_samples = batch_after_U_k_fn(gas_samples_x_i_sub_1, y_solvent, box_solvent)
        log_pi_x_i_sub_1 = -after_U_x_i_samples / kT
        log_pi_xi_combined = np.concatenate([log_pi_x_i_sub_1, [log_pi_x]])

        # 8) Compute the log ratio
        log_ratio = logsumexp(log_pi_yi) - logsumexp(log_pi_xi_combined)
        ratio = np.exp(log_ratio)

        # decide if we accept or reject
        if np.random.rand() < ratio:
            num_accept += 1
            # use new frame
            next_x = y_solvent
        else:
            num_reject += 1
            # use old frame
            next_x = x_solvent

        print(
            f"frame {iteration} ratio {ratio} log_ratio {log_ratio} accepts {num_accept} rejects {num_reject} solvent_torsion {solvent_torsion}"
        )

    enhanced_torsions = np.asarray(enhanced_torsions)

    # (ytz): useful for visualization, so please leave this comment here.
    # from matplotlib import pyplot as plt
    # plt.hist(vanilla_solvent_torsions, bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.5)
    # plt.hist(mtm_solvent_torsions, bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.5)
    # plt.show()

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

    vanilla_samples_lhs, _ = np.histogram(vanilla_torsions, bins=50, range=(-np.pi, 0), density=True)

    # check for consistency with vanilla samples
    assert np.mean((enhanced_torsions_lhs - vanilla_samples_lhs) ** 2) < 5e-2
