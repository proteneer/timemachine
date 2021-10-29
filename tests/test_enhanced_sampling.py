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


from timemachine.potentials import bonded
import numpy as np
from md import enhanced
import test_ligands

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers


def get_ff():
    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
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
    ff = get_ff()
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
    assert np.mean((enhanced_torsions_lhs - vanilla_samples_lhs) ** 2) < 1e-2
