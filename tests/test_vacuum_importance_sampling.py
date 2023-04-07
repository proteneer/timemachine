# test importance sampling in the gas-phase.

import jax
import numpy as np
import pytest

from timemachine import testsystems
from timemachine.ff import Forcefield
from timemachine.md import enhanced
from timemachine.potentials import bonded

# from matplotlib import pyplot as plt


def get_ff_am1ccc():
    ff = Forcefield.load_default()
    return ff


# @pytest.mark.nightly(reason="This takes too long to run on CI")
@pytest.mark.nogpu
def test_vacuum_importance_sampling():
    """
    This tests importance sampling in the gas-phase, where samples generated
    from a proposal distribution p_easy are reweighted into the target p_decharged.

    The intuition is that we can easily overcome various intramolecular barriers
    such as torsions, steric clashes, by turning off specific torsions and nonbonded
    terms.
    """
    mol, torsion_idxs = testsystems.ligands.get_biphenyl()
    ff = get_ff_am1ccc()
    temperature = 300

    state = enhanced.VacuumState(mol, ff)

    seed = 2021

    num_samples = 1000000

    weighted_xv_samples, log_weights = enhanced.generate_log_weighted_samples(
        mol, temperature, state.U_easy, state.U_decharged, seed, num_batches=num_samples
    )

    enhanced_xv_samples = enhanced.sample_from_log_weights(weighted_xv_samples, log_weights, 1000000)
    enhanced_samples = np.array([x for (x, v) in enhanced_xv_samples])
    print("enhanced_samples", enhanced_samples.shape)

    @jax.jit
    def get_torsion(x_l):
        ci = x_l[torsion_idxs[:, 0]]
        cj = x_l[torsion_idxs[:, 1]]
        ck = x_l[torsion_idxs[:, 2]]
        cl = x_l[torsion_idxs[:, 3]]
        return bonded.signed_torsion_angle(ci, cj, ck, cl)

    batch_torsion_fn = jax.vmap(get_torsion)
    enhanced_torsions = batch_torsion_fn(enhanced_samples)
    print("enhanced_torsions", enhanced_torsions.shape)

    num_negative = np.sum(enhanced_torsions < 0)
    num_positive = np.sum(enhanced_torsions >= 0)
    print("num_negative", num_negative)
    print("num_positive", num_positive)

    # should be roughly 50/50
    assert np.abs(num_negative / (num_negative + num_positive) - 0.5) < 0.05

    # check that the distributions on the lhs look roughly identical
    # lhs is (-np.pi, 0) and rhs is (0, np.pi)
    enhanced_torsions_lhs, binsa = np.histogram(enhanced_torsions, bins=50, range=(-np.pi, 0), density=True)
    enhanced_torsions_rhs, binsb = np.histogram(enhanced_torsions, bins=50, range=(0, np.pi), density=True)
    print("bins_lhs", list(binsa))
    print("enhanced_torsions_lhs", enhanced_torsions_lhs.shape, list(enhanced_torsions_lhs))
    print("bins_rhs", list(binsb))
    print("enhanced_torsions_rhs", enhanced_torsions_rhs.shape, list(enhanced_torsions_rhs))
    print("int_lhs", np.sum(enhanced_torsions_lhs * np.diff(binsa)))
    print("int_rhs", np.sum(enhanced_torsions_rhs * np.diff(binsb)))

    # plt.hist(enhanced_torsions, bins=list(binsa) + list(binsb), histtype='stepfilled')
    # plt.savefig('hist.png')

    # check for symmetry about theta=0
    assert np.mean((enhanced_torsions_lhs - enhanced_torsions_rhs[::-1]) ** 2) < 5e-2
    weighted_xv_samples, log_weights = enhanced.generate_log_weighted_samples(
        mol,
        temperature,
        state.U_decharged,
        state.U_decharged,
        num_batches=5000,  # don't need as many batches since we don't have to prune
        seed=seed,
    )

    vanilla_xv_samples = enhanced.sample_from_log_weights(weighted_xv_samples, log_weights, 100000)
    vanilla_samples = np.array([x for (x, v) in vanilla_xv_samples])

    vanilla_torsions = batch_torsion_fn(vanilla_samples).reshape(-1)
    vanilla_samples_rhs, _ = np.histogram(vanilla_torsions, bins=50, range=(0, np.pi), density=True)

    # check for consistency with vanilla samples
    assert np.mean((enhanced_torsions_lhs - vanilla_samples_rhs) ** 2) < 5e-2
