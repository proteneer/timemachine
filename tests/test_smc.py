import numpy as np
import pytest
from jax import jit, vmap
from scipy.special import logsumexp

from timemachine.fe.reweighting import one_sided_exp
from timemachine.md.smc import (
    Resampler,
    conditional_multinomial_resample,
    effective_sample_size,
    get_endstate_samples_from_smc_result,
    multinomial_resample,
    null_resample,
    sequential_monte_carlo,
)
from timemachine.testsystems.gaussian1d import make_gaussian_testsystem


def generate_log_weights(n):
    """sample an array of n log_weights,
    log_weights[i] ~ N(mu=mean, sigma=exp(log_stddev)) for i in range(n)
    where
    mean       ~ N(mu=0, sigma=1)
    log_stddev ~ N(mu=0, sigma=1)
    """
    mean = np.random.randn()
    log_stddev = np.random.randn()
    stddev = np.exp(log_stddev)
    log_weights = stddev * np.random.randn(n) + mean
    return log_weights


@pytest.mark.parametrize("resampling_fxn", [null_resample, multinomial_resample, conditional_multinomial_resample])
def test_resampler(resampling_fxn: Resampler):
    """On a collection of random log_weights vectors of varying size, assert that:
    * total weight before and after resampling are consistent, and
    * resampled indices are all in range

    TODO: future refinements of this test might:
    * assert that estimates of expectations are self-consistent
        sum([exp(log_weights[i]) * f(x[i])]) ~= sum([exp(resampled_log_weights[i]) * f(x[resampled_indices[i]])])

    Notes
    -----
    * Non-requirement: len(log_weights) == len(resampled_log_weights)
        (e.g. in Bernoulli resampling, not yet implemented)
    * A nice place to look for generic test ideas might be
        [Webber, 2019] Unifying Sequential Monte Carlo with Resampling Matrices
        https://arxiv.org/abs/1903.12583
    """
    np.random.seed(2022)

    n_instances = 100
    for _ in range(n_instances):
        n_particles = np.random.randint(1, 100)
        log_weights = generate_log_weights(n_particles)

        # apply resampling_fxn
        resampled_indices, resampled_log_weights = resampling_fxn(log_weights)

        # assert sum of weights before equals sum of weights after
        log_sum_weight_before_resampling = logsumexp(log_weights)
        log_sum_weight_after_resampling = logsumexp(resampled_log_weights)
        np.testing.assert_almost_equal(log_sum_weight_after_resampling, log_sum_weight_before_resampling)

        # assert resampled_indices are all between 0 and n_particles
        assert min(resampled_indices) >= 0
        assert max(resampled_indices) < n_particles


def test_effective_sample_size():
    """On a collection of random log_weights vectors of varying size, assert that:
    * ess >= 1, and
    * ess <= n_particles

    Also assert that:
    * on a vector of constant log weights `zeros(n_particles) + constant`,
        ess == n_particles, regardless of constant
    * on a vector `[0, -inf, -inf, ...] + constant`, and
        ess == 1, regardless of constant
    """
    np.random.seed(2022)

    # random weights: ESS should be between 1 and n_particles
    n_instances = 100
    for _ in range(n_instances):
        n_particles = np.random.randint(1, 100)
        log_weights = generate_log_weights(n_particles)
        ess = effective_sample_size(log_weights)

        assert ess >= 1
        assert ess <= n_particles

    # constant weights: ESS should be n_particles
    for _ in range(n_instances):
        n_particles = np.random.randint(1, 100)
        constant = np.random.randn()
        log_weights = np.zeros(n_particles) + constant
        ess = effective_sample_size(log_weights)

        np.testing.assert_almost_equal(ess, n_particles)

    # degenerate weights: ESS should be 1
    for _ in range(n_instances):
        n_particles = np.random.randint(1, 100)
        constant = np.random.randn()
        index = np.random.randint(n_particles)
        log_weights = np.log(np.zeros(n_particles))
        log_weights[index] = constant

        ess = effective_sample_size(log_weights)

        np.testing.assert_almost_equal(ess, 1)


@pytest.mark.parametrize("resampling_fxn", [null_resample, multinomial_resample, conditional_multinomial_resample])
def test_sequential_monte_carlo(resampling_fxn: Resampler):
    """Run SMC with the desired resampling_fxn on a Gaussian 1D test system, and assert that:
    * running estimates of the free energy as a fxn of lambda match analytical free energies, and
    * endstate samples have expected mean and stddev
    """
    np.random.seed(2022)

    # construct test system
    target_mean, target_log_sigma = 1, -2
    params = np.array([target_mean, target_log_sigma])  # TODO: randomize / parameterize
    n_particles = 10000
    n_windows = 100
    lambdas = np.linspace(0, 1, n_windows)
    u_fxn, _, sample, reduced_free_energy = make_gaussian_testsystem()
    ref_free_energies = np.array([reduced_free_energy(lam, params) for lam in lambdas])
    ref_delta_fs = ref_free_energies - ref_free_energies[0]

    # prepare inputs for SMC
    samples = sample(0, params, n_particles).flatten()
    assert samples.shape == (n_particles,)

    vec_u_fxn = jit(vmap(u_fxn, in_axes=(0, None, None)))

    def log_prob(xs, lam):
        xs = np.array(xs)
        return -vec_u_fxn(xs, lam, params)

    def propagate(xs, lam):
        """random-walk metropolis with N(0, proposal_scale(lam)) proposals"""
        n = len(xs)
        xs = np.array(xs)
        assert xs.shape == (n,)

        proposal_scale = np.exp(lam * target_log_sigma)  # to match annealed_gaussian_def  # TODO: extract
        proposals = xs + proposal_scale * np.random.randn(n)
        assert proposals.shape == xs.shape

        # metropolis
        log_probs_before = log_prob(xs, lam)
        log_probs_after = log_prob(proposals, lam)
        log_accept_probs = np.minimum(log_probs_after - log_probs_before, 0)
        accept_probs = np.exp(log_accept_probs)
        assert accept_probs.shape == (n,)

        # accept / reject
        test_vals = np.random.rand(n)
        accept_mask = accept_probs >= test_vals
        updated = np.where(accept_mask, proposals, xs)
        assert updated.shape == xs.shape

        return updated

    # apply SMC
    result_dict = sequential_monte_carlo(samples, lambdas, propagate, log_prob, resampling_fxn)

    # compute running delta_f estimates
    log_weights_traj = result_dict["log_weights_traj"]
    assert log_weights_traj.shape == (n_windows, n_particles)
    running_estimates = np.array([one_sided_exp(-log_weights) for log_weights in log_weights_traj])

    # assert all running estimates are within ~ 0.1 kT of reference
    np.testing.assert_array_almost_equal(running_estimates, ref_delta_fs, decimal=1)

    # assert this isn't a "no-op" test
    constant_predictions = np.ones(n_windows) * np.mean(ref_delta_fs)
    mse_vs_smc_estimates = np.mean((ref_delta_fs - running_estimates) ** 2)
    mse_vs_constant = np.mean((ref_delta_fs - constant_predictions) ** 2)
    assert mse_vs_constant / mse_vs_smc_estimates > 1000

    # extract endstate samples
    samples_0, samples_1 = get_endstate_samples_from_smc_result(result_dict, propagate, lambdas)

    # assert these have close to the expected mean and variance
    mean_0, stddev_0 = np.mean(samples_0), np.std(samples_0)
    mean_1, stddev_1 = np.mean(samples_1), np.std(samples_1)

    np.testing.assert_almost_equal(mean_0, 0, decimal=1)
    np.testing.assert_almost_equal(np.log(stddev_0), np.log(1), decimal=1)

    np.testing.assert_almost_equal(mean_1, target_mean, decimal=1)
    np.testing.assert_almost_equal(np.log(stddev_1), target_log_sigma, decimal=1)
