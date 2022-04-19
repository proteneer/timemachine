from timemachine.md.smc import (
    #sequential_monte_carlo,
    Resampler,
    effective_sample_size,
    null_resample,
    multinomial_resample,
    conditional_multinomial_resample,
    effective_sample_size,
    #fractional_effective_sample_size,
    
)
import numpy as np
from scipy.special import logsumexp
import pytest

def generate_log_weights(n):
    """sample log_weights ~ N(mu=mean, sigma=exp(log_stddev))
    where
    mean ~ N(mu=0, sigma=1)
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
    * total weight before and after resampling are consistent
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
    * ess >= 1
    * ess <= n_particles
    
    Also assert that:
    * on a vector of constant log weights `zeros(n_particles) + constant`,
        ess == n_particles, regardless of constant
    * on a vector `[0, -inf, -inf, ...] + constant`,
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
