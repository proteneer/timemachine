from timemachine.md.smc import (
    #sequential_monte_carlo,
    Resampler,
    null_resample,
    multinomial_resample,
    conditional_multinomial_resample,
    #effective_sample_size,
    #fractional_effective_sample_size,
    
)
import numpy as np
from scipy.special import logsumexp


def assert_resampler_correct(resampling_fxn: Resampler):
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

        # sample log_weights ~ N(mu=mean, sigma=exp(log_stddev))
        mean = np.random.randn()
        log_stddev = np.random.randn()
        stddev = np.exp(log_stddev)
        log_weights = stddev * np.random.randn(n_particles) + mean
        
        # apply resampling_fxn
        resampled_indices, resampled_log_weights = resampling_fxn(log_weights)

        # assert sum of weights before equals sum of weights after
        log_sum_weight_before_resampling = logsumexp(log_weights)
        log_sum_weight_after_resampling = logsumexp(resampled_log_weights)
        np.testing.assert_almost_equal(log_sum_weight_after_resampling, log_sum_weight_before_resampling)

        # assert resampled_indices are all between 0 and n_particles
        assert min(resampled_indices) >= 0
        assert max(resampled_indices) < n_particles


def test_resamplers():
    resamplers = [null_resample, multinomial_resample, conditional_multinomial_resample]
    for resampler in resamplers:
        assert_resampler_correct(resampler)