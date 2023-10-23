import numpy as np
import pytest
from scipy.stats import ks_2samp

from timemachine.lib import custom_ops

pytestmark = [pytest.mark.memcheck]


@pytest.mark.parametrize("seed", [2022])
@pytest.mark.parametrize("size, num_samples", [(500, 1500), (1000, 3000)])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_random_sampler(seed, size, num_samples, precision):
    rng = np.random.default_rng(seed)

    # Get a random probability vector.
    probs = rng.exponential(scale=1 / 250, size=size)
    prob_total = np.sum(probs)
    probs /= prob_total
    # Sort the probabilities to easily visualize
    probs = np.sort(probs)[::-1]

    klass = custom_ops.RandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.RandomSampler_f64

    x = np.arange(size)
    ref_selection = rng.choice(x, size=num_samples, p=probs)
    sampler = klass(size, seed)

    test_selection = sampler.sample(num_samples, probs)
    assert len(test_selection) == num_samples

    ks, pv = ks_2samp(ref_selection, test_selection)
    assert ks < 0.05, (pv, ks)


@pytest.mark.parametrize("seed", [2022, 2023])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_random_sampler_simple_distribution(seed, precision):
    """Very basic test that doesn't rely on the KS test to verify correctness"""
    num_samples = 1000

    # Setup weights such that expected percentages are obvious
    weights = np.array([7.5, 2.5])
    expected_percentages = np.array([0.75, 0.25])

    size = len(weights)

    klass = custom_ops.RandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.RandomSampler_f64

    sampler = klass(size, seed)

    test_selection = sampler.sample(num_samples, weights)
    assert len(test_selection) == num_samples

    _, counts = np.unique(test_selection, return_counts=True)
    percentages = counts / num_samples
    np.testing.assert_allclose(expected_percentages, percentages, atol=0.05)
