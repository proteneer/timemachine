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

    klass = custom_ops.WeightedRandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.WeightedRandomSampler_f64

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

    klass = custom_ops.WeightedRandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.WeightedRandomSampler_f64

    sampler = klass(size, seed)

    test_selection = sampler.sample(num_samples, weights)
    assert len(test_selection) == num_samples

    _, counts = np.unique(test_selection, return_counts=True)
    percentages = counts / num_samples
    np.testing.assert_allclose(expected_percentages, percentages, atol=0.05)


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("size, num_samples", [(500, 1500), (1000, 3000), (1000, 10000)])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_segmented_random_sampler(seed, size, num_samples, precision):
    rng = np.random.default_rng(seed)

    # Get a random probability vector.
    probs = rng.exponential(scale=1 / 250, size=(num_samples, size))

    klass = custom_ops.SegmentedWeightedRandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.SegmentedWeightedRandomSampler_f64

    x = np.arange(size)
    ref_selection = np.array([rng.choice(x, size=1, p=batch / np.sum(batch)) for batch in probs]).reshape(-1)
    sampler = klass(size, num_samples, seed)

    test_selection = sampler.sample(probs.tolist())
    assert len(test_selection) == num_samples

    ks, pv = ks_2samp(ref_selection, test_selection)
    assert ks < 0.05, (pv, ks)


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("num_samples", [8000])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_segmented_random_sampler_jagged_batches_simple_distributions(seed, num_samples, precision):
    """Make sure that jagged arrays are handled_correctly"""
    klass = custom_ops.SegmentedWeightedRandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.SegmentedWeightedRandomSampler_f64

    weights = []
    for _ in range(num_samples // 2):
        weights.append([7.5, 2.5])
        weights.append([3.3, 3.4, 3.3])
    expected_percentages = [[0.75, 0.25], [0.33, 0.34, 0.33]]

    largest_segment = len(max(weights, key=lambda x: len(x)))

    sampler = klass(largest_segment, len(weights), seed)

    test_selection = sampler.sample(weights)
    assert len(test_selection) == len(weights)

    # Check the pairs of two
    _, counts = np.unique(np.array(test_selection)[::2], return_counts=True)
    percentages = counts / (len(weights) // 2)
    np.testing.assert_allclose(expected_percentages[0], percentages, atol=0.05)

    # Check the triplet
    _, counts = np.unique(np.array(test_selection)[1:][::2], return_counts=True)
    percentages = counts / (len(weights) // 2)
    np.testing.assert_allclose(expected_percentages[1], percentages, atol=0.05)


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("num_samples", [1000])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_segmented_random_sampler_simple_distribution(seed, num_samples, precision):
    """Very basic test that doesn't rely on the KS test to verify correctness"""

    # Setup weights such that expected percentages are obvious
    weights = [[7.5, 2.5] for _ in range(num_samples)]
    expected_percentages = np.array([0.75, 0.25])

    klass = custom_ops.SegmentedWeightedRandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.SegmentedWeightedRandomSampler_f64

    sampler = klass(len(weights[0]), num_samples, seed)

    test_selection = sampler.sample(weights)
    assert len(test_selection) == num_samples

    _, counts = np.unique(test_selection, return_counts=True)
    percentages = counts / num_samples
    np.testing.assert_allclose(expected_percentages, percentages, atol=0.05)


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("num_samples", [1000])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_segmented_random_sampler_zero_probability(seed, num_samples, precision):
    """Make sure that if we zero out a probability we never sample that value"""

    # Setup weights such that expected percentages are obvious
    weights = [[0.0, 2.5] for _ in range(num_samples)]

    klass = custom_ops.SegmentedWeightedRandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.SegmentedWeightedRandomSampler_f64

    sampler = klass(len(weights[0]), num_samples, seed)

    test_selection = sampler.sample(weights)
    assert len(test_selection) == num_samples

    # All of the values will be the index that is non-zero
    assert np.all(np.array(test_selection) == 1)
