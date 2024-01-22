import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import ks_2samp

from timemachine.lib import custom_ops

pytestmark = [pytest.mark.memcheck]


def normalize_probabilities(vals: NDArray) -> NDArray:
    return vals / np.sum(vals)


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
    ref_selection = np.array([rng.choice(x, size=1, p=normalize_probabilities(batch)) for batch in probs]).reshape(-1)
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

    rng = np.random.default_rng(seed)

    weights = []
    for _ in range(num_samples // 2):
        #  Scale each weight by a uniform value
        weights.append(np.array([7.5, 2.5]) * rng.uniform() * 100.0)
        weights.append(np.array([3.3, 3.4, 3.3]) * rng.uniform() * 100.0)
    expected_percentages = [normalize_probabilities(weights[0]), normalize_probabilities(weights[1])]

    largest_segment = len(max(weights, key=lambda x: len(x)))

    sampler = klass(largest_segment, len(weights), seed)
    sampler_b = klass(largest_segment, num_samples, seed)

    test_selection = sampler.sample(weights)
    assert len(test_selection) == len(weights)

    # Should produce an identical result
    np.testing.assert_array_equal(sampler_b.sample(weights), test_selection)

    # Check the pairs of two by grabbing the even indices
    _, counts = np.unique(np.array(test_selection)[::2], return_counts=True)
    percentages = counts / (len(weights) // 2)
    np.testing.assert_allclose(expected_percentages[0], percentages, atol=0.05)

    # Check the set of three values by grabbing the odd indices
    _, counts = np.unique(np.array(test_selection)[1:][::2], return_counts=True)
    percentages = counts / (len(weights) // 2)
    np.testing.assert_allclose(expected_percentages[1], percentages, atol=0.05)


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("num_samples", [1000])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_segmented_random_sampler_simple_distribution(seed, num_samples, precision):
    """Very basic test that doesn't rely on the KS test to verify correctness"""

    rng = np.random.default_rng(seed)
    # Setup weights such that expected percentages are obvious
    weights = [np.array([7.5, 2.5]) * rng.uniform() * 100.0 for _ in range(num_samples)]
    expected_percentages = normalize_probabilities(weights[0])

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
