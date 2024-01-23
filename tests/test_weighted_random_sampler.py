import numpy as np
import pytest
from numpy.typing import NDArray

from timemachine.lib import custom_ops

pytestmark = [pytest.mark.memcheck]


def normalize_probabilities(vals: NDArray) -> NDArray:
    return vals / np.sum(vals)


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_segmented_random_sampler_validation(seed, precision):
    klass = custom_ops.SegmentedWeightedRandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.SegmentedWeightedRandomSampler_f64

    max_vals_per_segment = 5
    segments = 2

    sampler = klass(max_vals_per_segment, segments, seed)

    with pytest.raises(RuntimeError, match="number of segments don't match"):
        sampler.sample([[1.0]])

    with pytest.raises(RuntimeError, match="total values is greater than buffer size"):
        sampler.sample([[1.0] * (max_vals_per_segment + 1) for _ in range(segments)])

    with pytest.raises(RuntimeError, match="empty probability distribution not allowed"):
        sampler.sample([[5.0, 3.0, 10.0], []])

    with pytest.raises(RuntimeError, match="all values in log space must be finite and non-negative"):
        sampler.sample([[np.inf], [np.inf]])

    with pytest.raises(RuntimeError, match="all values in log space must be finite and non-negative"):
        sampler.sample([[-np.inf], [-np.inf]])

    with pytest.raises(RuntimeError, match="all values in log space must be finite and non-negative"):
        sampler.sample([[np.nan], [np.nan]])

    with pytest.raises(RuntimeError, match="all values in log space must be finite and non-negative"):
        sampler.sample([[1.0], [-0.1]])

    with pytest.raises(RuntimeError, match="all values in log space must be finite and non-negative"):
        sampler.sample([[1.0], [0.0]])


@pytest.mark.parametrize("seed", [2024, 2025, 2026, 2027])
@pytest.mark.parametrize("size, num_distributions, num_samples", [(50, 1000, 5), (500, 1000, 5)])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_segmented_random_sampler(seed, size, num_distributions, num_samples, precision):
    rng = np.random.default_rng(seed)

    # Get a random probability vector.
    probs = []
    for _ in range(num_distributions):
        # some rows will be "spiky" (small alpha), some rows will be more uniform (large alpha)
        alpha = rng.uniform(0.5, 10.0) * np.ones(size)
        prob_vec = rng.dirichlet(alpha)

        # Modify the probability to denormalize it
        unnormalized_prob_vec = prob_vec * rng.uniform(5, 100.0)
        probs.append(unnormalized_prob_vec)
    probs = np.array(probs)

    klass = custom_ops.SegmentedWeightedRandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.SegmentedWeightedRandomSampler_f64

    x = np.arange(size)
    ref_selection = np.array([rng.choice(x, size=num_samples, p=normalize_probabilities(batch)) for batch in probs])
    assert len(ref_selection) == num_distributions
    sampler = klass(size, num_distributions, seed)

    test_selection = [sampler.sample(probs.tolist()) for _ in range(num_samples)]
    assert len(test_selection) == num_samples
    test_selection = np.array(test_selection)
    test_selection = test_selection.transpose()

    _, test_counts = np.unique(test_selection, axis=-1, return_counts=True)
    test_percentages = test_counts / test_selection.shape[0]

    _, ref_counts = np.unique(ref_selection, axis=-1, return_counts=True)
    ref_percentages = ref_counts / ref_selection.shape[0]

    np.testing.assert_allclose(test_percentages, ref_percentages, rtol=0.45, atol=1e-5)


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("repeats", [10])
@pytest.mark.parametrize("num_samples", [8000])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_segmented_random_sampler_jagged_batches_simple_distributions(seed, repeats, num_samples, precision):
    """Make sure that jagged arrays are handled_correctly"""
    klass = custom_ops.SegmentedWeightedRandomSampler_f32
    if precision == np.float64:
        klass = custom_ops.SegmentedWeightedRandomSampler_f64

    rng = np.random.default_rng(seed)

    weights = []
    for _ in range(num_samples // 2):
        # Scale each weight by a uniform value
        weights.append(np.array([7.5, 2.5]) * rng.uniform() * 100.0)
        weights.append(np.array([3.3, 3.4, 3.3]) * rng.uniform() * 100.0)
    expected_percentages = [normalize_probabilities(weights[0]), normalize_probabilities(weights[1])]

    largest_segment = len(max(weights, key=lambda x: len(x)))

    sampler = klass(largest_segment, len(weights), seed)
    sampler_b = klass(largest_segment, num_samples, seed)

    # Verify that calling the same sampler multiple times produces passes the test
    for _ in range(repeats):
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
