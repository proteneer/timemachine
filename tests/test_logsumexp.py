"""Tests the SegmentedSumExp class that is used in the Cuda code to compute LogSumExp.
The host code computes the logsumexp directly"""

import numpy as np
import pytest
from scipy.special import logsumexp

from timemachine.lib import custom_ops


@pytest.mark.memcheck
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_segmented_cuda_logsumexp_validation(precision):
    max_values_per_segment = 5
    num_segments = 2
    if precision == np.float32:
        summer = custom_ops.SegmentedSumExp_f32(max_values_per_segment, num_segments)
    else:
        summer = custom_ops.SegmentedSumExp_f64(max_values_per_segment, num_segments)

    with pytest.raises(RuntimeError, match="number of segments must be less than or equal"):
        summer.logsumexp([[1.0] for _ in range(num_segments + 1)])

    with pytest.raises(RuntimeError, match="empty array not allowed"):
        summer.logsumexp([[1.0], []])

    with pytest.raises(RuntimeError, match="total values is greater than buffer size"):
        summer.logsumexp([[1.0] * (max_values_per_segment + 1) for _ in range(num_segments)])


@pytest.mark.memcheck
@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 0.0, 0.0), (np.float32, 1e-7, 1e-7)])
@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("loc", [1000, 0, -1000])
@pytest.mark.parametrize("shape", [(1, 2), (100, 100), (1000, 1000)])
def test_segmented_cuda_logsumexp(precision, atol, rtol, seed, loc, shape):
    rng = np.random.default_rng(seed)

    values = rng.normal(size=shape, loc=loc) * 1000.0

    if precision == np.float32:
        summer = custom_ops.SegmentedSumExp_f32(shape[1], shape[0])
    else:
        summer = custom_ops.SegmentedSumExp_f64(shape[1], shape[0])

    test_vals = summer.logsumexp(values)
    assert len(test_vals) == shape[0]
    # Verify that the results are deterministic
    np.testing.assert_array_equal(test_vals, summer.logsumexp(values))

    for test_val, vals in zip(test_vals, values):
        np.testing.assert_allclose(logsumexp(vals), test_val, atol=atol, rtol=rtol)


@pytest.mark.memcheck
@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 0.0, 0.0), (np.float32, 1e-7, 1e-7)])
@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("loc", [1000, 0, -1000])
@pytest.mark.parametrize("num_samples", [2, 5, 10, 100])
def test_segmented_cuda_logsumexp_ragged_arrays(precision, atol, rtol, seed, loc, num_samples):
    rng = np.random.default_rng(seed)
    max_values_per_segment = 100
    values = [rng.normal(size=rng.integers(1, max_values_per_segment), loc=loc) * 1000.0 for _ in range(num_samples)]

    klass = custom_ops.SegmentedSumExp_f64
    if precision == np.float32:
        klass = custom_ops.SegmentedSumExp_f32

    summer = klass(max_values_per_segment, num_samples)

    test_vals = summer.logsumexp(values)
    assert len(test_vals) == num_samples
    # Verify that the results are deterministic
    np.testing.assert_array_equal(test_vals, summer.logsumexp(values))

    unsegmented_version = klass(max_values_per_segment, 1)
    for test_val, vals in zip(test_vals, values):
        ref_logsumexp = logsumexp(vals)
        np.testing.assert_allclose(ref_logsumexp, test_val, atol=atol, rtol=rtol)
        np.testing.assert_allclose(ref_logsumexp, unsegmented_version.logsumexp([vals])[0], atol=atol, rtol=rtol)


@pytest.mark.memcheck
@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 0.0, 0.0), (np.float32, 1e-7, 1e-7)])
@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("count", [1, 5])
@pytest.mark.parametrize(
    "edge_case",
    [[1000.0, np.nan, 0.4], [100.0, np.inf, 5.0], [2024.0, -np.inf, 0.5], [np.inf, -np.inf, np.inf]],
)
def test_segmented_cuda_logsumexp_edge_cases(precision, atol, rtol, seed, count, edge_case):
    max_values_per_segment = 20
    rng = np.random.default_rng(seed)
    # Shuffle the ordering of the edge case, should be independent of ordering
    rng.shuffle(edge_case)
    # Only one sample will
    samples = [edge_case * count, rng.normal(size=max_values_per_segment).tolist()]

    rng.shuffle(samples)

    if precision == np.float32:
        summer = custom_ops.SegmentedSumExp_f32(max_values_per_segment, len(samples))
    else:
        summer = custom_ops.SegmentedSumExp_f64(max_values_per_segment, len(samples))

    test_vals = summer.logsumexp(samples)
    assert len(test_vals) == len(samples)
    # Verify that the results are deterministic
    np.testing.assert_array_equal(test_vals, summer.logsumexp(samples))

    for test_val, vals in zip(test_vals, samples):
        ref_logsumexp = logsumexp(vals)
        if np.isfinite(ref_logsumexp):
            np.testing.assert_allclose(ref_logsumexp, test_val, atol=atol, rtol=rtol)
        else:
            assert np.isnan(test_val)
