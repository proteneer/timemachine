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
        summer = custom_ops.SegmentedLogSumExp_f32(max_values_per_segment, num_segments)
    else:
        summer = custom_ops.SegmentedLogSumExp_f64(max_values_per_segment, num_segments)

    with pytest.raises(RuntimeError, match="number of segments don't match"):
        summer.sum([[1.0]])

    with pytest.raises(RuntimeError, match="empty array not allowed"):
        summer.sum(
            [
                [
                    1.0,
                ],
                [],
            ]
        )

    with pytest.raises(RuntimeError, match="total values is greater than buffer size"):
        summer.sum([[1.0] * (max_values_per_segment + 1) for _ in range(num_segments)])


@pytest.mark.memcheck
@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 1e-8, 1e-8), (np.float32, 2e-5, 1e-5)])
@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("loc", [1000, 0, -1000])
@pytest.mark.parametrize("shape", [(1, 2), (100, 100), (1000, 1000)])
def test_segmented_cuda_logsumexp(precision, atol, rtol, seed, loc, shape):
    rng = np.random.default_rng(seed)

    values = rng.normal(size=shape, loc=loc) * 1000.0

    if precision == np.float32:
        summer = custom_ops.SegmentedLogSumExp_f32(shape[1], shape[0])
    else:
        summer = custom_ops.SegmentedLogSumExp_f64(shape[1], shape[0])

    test_vals = summer.sum(values)
    assert len(test_vals) == shape[0]
    np.testing.assert_array_equal(test_vals, summer.sum(values))

    for test_val, vals in zip(test_vals, values):
        np.testing.assert_allclose(logsumexp(vals), test_val)


@pytest.mark.memcheck
@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 1e-8, 1e-8), (np.float32, 2e-5, 1e-5)])
@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("loc", [1000, 0, -1000])
@pytest.mark.parametrize("num_samples", [2, 5, 10, 100])
def test_segmented_cuda_logsumexp_ragged_arrays(precision, atol, rtol, seed, loc, num_samples):
    rng = np.random.default_rng(seed)
    max_values_per_segment = 100
    values = [rng.normal(size=rng.integers(1, max_values_per_segment), loc=loc) * 1000.0 for _ in range(num_samples)]

    if precision == np.float32:
        summer = custom_ops.SegmentedLogSumExp_f32(max_values_per_segment, num_samples)
    else:
        summer = custom_ops.SegmentedLogSumExp_f64(max_values_per_segment, num_samples)

    test_vals = summer.sum(values)
    assert len(test_vals) == num_samples
    np.testing.assert_array_equal(test_vals, summer.sum(values))

    for test_val, vals in zip(test_vals, values):
        np.testing.assert_allclose(logsumexp(vals), test_val)
