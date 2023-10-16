import numpy as np
import pytest
from scipy.special import logsumexp

from timemachine.lib import custom_ops


@pytest.mark.memcheck
@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 1e-8, 1e-8), (np.float32, 2e-5, 1e-5)])
@pytest.mark.parametrize("seed", list(range(2)))
@pytest.mark.parametrize("loc", [1000, 0, -1000])
@pytest.mark.parametrize("size", [2, 100, 10000])
def test_cuda_logsumexp(precision, atol, rtol, seed, loc, size):
    rng = np.random.default_rng(seed)

    values = rng.normal(size=size, loc=loc) * 1000.0

    reference_val = logsumexp(values)

    if precision == np.float32:
        summer = custom_ops.LogSumExp_f32(len(values))
    else:
        summer = custom_ops.LogSumExp_f64(len(values))

    test_val = summer.sum(values)
    assert len(test_val) == 1  # Have to return numpy array to get values back out

    np.testing.assert_allclose(reference_val, test_val)
