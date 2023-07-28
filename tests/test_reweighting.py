import numpy as np
import pymbar
import pytest

from timemachine.fe.reweighting import one_sided_exp


@pytest.mark.nogpu
def test_one_sided_exp():
    """assert consistency with pymbar.EXP on random instances + instances containing +inf work"""

    np.random.seed(2022)
    num_instances = 100

    for _ in range(num_instances):
        # instance parameters
        num_works = np.random.randint(1, 100)
        mean = np.random.randn() * 10
        stddev = np.exp(np.random.randn())

        # random instance
        reduced_works = np.random.randn(num_works) * stddev + mean

        # compare estimates
        pymbar_estimate, _ = pymbar.EXP(reduced_works)
        tm_estimate = one_sided_exp(reduced_works)

        assert np.isclose(tm_estimate, pymbar_estimate)

    # also check +inf
    reduced_works = np.array([+np.inf, 0])
    assert np.isclose(one_sided_exp(reduced_works), pymbar.EXP(reduced_works)[0])
