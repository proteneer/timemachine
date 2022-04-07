import numpy as np
import pytest

from timemachine.fe.lambda_schedule import (
    construct_pre_optimized_absolute_lambda_schedule_solvent,
    interpolate_pre_optimized_protocol,
    validate_lambda_schedule,
)


def test_validate_lambda_schedule():
    """check that assertions fail when they should"""

    # Want to test 2 sizes, the latter is the one currently used in RABFE
    for K in [50, 64]:
        good_lambda_schedule = np.linspace(0, 1, K)
        reversed_schedule = good_lambda_schedule[::-1]
        truncated_schedule = good_lambda_schedule[1:]

        validate_lambda_schedule(good_lambda_schedule, K)

        with pytest.raises(AssertionError):
            validate_lambda_schedule(reversed_schedule, K)

        with pytest.raises(AssertionError):
            validate_lambda_schedule(truncated_schedule, K - 1)

        with pytest.raises(AssertionError):
            validate_lambda_schedule(truncated_schedule, K)


def test_interpolate_pre_optimized_protocol():
    linear = np.linspace(0, 1, 50)
    nonlinear = np.linspace(0, 1, 64) ** 2

    for sched in [linear, nonlinear]:
        # recover ~exactly the initial schedule
        K = len(sched)
        sched_prime = interpolate_pre_optimized_protocol(sched, K)
        assert np.allclose(sched, sched_prime)

        # produce valid protocols when downsampling
        reduced = interpolate_pre_optimized_protocol(sched, K // 2)
        validate_lambda_schedule(reduced, K // 2)


def test_pre_optimized_solvent_decoupling_schedule():
    for K in [10, 50, 64, 128]:
        sched = construct_pre_optimized_absolute_lambda_schedule_solvent(K)
        validate_lambda_schedule(sched, K)
