import numpy as np
import pytest

from timemachine.fe.lambda_schedule import (
    bisect_lambda_schedule,
    interpolate_pre_optimized_protocol,
    validate_lambda_schedule,
)

pytestmark = [pytest.mark.nocuda]


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


@pytest.mark.parametrize("interval", [(0.0, 1.0), [0.25, 0.5]])
@pytest.mark.parametrize("n_windows", [1, 2, 3, 4, 8, 9, 16, 32, 48, 49])
def test_bisect_lambda_schedule(interval, n_windows):
    if n_windows < 2:
        with pytest.raises(AssertionError):
            bisect_lambda_schedule(n_windows)
        return

    schedule = bisect_lambda_schedule(n_windows, lambda_interval=interval)
    if interval == (0.0, 1.0):
        validate_lambda_schedule(schedule, n_windows)
    else:
        assert schedule[0] == interval[0]
        assert schedule[-1] == interval[1]
    if n_windows >= 3:
        mid_point = interval[0] + (interval[1] - interval[0]) / 2
        assert mid_point in schedule, "midpoint must be in schedule if at least 2 windows due to bisection"
    differences = np.diff(schedule)
    min_diff = np.min(differences)
    max_diff = np.max(differences)
    # For linspace the values can be off ever so slightly
    if not np.allclose(min_diff, max_diff):
        # As there are more windows, there end up being more 'gaps', but the values
        # should be binary, a smallest gap and a largest gap
        min_diffs = np.count_nonzero(differences == np.min(differences))
        max_diffs = np.count_nonzero(differences == np.max(differences))
        assert min_diffs + max_diffs == len(differences)

        lower_half = differences[: len(differences) // 2]
        upper_half = differences[len(differences) // 2 :]

        # At most two of the smallest values will be on 'one side'.
        # For example:
        # with 4 windows
        # [0.0, 1.0]
        # Insert 0.5 on the 'lower' half
        # [0.0, 0.5, 1.0]
        # insert 0.75 on the 'upper' half
        # [0.0, 0.5, 0.75, 1.0]
        # Which leaves differences of [0.5, 0.25, 0.25]
        # Which when split is [0.5] [0.25, 0.25]
        np.testing.assert_allclose(
            np.count_nonzero(lower_half == np.min(differences)),
            np.count_nonzero(upper_half == np.min(differences)),
            atol=2.0,
            rtol=0.0,
        )
    else:
        assert np.allclose(differences, differences[0])
