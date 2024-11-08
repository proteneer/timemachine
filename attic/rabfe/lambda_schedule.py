import numpy as np

from timemachine.fe.lambda_schedule import validate_lambda_schedule

def construct_conversion_lambda_schedule(num_windows):
    lambda_schedule = np.linspace(0, 1, num_windows)
    validate_lambda_schedule(lambda_schedule, num_windows)
    return lambda_schedule


def construct_absolute_lambda_schedule_complex(num_windows, nonbonded_cutoff=DEFAULT_NONBONDED_CUTOFF):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    * manually optimized by YTZ
    * assumes nonbonded_cutoff = 1.2
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """
    assert nonbonded_cutoff == 1.2

    A = int(0.20 * num_windows)
    B = int(0.50 * num_windows)
    C = num_windows - A - B

    lambda_schedule = np.concatenate(
        [
            np.linspace(0.0, 0.1, A, endpoint=False),
            np.linspace(0.1, 0.3, B, endpoint=False),
            np.linspace(0.3, 1.0, C, endpoint=True),
        ]
    )

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule


def construct_absolute_lambda_schedule_solvent(num_windows, nonbonded_cutoff=DEFAULT_NONBONDED_CUTOFF):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    * manually optimized by YTZ
    * assumes nonbonded_cutoff = 1.2
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """
    assert nonbonded_cutoff == 1.2

    A = int(0.20 * num_windows)
    B = int(0.66 * num_windows)
    D = 1  # need only one window from 0.6 to 1.0
    C = num_windows - A - B - D

    # optimizing the overlap based on eyeballing absolute hydration free energies
    # there's probably some better way to deal with this by inspecting the curvature
    lambda_schedule = np.concatenate(
        [
            np.linspace(0.0, 0.08, A, endpoint=False),
            np.linspace(0.08, 0.27, B, endpoint=False),
            np.linspace(0.27, 0.50, C, endpoint=True),
            [1.0],
        ]
    )

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule