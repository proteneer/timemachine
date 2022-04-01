import numpy as np
from timemachine.fe.free_energy_rabfe import validate_lambda_schedule


def construct_relative_lambda_schedule(num_windows, nonbonded_cutoff=1.2):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    * manually optimized by YTZ
    * assumes nonbonded cutoff = 1.2 nm
        (since decoupling_distance = lambda * nonbonded_cutoff,
        this schedule will not be appropriate for nonbonded_cutoff != 1.2!)
    """
    assert nonbonded_cutoff == 1.2

    A = int(0.15 * num_windows)
    B = int(0.60 * num_windows)
    C = num_windows - A - B

    # optimizing the overlap based on eyeballing absolute hydration free energies
    # there's probably some better way to deal with this by inspecting the curvature
    lambda_schedule = np.concatenate(
        [
            np.linspace(0.00, 0.08, A, endpoint=False),
            np.linspace(0.08, 0.27, B, endpoint=False),
            np.linspace(0.27, 1.00, C, endpoint=True),
        ]
    )

    validate_lambda_schedule(lambda_schedule, num_windows)

    return lambda_schedule
