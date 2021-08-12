import numpy as np
from ff.handlers.bonded import HarmonicAngleHandler, HarmonicBondHandler, ImproperTorsionHandler, ProperTorsionHandler
from ff.handlers.nonbonded import LennardJonesHandler, AM1CCCHandler

default_learning_rates = {
    HarmonicBondHandler:    np.zeros(2),    # k, length
    HarmonicAngleHandler:   np.zeros(2),    # k, angle
    ProperTorsionHandler:   np.zeros(3),    # k, phase, periodicity
    ImproperTorsionHandler: np.zeros(3),    # k, phase, periodicity
    AM1CCCHandler:          np.ones(1),     # charge increment
    LennardJonesHandler:    np.ones(2),     # epsilon, sigma
}

def learning_rates_like_params(
        ordered_handles, ordered_params, learning_rates=default_learning_rates
):
    """Get a list of arrays of same shape as ordered_params, but containing
    learning rates"""

    ordered_learning_rates = []
    for handle, params in zip(ordered_handles, ordered_params):
        lr_row = learning_rates[handle.__class__]
        lr_array = np.array([lr_row] * len(params))

        if handle == AM1CCCHandler:
            # outlier: not in shape (n_types, params_per_type)
            lr_array = lr_array.flatten()

        assert lr_array.shape == params.shape, f"{lr_array.shape}, {params.shape}"
        ordered_learning_rates.append(lr_array)

    return ordered_learning_rates
