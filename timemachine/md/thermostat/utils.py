import numpy as np
from numpy.typing import NDArray

from timemachine.constants import BOLTZ


def sample_velocities(masses: NDArray, temperature: float, seed: int) -> NDArray:
    """Sample Maxwell-Boltzmann velocities ~ N(0, sqrt(kB T / m)

    Parameters
    ----------

    masses:
        Array of masses

    temperature:
        float representing temperature in kelvin

    seed:
        integer to use to use as seed

    Returns
    -------
    (N, 3) velocities array, where N is the length of masses
    """
    n_particles = len(masses)
    spatial_dim = 3

    rng = np.random.default_rng(seed)
    v_unscaled = rng.standard_normal(size=(n_particles, spatial_dim))

    # intended to be consistent with timemachine.integrator:langevin_coefficients
    sigma = np.sqrt(BOLTZ * temperature) * np.sqrt(1 / masses)
    v_scaled = v_unscaled * np.expand_dims(sigma, axis=1)

    assert v_scaled.shape == (n_particles, spatial_dim)

    return v_scaled
