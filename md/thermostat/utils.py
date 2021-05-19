import numpy as np
from timemachine.constants import BOLTZ
from simtk import unit


def sample_velocities(masses: unit.Quantity, temperature: unit.Quantity) -> np.array:
    """Sample Maxwell-Boltzmann velocities ~ N(0, sqrt(kB T / m)"""
    n_particles = len(masses)
    spatial_dim = 3

    v_unscaled = np.random.randn(n_particles, spatial_dim)

    # intended to be consistent with timemachine.integrator:langevin_coefficients
    sigma = np.sqrt(BOLTZ * temperature.value_in_unit.kelvin) * np.sqrt(1 / masses)
    v_scaled = v_unscaled * np.expand_dims(sigma, axis=1)

    assert v_scaled.shape == (n_particles, spatial_dim)

    return v_scaled
