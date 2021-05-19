import numpy as np
from timemachine.constants import BOLTZ
from simtk import unit
from timemachine.lib import custom_ops
from typing import Tuple


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


def run_thermostatted_md(
        integrator_impl, bound_impls,
        x: np.array, box: np.array, v: np.array,
        lam: float, n_steps=5) -> Tuple[np.array, np.array]:

    ctxt = custom_ops.Context(
        x.coords,
        v,
        box,
        integrator_impl,
        bound_impls
    )

    # arguments: lambda_schedule, du_dl_interval, x_interval
    _, _ = ctxt.multiple_steps(lam * np.ones(n_steps), 0, 0)
    x_t = ctxt.get_x_t()
    v_t = ctxt.get_v_t()

    return x_t, v_t

