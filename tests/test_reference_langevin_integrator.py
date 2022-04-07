from itertools import product

import numpy as np
from jax import grad, jit
from jax import numpy as jnp

from timemachine.constants import BOLTZ
from timemachine.integrator import LangevinIntegrator
from timemachine.testsystems.relative import hif2a_ligand_pair


def test_reference_langevin_integrator(threshold=1e-4):
    """Assert approximately canonical sampling of e^{-x^4 / kBT},
    for various settings of temperature, friction, timestep, and mass"""

    np.random.seed(2021)

    potential_fxn = lambda x: x ** 4
    force_fxn = lambda x: -4 * x ** 3

    # loop over settings
    temperatures = [200, 300]
    frictions = [0.1, +np.inf]
    dts = [0.1, 0.15]
    masses = [1.0, 2.0]

    all_combinations = list(product(temperatures, frictions, dts, masses))
    settings = [all_combinations[0], all_combinations[-1]]  # reduced to speed up CI
    print(f"testing reference integrator for {len(settings)} combinations of settings:")
    print("(temperature, friction, dt, mass) -> histogram_mse")

    for (temperature, friction, dt, mass) in settings:
        # generate n_production_steps * n_copies samples
        n_copies = 10000
        langevin = LangevinIntegrator(force_fxn, mass, temperature, dt, friction)

        x0, v0 = 0.1 * np.ones((2, n_copies))
        xs, vs = langevin.multiple_steps(x0, v0, n_steps=5000)
        samples = xs[10:].flatten()

        # summarize using histogram
        y_empirical, edges = np.histogram(samples, bins=100, range=(-2, +2), density=True)
        x_grid = (edges[1:] + edges[:-1]) / 2

        # compare with e^{-U(x) / kB T} / Z
        y = np.exp(-potential_fxn(x_grid) / (BOLTZ * temperature))
        y_ref = y / np.trapz(y, x_grid)

        histogram_mse = np.mean((y_ref - y_empirical) ** 2)
        print(f"{(temperature, friction, dt, mass)}".ljust(33), "->", histogram_mse)

        assert histogram_mse < threshold


def test_reference_langevin_integrator_with_custom_ops():
    """Run reference LangevinIntegrator on an alchemical ligand in vacuum under a few settings:
    * assert minimizer-like behavior when run at 0 temperature,
    * assert stability when run at room temperature"""

    np.random.seed(2021)

    # define a force fxn using a mix of optimized custom_ops and prototype-friendly Jax
    rfe = hif2a_ligand_pair
    unbound_potentials, sys_params, masses = rfe.prepare_vacuum_edge(rfe.ff.get_ordered_params())
    coords = rfe.prepare_combined_coords()
    bound_potentials = [
        ubp.bind(params).bound_impl(np.float32) for (ubp, params) in zip(unbound_potentials, sys_params)
    ]
    box = 100 * np.eye(3)

    def custom_op_force_component(coords):
        du_dxs = np.array([bp.execute(coords, box, 0.5)[0] for bp in bound_potentials])
        return -np.sum(du_dxs, 0)

    def jax_restraint(coords):
        center = jnp.mean(coords, 0)
        return jnp.sum(center ** 4)

    @jit
    def jax_force_component(coords):
        return -grad(jax_restraint)(coords)

    def force(coords):
        return custom_op_force_component(coords) + jax_force_component(coords)

    def F_norm(coords):
        return np.linalg.norm(force(coords))

    # define a few integrators
    dt, temperature, friction = 1.5e-3, 300.0, 10.0

    # zero temperature, infinite friction
    # (gradient descent, with no momentum)
    descender = LangevinIntegrator(force, masses, 0.0, dt, np.inf)

    # zero temperature, finite friction
    # (gradient descent, with momentum)
    dissipator = LangevinIntegrator(force, masses, 0.0, dt, friction)

    # finite temperature, finite friction
    # (Langevin, with momentum)
    sampler = LangevinIntegrator(force, masses, temperature, dt, friction)

    # apply them
    x_0 = np.array(coords)
    v_0 = np.zeros_like(x_0)

    # assert gradient descent doesn't go far, but makes force norm much smaller
    xs, vs = descender.multiple_steps(x_0, v_0, n_steps=1000)
    force_reduction_factor = F_norm(xs[0]) / F_norm(xs[-1])
    assert force_reduction_factor > 50
    assert np.abs(xs[-1] - xs[0]).max() < 0.1

    # assert *inertial* gradient descent doesn't go far, but makes force norm much smaller
    xs, vs = dissipator.multiple_steps(x_0, v_0, n_steps=1000)
    force_reduction_factor = F_norm(xs[0]) / F_norm(xs[-1])
    assert force_reduction_factor > 50
    assert np.abs(xs[-1] - xs[0]).max() < 1

    x_min = xs[-1]

    # assert that finite temperature simulation initialized from x_min
    # (1) doesn't blow up
    # (2) goes uphill
    # (3) doesn't go very far
    xs, vs = sampler.multiple_steps(x_min, v_0, n_steps=1000)
    assert F_norm(xs[-1]) / len(coords) < 1e3
    assert F_norm(xs[-1]) > F_norm(xs[0])
    assert np.abs(xs[-1] - xs[0]).max() < 1
