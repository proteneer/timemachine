from itertools import product

import jax
import numpy as np
import pytest
from jax import grad, jit
from jax import numpy as jnp

from timemachine.constants import BOLTZ
from timemachine.fe import utils
from timemachine.fe.single_topology import SingleTopology
from timemachine.ff import Forcefield
from timemachine.integrator import LangevinIntegrator
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


@pytest.mark.nocuda
def test_reference_langevin_integrator(threshold=1e-4):
    """Assert approximately canonical sampling of e^{-x^4 / kBT},
    for various settings of temperature, friction, timestep, and mass"""

    np.random.seed(2021)

    potential_fxn = lambda x: x**4
    force_fxn = lambda x: -4 * x**3

    # loop over settings
    temperatures = [200, 300]
    frictions = [0.1, +np.inf]
    dts = [0.1, 0.15]
    masses = [1.0, 2.0]

    all_combinations = list(product(temperatures, frictions, dts, masses))
    settings = [all_combinations[0], all_combinations[-1]]  # reduced to speed up CI
    print(f"testing reference integrator for {len(settings)} combinations of settings:")
    print("(temperature, friction, dt, mass) -> histogram_mse")

    for temperature, friction, dt, mass in settings:
        # generate n_production_steps * n_copies samples
        n_copies = 2500
        langevin = LangevinIntegrator(force_fxn, mass, temperature, dt, friction)

        x0, v0 = 0.1 * np.ones((2, n_copies))
        xs, vs = langevin.multiple_steps(x0, v0, n_steps=2500)
        samples = xs[10:].flatten()

        # summarize using histogram
        y_empirical, edges = np.histogram(samples, bins=100, range=(-2, +2), density=True)
        x_grid = (edges[1:] + edges[:-1]) / 2

        # compare with e^{-U(x) / kB T} / Z
        y = np.exp(-potential_fxn(x_grid) / (BOLTZ * temperature))
        y_ref = y / np.trapezoid(y, x_grid)

        histogram_mse = np.mean((y_ref - y_empirical) ** 2)
        print(f"{(temperature, friction, dt, mass)}".ljust(33), "->", histogram_mse)

        assert histogram_mse < threshold


@pytest.mark.nocuda
def test_reference_langevin_integrator_deterministic():
    """Asserts that trajectories are deterministic given a seed value"""
    force_fxn = lambda x: -4 * x**3
    langevin = LangevinIntegrator(force_fxn, masses=1.0, temperature=300.0, dt=0.1, friction=1.0)
    x0, v0 = 0.1 * jax.random.uniform(jax.random.PRNGKey(1), shape=(2, 5))

    def assert_deterministic(f):
        xs1, vs1 = f(1)

        # same seed should yield same result
        xs2, vs2 = f(1)
        np.testing.assert_array_equal(xs1, xs2)
        np.testing.assert_array_equal(vs1, vs2)

        # different seed should give different result
        xs3, vs3 = f(2)
        assert not np.allclose(xs2, xs3)
        assert not np.allclose(vs2, vs3)

    assert_deterministic(lambda seed: langevin.multiple_steps(x0, v0, rng=np.random.default_rng(seed)))
    assert_deterministic(lambda seed: langevin.multiple_steps_lax(jax.random.PRNGKey(seed), x0, v0))


@pytest.mark.nocuda
def test_reference_langevin_integrator_consistent():
    """
    Asserts that the result of the implementation based on jax.lax
    primitives is consistent with a simple for-loop implementation
    """
    force_fxn = lambda x: -4 * x**3
    langevin = LangevinIntegrator(force_fxn, masses=1.0, temperature=300.0, dt=0.1, friction=1.0)
    x0, v0 = 0.1 * jax.random.uniform(jax.random.PRNGKey(1), shape=(2, 5))
    key = jax.random.PRNGKey(1)

    def multiple_steps_reference(key, x, v, n_steps=1000):
        keys = jax.random.split(key, n_steps)
        xs, vs = [x], [v]

        for key in keys:
            new_x, new_v = langevin.step_lax(key, xs[-1], vs[-1])

            xs.append(new_x)
            vs.append(new_v)

        return np.array(xs), np.array(vs)

    xs1, vs1 = multiple_steps_reference(key, x0, v0)
    xs2, vs2 = langevin.multiple_steps_lax(key, x0, v0)

    # NOTE: result of the jax.lax implementation is NOT bitwise
    # equivalent to the pure Python implementation. This might be due
    # to loop-unrolling and reassociation optimizations performed by
    # XLA
    np.testing.assert_allclose(xs1, xs2)
    np.testing.assert_allclose(vs1, vs2)


def test_reference_langevin_integrator_with_custom_ops():
    """Run reference LangevinIntegrator on an alchemical ligand in vacuum under a few settings:
    * assert minimizer-like behavior when run at 0 temperature,
    * assert stability when run at room temperature"""

    seed = 2021
    np.random.seed(seed)
    temperature = 300
    st = get_hif2a_ligand_pair_single_topology()
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    vac_sys = st.setup_intermediate_state(0.5)
    x_a = utils.get_romol_conf(st.mol_a)
    x_b = utils.get_romol_conf(st.mol_b)
    coords = st.combine_confs(x_a, x_b)
    potentials = vac_sys.get_U_fns()
    masses = np.array(st.combine_masses())

    impls = [bp.to_gpu(np.float32).bound_impl for bp in potentials]
    box = 100 * np.eye(3)

    def custom_op_force_component(coords):
        du_dxs = np.array([bp.execute(coords, box)[0] for bp in impls])
        return -np.sum(du_dxs, 0)

    def jax_restraint(coords):
        center = jnp.mean(coords, 0)
        return jnp.sum(center**4)

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
