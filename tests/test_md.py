import unittest
from itertools import product

import numpy as np

from jax.config import config; config.update("jax_enable_x64", True)
import jax
from jax import grad, numpy as jnp, jit

from timemachine.constants import BOLTZ
from timemachine.lib import custom_ops, potentials
from timemachine.integrator import langevin_coefficients, LangevinIntegrator

from common import prepare_nb_system

class TestContext(unittest.TestCase):

    def test_fwd_mode(self):
        """
        This test ensures that we can reverse-mode differentiate
        observables that are dU_dlambdas of each state. We provide
        adjoints with respect to each computed dU/dLambda.
        """

        np.random.seed(4321)

        N = 8
        D = 3

        x0 = np.random.rand(N,D).astype(dtype=np.float64)*2

        E = 2

        lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
        lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

        params, ref_nrg_fn, test_nrg = prepare_nb_system(
            x0,
            E,
            lambda_plane_idxs,
            lambda_offset_idxs,
            p_scale=3.0,
            # cutoff=0.5,
            cutoff=1.0
        )

        masses = np.random.rand(N)

        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        num_steps = 5
        temperature = 300
        dt = 2e-3
        friction = 0.0
        ca, cbs, ccs = langevin_coefficients(temperature, dt, friction, masses)

        # not convenient to simulate identical trajectories otherwise
        assert (ccs == 0).all()

        lamb = np.random.rand()
        lambda_us = []
        lambda_windows = np.array([lamb+0.05, lamb, lamb-0.05])

        def integrate_once_through(
            x_t,
            v_t,
            box,
            params):

            dU_dx_fn = jax.grad(ref_nrg_fn, argnums=(0,))
            dU_dp_fn = jax.grad(ref_nrg_fn, argnums=(1,))
            dU_dl_fn = jax.grad(ref_nrg_fn, argnums=(3,))

            all_du_dls = []
            all_du_dps = []
            all_xs = []
            all_du_dxs = []
            all_us = []
            all_lambda_us = []
            for step in range(num_steps):
                u = ref_nrg_fn(x_t, params, box, lamb)
                all_us.append(u)
                du_dl = dU_dl_fn(x_t, params, box, lamb)[0]
                all_du_dls.append(du_dl)
                du_dp = dU_dp_fn(x_t, params, box, lamb)[0]
                all_du_dps.append(du_dp)
                du_dx = dU_dx_fn(x_t, params, box, lamb)[0]
                all_du_dxs.append(du_dx)
                all_xs.append(x_t)

                lus = []
                for lamb_u in lambda_windows:
                    lus.append(ref_nrg_fn(x_t, params, box, lamb_u))

                all_lambda_us.append(lus)
                noise = np.random.randn(*v_t.shape)

                v_mid = v_t + np.expand_dims(cbs, axis=-1) * du_dx

                v_t = ca * v_mid + np.expand_dims(ccs, axis=-1) * noise
                x_t += 0.5 * dt * (v_mid + v_t)

                # note that we do not calculate the du_dl of the last frame.
            return all_xs, all_du_dxs, all_du_dps, all_du_dls, all_us, all_lambda_us

        box = np.eye(3)*3.0

        # when we have multiple parameters, we need to set this up correctly
        ref_all_xs, ref_all_du_dxs, ref_all_du_dps, ref_all_du_dls, ref_all_us, ref_all_lambda_us = integrate_once_through(
            x0,
            v0,
            box,
            params
        )

        intg = custom_ops.LangevinIntegrator(
            dt,
            ca,
            cbs,
            ccs,
            1234
        )

        bp = test_nrg.bind(params).bound_impl(precision=np.float64)
        bps = [bp]

        ctxt = custom_ops.Context(
            x0,
            v0,
            box,
            intg,
            bps
        )

        test_obs = custom_ops.AvgPartialUPartialParam(bp, 1)
        test_obs_f2 = custom_ops.AvgPartialUPartialParam(bp, 2)

        obs = [test_obs, test_obs_f2]

        for o in obs:
            ctxt.add_observable(o)

        test_avg_du_dp = test_obs.avg_du_dp()
        ref_init_du_dps = np.zeros_like(test_avg_du_dp)
        np.testing.assert_array_equal(test_avg_du_dp[:, 0], ref_init_du_dps[:, 0])
        np.testing.assert_array_equal(test_avg_du_dp[:, 1], ref_init_du_dps[:, 1])
        np.testing.assert_array_equal(test_avg_du_dp[:, 2], ref_init_du_dps[:, 2])

        for step in range(num_steps):
            if step < 2:
                # Until we have run 3 steps, std is 0
                test_std_du_dp = test_obs.std_du_dp()

                np.testing.assert_array_equal(test_std_du_dp[:, 0], ref_init_du_dps[:, 0])
                np.testing.assert_array_equal(test_std_du_dp[:, 1], ref_init_du_dps[:, 1])
                np.testing.assert_array_equal(test_std_du_dp[:, 2], ref_init_du_dps[:, 2])
            print("comparing step", step)
            test_x_t = ctxt.get_x_t()
            np.testing.assert_allclose(test_x_t, ref_all_xs[step])
            ctxt.step(lamb)
            test_v_t = ctxt.get_v_t()
            test_du_dx_t = ctxt._get_du_dx_t_minus_1()
            # test_u_t = ctxt._get_u_t_minus_1()
            # np.testing.assert_allclose(test_u_t, ref_all_us[step])
            np.testing.assert_allclose(test_du_dx_t, ref_all_du_dxs[step])


        ref_avg_du_dps = np.mean(ref_all_du_dps, axis=0)
        ref_std_du_dps = np.std(ref_all_du_dps, axis=0)

        # the fixed point accumulator makes it hard to converge some of these
        # if the derivative is super small - in which case they probably don't matter
        # anyways
        np.testing.assert_allclose(test_obs.avg_du_dp()[:, 0], ref_avg_du_dps[:, 0], 1.5e-6)
        np.testing.assert_allclose(test_obs.avg_du_dp()[:, 1], ref_avg_du_dps[:, 1], 1.5e-6)
        np.testing.assert_allclose(test_obs.avg_du_dp()[:, 2], ref_avg_du_dps[:, 2], 5e-5)

        np.testing.assert_allclose(test_obs.std_du_dp()[:, 0], ref_std_du_dps[:, 0], 1.5e-6)
        np.testing.assert_allclose(test_obs.std_du_dp()[:, 1], ref_std_du_dps[:, 1], 1.5e-6)
        np.testing.assert_allclose(test_obs.std_du_dp()[:, 2], ref_std_du_dps[:, 2], 5e-5)

        # test the multiple_steps method
        ctxt_2 = custom_ops.Context(
            x0,
            v0,
            box,
            intg,
            bps
        )

        lambda_schedule = np.ones(num_steps)*lamb

        du_dl_interval = 3
        x_interval = 2
        start_box = ctxt_2.get_box()
        test_du_dls, test_xs, test_boxes = ctxt_2.multiple_steps(lambda_schedule, du_dl_interval, x_interval)
        end_box = ctxt_2.get_box()

        np.testing.assert_allclose(
            test_du_dls,
            ref_all_du_dls[::du_dl_interval]
        )

        np.testing.assert_allclose(
            test_xs,
            ref_all_xs[::x_interval]
        )
        np.testing.assert_array_equal(start_box, end_box)
        for i in range(test_boxes.shape[0]):
            np.testing.assert_array_equal(start_box, test_boxes[i])
        self.assertEqual(test_boxes.shape[0], test_xs.shape[0])
        self.assertEqual(test_boxes.shape[1], D)
        self.assertEqual(test_boxes.shape[2], test_xs.shape[2])

        # test the multiple_steps_U method
        ctxt_3 = custom_ops.Context(
            x0,
            v0,
            box,
            intg,
            bps
        )

        u_interval = 3
 
        test_us, test_xs, test_boxes = ctxt_3.multiple_steps_U(
            lamb,
            num_steps,
            lambda_windows,
            u_interval,
            x_interval)

        np.testing.assert_array_almost_equal(
            ref_all_lambda_us[::u_interval],
            test_us
        )

        np.testing.assert_array_almost_equal(
            ref_all_xs[::x_interval],
            test_xs
        )

        test_us, test_xs, test_boxes = ctxt_3.multiple_steps_U(
            lamb,
            num_steps,
            np.array([], dtype=np.float64),
            u_interval,
            x_interval
        )

        assert test_us.shape == (2, 0)

class TestObservable(unittest.TestCase):

    def test_avg_potential_param_sizes_is_zero(self):
        np.random.seed(814)

        N = 8
        D = 3

        x0 = np.random.rand(N,D).astype(dtype=np.float64)*2

        masses = np.random.rand(N)

        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        num_steps = 3
        lambda_schedule = np.random.rand(num_steps)
        ca = np.random.rand()
        cbs = -np.random.rand(len(masses))/1
        ccs = np.zeros_like(cbs)

        dt = 2e-3
        lamb = np.random.rand()
        box = np.eye(3)*1.5

        intg = custom_ops.LangevinIntegrator(
            dt,
            ca,
            cbs,
            ccs,
            814
        )

        # Construct a 'bad' centroid restraint
        potential = potentials.CentroidRestraint(
            np.random.randint(N, size=5, dtype=np.int32),
            np.random.randint(N, size=5, dtype=np.int32),
            10.0,
            0.0
        )
        # Bind to empty params
        bp = potential.bind(np.zeros(0)).bound_impl(precision=np.float64)

        ctxt = custom_ops.Context(
            x0,
            v0,
            box,
            intg,
            [bp]
        )


        du_dp_obs = custom_ops.AvgPartialUPartialParam(bp, 1)
        ctxt.add_observable(du_dp_obs)

        test_avg_du_dp = du_dp_obs.avg_du_dp()
        zero_du_dps = np.zeros_like(test_avg_du_dp)

        for _ in range(num_steps):
            # For all steps, should get back an empty du_dp avg/std
            test_avg_du_dp = du_dp_obs.avg_du_dp()
            test_std_du_dp = du_dp_obs.std_du_dp()

            np.testing.assert_array_equal(test_avg_du_dp, zero_du_dps)
            np.testing.assert_array_equal(test_std_du_dp, zero_du_dps)

            ctxt.step(lamb)


def test_reference_langevin_integrator(threshold=1e-4):
    """Assert approximately canonical sampling of e^{-x^4 / kBT},
    for various settings of temperature, friction, timestep, and mass"""

    np.random.seed(2021)

    potential_fxn = lambda x: x ** 4
    force_fxn = lambda x: - 4 * x ** 3

    # loop over settings
    temperatures = [200, 300]
    frictions = [0.1, +np.inf]
    dts = [0.1, 0.15]
    masses = [1.0, 2.0]

    settings = list(product(temperatures, frictions, dts, masses))
    print(f'testing reference integrator for {len(settings)} combinations of settings:')
    print('(temperature, friction, dt, mass) -> histogram_mse')

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
        print(f'{(temperature, friction, dt, mass)}'.ljust(33), '->', histogram_mse)

        assert histogram_mse < threshold


def test_reference_langevin_integrator_with_custom_ops():
    """Run reference LangevinIntegrator on an alchemical ligand in vacuum under a few settings:
    * assert minimizer-like behavior when run at 0 temperature,
    * assert stability when run at room temperature"""

    np.random.seed(2021)

    # define a force fxn using a mix of optimized custom_ops and prototype-friendly Jax

    from testsystems.relative import hif2a_ligand_pair
    ff_params = hif2a_ligand_pair.ff.get_ordered_params()
    unbound_potentials, sys_params, masses, coords = hif2a_ligand_pair.prepare_vacuum_edge(ff_params)
    bound_potentials = [ubp.bind(params).bound_impl(np.float32) for (ubp, params) in
                        zip(unbound_potentials, sys_params)]
    box = 100 * np.eye(3)

    def custom_op_force_component(coords):
        du_dxs = np.array([bp.execute(coords, box, 0.5)[0] for bp in bound_potentials])
        return - np.sum(du_dxs, 0)

    def jax_restraint(coords):
        center = jnp.mean(coords, 0)
        return jnp.sum(center ** 4)

    @jit
    def jax_force_component(coords):
        return - grad(jax_restraint)(coords)

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


if __name__ == "__main__":
    unittest.main()
