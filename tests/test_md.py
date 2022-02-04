import unittest

import numpy as np

from jax.config import config

config.update("jax_enable_x64", True)
import jax

from timemachine.lib import custom_ops, potentials
from timemachine.integrator import langevin_coefficients

from common import prepare_nb_system


class TestContext(unittest.TestCase):
    def test_set_and_get(self):
        """
        This test the setters and getters in the context.
        """

        np.random.seed(4321)

        N = 8
        D = 3

        x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

        E = 2

        lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
        lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

        params, _, test_nrg = prepare_nb_system(
            x0,
            E,
            lambda_plane_idxs,
            lambda_offset_idxs,
            p_scale=3.0,
            cutoff=1.0,
        )

        masses = np.random.rand(N)
        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        temperature = 300
        dt = 2e-3
        friction = 0.0
        ca, cbs, ccs = langevin_coefficients(temperature, dt, friction, masses)

        box = np.eye(3) * 3.0
        intg = custom_ops.LangevinIntegrator(dt, ca, cbs, ccs, 1234)

        bp = test_nrg.bind(params).bound_impl(precision=np.float64)
        bps = [bp]

        ctxt = custom_ops.Context(x0, v0, box, intg, bps)

        np.testing.assert_equal(ctxt.get_x_t(), x0)
        np.testing.assert_equal(ctxt.get_v_t(), v0)
        np.testing.assert_equal(ctxt.get_box(), box)

        new_x = np.random.rand(N, 3)
        ctxt.set_x_t(new_x)

        np.testing.assert_equal(ctxt.get_x_t(), new_x)

    def test_fwd_mode(self):
        """
        This test ensures that we can reverse-mode differentiate
        observables that are dU_dlambdas of each state. We provide
        adjoints with respect to each computed dU/dLambda.
        """

        np.random.seed(4321)

        N = 8
        D = 3

        x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

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
            cutoff=1.0,
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
        lambda_windows = np.array([lamb + 0.05, lamb, lamb - 0.05])

        def integrate_once_through(x_t, v_t, box, params):

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

        box = np.eye(3) * 3.0

        # when we have multiple parameters, we need to set this up correctly
        (
            ref_all_xs,
            ref_all_du_dxs,
            ref_all_du_dps,
            ref_all_du_dls,
            ref_all_us,
            ref_all_lambda_us,
        ) = integrate_once_through(x0, v0, box, params)

        intg = custom_ops.LangevinIntegrator(dt, ca, cbs, ccs, 1234)

        bp = test_nrg.bind(params).bound_impl(precision=np.float64)
        bps = [bp]

        ctxt = custom_ops.Context(x0, v0, box, intg, bps)

        for step in range(num_steps):
            print("comparing step", step)
            test_x_t = ctxt.get_x_t()
            np.testing.assert_allclose(test_x_t, ref_all_xs[step])
            ctxt.step(lamb)
            test_du_dx_t = ctxt._get_du_dx_t_minus_1()
            # test_u_t = ctxt._get_u_t_minus_1()
            # np.testing.assert_allclose(test_u_t, ref_all_us[step])
            np.testing.assert_allclose(test_du_dx_t, ref_all_du_dxs[step])

        ref_avg_du_dps = np.mean(ref_all_du_dps, axis=0)
        ref_std_du_dps = np.std(ref_all_du_dps, axis=0)

        # test the multiple_steps method
        ctxt_2 = custom_ops.Context(x0, v0, box, intg, bps)

        lambda_schedule = np.ones(num_steps) * lamb

        du_dl_interval = 3
        x_interval = 2
        start_box = ctxt_2.get_box()
        test_du_dls, test_xs, test_boxes = ctxt_2.multiple_steps(lambda_schedule, du_dl_interval, x_interval)
        end_box = ctxt_2.get_box()

        np.testing.assert_allclose(test_du_dls, ref_all_du_dls[::du_dl_interval])

        np.testing.assert_allclose(test_xs, ref_all_xs[::x_interval])
        np.testing.assert_array_equal(start_box, end_box)
        for i in range(test_boxes.shape[0]):
            np.testing.assert_array_equal(start_box, test_boxes[i])
        self.assertEqual(test_boxes.shape[0], test_xs.shape[0])
        self.assertEqual(test_boxes.shape[1], D)
        self.assertEqual(test_boxes.shape[2], test_xs.shape[2])

        # test the multiple_steps_U method
        ctxt_3 = custom_ops.Context(x0, v0, box, intg, bps)

        u_interval = 3

        test_us, test_xs, test_boxes = ctxt_3.multiple_steps_U(lamb, num_steps, lambda_windows, u_interval, x_interval)

        np.testing.assert_array_almost_equal(ref_all_lambda_us[::u_interval], test_us)

        np.testing.assert_array_almost_equal(ref_all_xs[::x_interval], test_xs)

        test_us, test_xs, test_boxes = ctxt_3.multiple_steps_U(
            lamb, num_steps, np.array([], dtype=np.float64), u_interval, x_interval
        )

        assert test_us.shape == (2, 0)


class TestObservable(unittest.TestCase):
    def test_avg_potential_param_sizes_is_zero(self):
        np.random.seed(814)

        N = 8
        D = 3

        x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

        masses = np.random.rand(N)

        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        num_steps = 3
        lambda_schedule = np.random.rand(num_steps)
        ca = np.random.rand()
        cbs = -np.random.rand(len(masses)) / 1
        ccs = np.zeros_like(cbs)

        dt = 2e-3
        lamb = np.random.rand()
        box = np.eye(3) * 1.5

        intg = custom_ops.LangevinIntegrator(dt, ca, cbs, ccs, 814)

        # Construct a 'bad' centroid restraint
        potential = potentials.CentroidRestraint(
            np.random.randint(N, size=5, dtype=np.int32), np.random.randint(N, size=5, dtype=np.int32), 10.0, 0.0
        )
        # Bind to empty params
        bp = potential.bind(np.zeros(0)).bound_impl(precision=np.float64)

        ctxt = custom_ops.Context(x0, v0, box, intg, [bp])

        for _ in range(num_steps):
            ctxt.step(lamb)


if __name__ == "__main__":
    unittest.main()
