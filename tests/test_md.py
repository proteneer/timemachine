import unittest

import jax
import numpy as np
import pytest
from common import prepare_nb_system

from timemachine.integrator import langevin_coefficients
from timemachine.lib import custom_ops

pytestmark = [pytest.mark.memcheck]


class TestContext(unittest.TestCase):
    def test_multiple_steps_store_interval(self):
        np.random.seed(2022)

        N = 8
        D = 3

        x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

        E = 2

        params, potential = prepare_nb_system(x0, E, p_scale=3.0, cutoff=1.0)
        test_nrg = potential.to_gpu()

        masses = np.random.rand(N)
        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        temperature = 300
        dt = 2e-3
        friction = 0.0

        box = np.eye(3) * 3.0
        intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

        bp = test_nrg.bind(params).bound_impl(precision=np.float64)
        bps = [bp]

        ctxt = custom_ops.Context(x0, v0, box, intg, bps)
        test_xs, test_boxes = ctxt.multiple_steps(10, 10)
        assert len(test_xs) == 1
        assert len(test_xs) == len(test_boxes)
        # We should not get out the input frame
        assert np.any(np.not_equal(x0, test_xs[0]))

        # The current coordinates should match, as the number of steps and the interval match
        np.testing.assert_array_equal(test_xs[0], ctxt.get_x_t())
        _, _ = bps[0].execute(test_xs[0], test_boxes[0])

        # Given an interval greater than the number of steps, return empty arrays
        test_xs, test_boxes = ctxt.multiple_steps(10, 100)
        assert len(test_xs) == 0
        assert len(test_boxes) == 0

        # Given interval of 0, return the last frame
        test_xs, test_boxes = ctxt.multiple_steps(10, 0)
        assert len(test_xs) == 1
        assert len(test_boxes) == 1

        np.testing.assert_array_equal(test_xs[0], ctxt.get_x_t())
        _, _ = bps[0].execute(test_xs[0], test_boxes[0])

    def test_multiple_steps_U_store_interval(self):
        np.random.seed(2022)

        N = 8
        D = 3

        x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

        E = 2

        params, potential = prepare_nb_system(x0, E, p_scale=3.0, cutoff=1.0)
        test_nrg = potential.to_gpu()

        masses = np.random.rand(N)
        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        temperature = 300
        dt = 2e-3
        friction = 0.0

        box = np.eye(3) * 3.0
        intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

        bp = test_nrg.bind(params).bound_impl(precision=np.float64)
        bps = [bp]

        ctxt = custom_ops.Context(x0, v0, box, intg, bps)
        test_us, test_xs, test_boxes = ctxt.multiple_steps_U(10, 10, 10)
        assert len(test_xs) == 1
        assert test_us.shape == (1,)
        assert len(test_xs) == len(test_boxes)
        # We should not get out the input frame
        assert np.any(np.not_equal(x0, test_xs[0]))

        # The current coordinates should match, as the number of steps and the interval match
        np.testing.assert_array_equal(test_xs[0], ctxt.get_x_t())
        _, test_frame_u = bps[0].execute(test_xs[0], test_boxes[0])
        np.testing.assert_array_equal(test_us[0], test_frame_u)

        # Given an interval greater than the number of steps, return empty arrays
        test_us, test_xs, test_boxes = ctxt.multiple_steps_U(10, 100, 100)
        assert len(test_xs) == 0
        assert len(test_us) == 0
        assert len(test_boxes) == 0

        # Given interval of 0, return the last frame
        test_us, test_xs, test_boxes = ctxt.multiple_steps_U(10, 0, 0)
        assert len(test_xs) == 1
        assert test_us.shape == (1,)
        assert len(test_boxes) == 1

        np.testing.assert_array_equal(test_xs[0], ctxt.get_x_t())
        _, test_frame_u = bps[0].execute(test_xs[0], test_boxes[0])
        np.testing.assert_array_equal(test_us[0], test_frame_u)

    def test_set_and_get(self):
        """
        This test the setters and getters in the context.
        """

        np.random.seed(4321)

        N = 8
        D = 3

        x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

        E = 2

        params, potential = prepare_nb_system(x0, E, p_scale=3.0, cutoff=1.0)
        test_nrg = potential.to_gpu()

        masses = np.random.rand(N)
        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        temperature = 300
        dt = 2e-3
        friction = 0.0

        box = np.eye(3) * 3.0
        intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

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

        params, potential = prepare_nb_system(
            x0,
            E,
            p_scale=3.0,
            # cutoff=0.5,
            cutoff=1.0,
        )
        ref_nrg_fn = potential.to_reference()
        test_nrg = potential.to_gpu()

        masses = np.random.rand(N)

        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        num_steps = 12
        temperature = 300
        dt = 1.5e-3
        friction = 0.0
        ca, cbs, ccs = langevin_coefficients(temperature, dt, friction, masses)

        # not convenient to simulate identical trajectories otherwise
        assert (ccs == 0).all()

        def integrate_once_through(x_t, v_t, box, params):

            dU_dx_fn = jax.grad(ref_nrg_fn, argnums=(0,))
            dU_dp_fn = jax.grad(ref_nrg_fn, argnums=(1,))

            all_du_dps = []
            all_xs = []
            all_du_dxs = []
            all_us = []

            def compute_reference_values():
                u = ref_nrg_fn(x_t, params, box)
                all_us.append(u)
                du_dp = dU_dp_fn(x_t, params, box)[0]
                all_du_dps.append(du_dp)
                du_dx = dU_dx_fn(x_t, params, box)[0]
                all_du_dxs.append(du_dx)
                all_xs.append(x_t)

            for step in range(num_steps):
                compute_reference_values()

                noise = np.random.randn(*v_t.shape)
                force = -all_du_dxs[-1]
                v_mid = v_t + np.expand_dims(cbs, axis=-1) * force

                v_t = ca * v_mid + np.expand_dims(ccs, axis=-1) * noise
                x_t += 0.5 * dt * (v_mid + v_t)

            # Compute them for the last set of coords
            compute_reference_values()
            return all_xs, all_du_dxs, all_du_dps, all_us

        box = np.eye(3) * 3.0

        # when we have multiple parameters, we need to set this up correctly
        ref_all_xs, ref_all_du_dxs, ref_all_du_dps, ref_all_us = integrate_once_through(x0, v0, box, params)

        intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

        bp = test_nrg.bind(params).bound_impl(precision=np.float64)
        bps = [bp]

        ctxt = custom_ops.Context(x0, v0, box, intg, bps)
        ctxt.initialize()
        for step in range(num_steps):
            print("comparing step", step)
            test_x_t = ctxt.get_x_t()
            np.testing.assert_allclose(test_x_t, ref_all_xs[step])
            test_du_dx_t, _ = bp.execute(test_x_t, box)
            ctxt.step()
            # np.testing.assert_allclose(test_u_t, ref_all_us[step])
            np.testing.assert_allclose(test_du_dx_t, ref_all_du_dxs[step])
        ctxt.finalize()
        # test the multiple_steps method
        ctxt_2 = custom_ops.Context(x0, v0, box, intg, bps)

        x_interval = 2
        start_box = ctxt_2.get_box()
        test_xs, test_boxes = ctxt_2.multiple_steps(num_steps, x_interval)
        end_box = ctxt_2.get_box()

        np.testing.assert_allclose(test_xs, ref_all_xs[x_interval::x_interval])
        np.testing.assert_array_equal(start_box, end_box)
        for i in range(test_boxes.shape[0]):
            np.testing.assert_array_equal(start_box, test_boxes[i])
        self.assertEqual(test_boxes.shape[0], test_xs.shape[0])
        self.assertEqual(test_boxes.shape[1], D)
        self.assertEqual(test_boxes.shape[2], test_xs.shape[2])

        # test the multiple_steps_U method
        ctxt_3 = custom_ops.Context(x0, v0, box, intg, bps)

        u_interval = 3

        test_us, test_xs, test_boxes = ctxt_3.multiple_steps_U(num_steps, u_interval, x_interval)
        np.testing.assert_array_almost_equal(ref_all_us[u_interval::u_interval], test_us)

        np.testing.assert_array_almost_equal(ref_all_xs[x_interval::x_interval], test_xs)


if __name__ == "__main__":
    unittest.main()
