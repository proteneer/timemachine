import functools
import unittest

import numpy as np

from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp


from timemachine.lib import custom_ops
# from timemachine.lib import potentials
from timemachine.potentials import bonded, nonbonded

from common import GradientTest
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
        B = 5
        A = 0
        T = 0
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
            cutoff=1.5
        )

        masses = np.random.rand(N)

        v0 = np.random.rand(x0.shape[0], x0.shape[1])
        N = len(masses)

        num_steps = 5
        lambda_schedule = np.random.rand(num_steps)
        ca = np.random.rand()
        cbs = -np.random.rand(len(masses))/1
        ccs = np.zeros_like(cbs)

        dt = 2e-3
        lamb = np.random.rand()

        def loss_fn(du_dls):
            return jnp.sum(du_dls*du_dls)/du_dls.shape[0]

        def sum_loss_fn(du_dls):
            du_dls = np.sum(du_dls, axis=0)
            return jnp.sum(du_dls*du_dls)/du_dls.shape[0]

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
            for step in range(num_steps):
                u = ref_nrg_fn(x_t, params, box, lamb)
                all_us.append(u)
                du_dl = dU_dl_fn(x_t, params, box, lamb)[0]
                all_du_dls.append(du_dl)
                du_dp = dU_dp_fn(x_t, params, box, lamb)[0]
                all_du_dps.append(du_dp)
                du_dx = dU_dx_fn(x_t, params, box, lamb)[0]
                all_du_dxs.append(du_dx)
                v_t = ca*v_t + np.expand_dims(cbs, axis=-1)*du_dx
                x_t = x_t + v_t*dt
                all_xs.append(x_t)
                # note that we do not calculate the du_dl of the last frame.

            return all_xs, all_du_dxs, all_du_dps, all_du_dls, all_us

        box = np.eye(3)*1.5

        # when we have multiple parameters, we need to set this up correctly
        ref_all_xs, ref_all_du_dxs, ref_all_du_dps, ref_all_du_dls, ref_all_us = integrate_once_through(
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

        # test_obs_du_dl = custom_ops.AvgPartialUPartialLambda(bps, 1)
        # test_obs_f2_du_dl = custom_ops.AvgPartialUPartialLambda(bps, 2)
        # test_obs_f3_du_dl = custom_ops.FullPartialUPartialLambda(bps, 2)

        obs = [test_obs, test_obs_f2]

        for o in obs:
            ctxt.add_observable(o)

        for step in range(num_steps):
            print("comparing step", step)
            ctxt.step(lamb)
            test_x_t = ctxt.get_x_t()
            test_v_t = ctxt.get_v_t()
            test_du_dx_t = ctxt._get_du_dx_t_minus_1()
            # test_u_t = ctxt._get_u_t_minus_1()
            # np.testing.assert_allclose(test_u_t, ref_all_us[step])
            np.testing.assert_allclose(test_du_dx_t, ref_all_du_dxs[step])
            np.testing.assert_allclose(test_x_t, ref_all_xs[step])

        ref_avg_du_dls = np.mean(ref_all_du_dls, axis=0)
        ref_avg_du_dls_f2 = np.mean(ref_all_du_dls[::2], axis=0)

        # np.testing.assert_allclose(test_obs_du_dl.avg_du_dl(), ref_avg_du_dls)
        # np.testing.assert_allclose(test_obs_f2_du_dl.avg_du_dl(), ref_avg_du_dls_f2)

        # full_du_dls = test_obs_f3_du_dl.full_du_dl()
        # assert len(full_du_dls) == np.ceil(num_steps/2)
        # np.testing.assert_allclose(np.mean(full_du_dls), ref_avg_du_dls_f2)

        ref_avg_du_dps = np.mean(ref_all_du_dps, axis=0)
        ref_avg_du_dps_f2 = np.mean(ref_all_du_dps[::2], axis=0)

        # the fixed point accumulator makes it hard to converge some of these
        # if the derivative is super small - in which case they probably don't matter
        # anyways
        np.testing.assert_allclose(test_obs.avg_du_dp()[:, 0], ref_avg_du_dps[:, 0], 1.5e-6)
        np.testing.assert_allclose(test_obs.avg_du_dp()[:, 1], ref_avg_du_dps[:, 1], 1.5e-6)
        np.testing.assert_allclose(test_obs.avg_du_dp()[:, 2], ref_avg_du_dps[:, 2], 5e-5)

        # test the multiple_steps method
        intg_2 = custom_ops.LangevinIntegrator(
            dt,
            ca,
            cbs,
            ccs,
            1234
        )

        ctxt_2 = custom_ops.Context(
            x0,
            v0,
            box,
            intg,
            bps
        )

        lambda_schedule = np.ones(num_steps)*lamb

        du_dl_freq = 3
        test_du_dls = ctxt_2.multiple_steps(lambda_schedule, du_dl_freq)

        np.testing.assert_allclose(
            test_du_dls,
            ref_all_du_dls[::du_dl_freq]
        )


    # def test_multiple_steps(self):
    #     """
    #     Test that the multiple step context behaves as expected when storing du_dls
    #     """

    #     np.random.seed(4321)

    #     N = 8
    #     B = 5
    #     A = 0
    #     T = 0
    #     D = 3

    #     x0 = np.random.rand(N,D).astype(dtype=np.float64)*2

    #     E = 2

    #     lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
    #     lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

    #     params, ref_nrg_fn, test_nrg = prepare_nb_system(
    #         x0,
    #         E,
    #         lambda_plane_idxs,
    #         lambda_offset_idxs,
    #         p_scale=3.0,
    #         # cutoff=0.5,
    #         cutoff=1.5
    #     )

    #     masses = np.random.rand(N)

    #     v0 = np.random.rand(x0.shape[0], x0.shape[1])
    #     N = len(masses)

    #     num_steps = 5
    #     lambda_schedule = np.random.rand(num_steps)
    #     ca = np.random.rand()
    #     cbs = -np.random.rand(len(masses))/1
    #     ccs = np.zeros_like(cbs)

    #     dt = 2e-3

    #     box = np.eye(3)*1.5

    #     intg = custom_ops.LangevinIntegrator(
    #         dt,
    #         ca,
    #         cbs,
    #         ccs,
    #         1234
    #     )

    #     bp = test_nrg.bind(params).bound_impl(precision=np.float64)
    #     bps = [bp]

    #     ctxt = custom_ops.Context(
    #         x0,
    #         v0,
    #         box,
    #         intg,
    #         bps
    #     )

    #     store_du_dl_freq = 2

    #     full_du_dl_obs = custom_ops.FullPartialUPartialLambda(bps, store_du_dl_freq)

    #     ctxt.add_observable(full_du_dl_obs)

    #     lambda_schedule = np.random.rand(10)

    #     for lamb in lambda_schedule:
    #         ctxt.step(lamb)

    #     ref_du_dls = full_du_dl_obs.full_du_dl()

    #     # need to re-seed the integrator
    #     intg_2 = custom_ops.LangevinIntegrator(
    #         dt,
    #         ca,
    #         cbs,
    #         ccs,
    #         1234
    #     )

    #     ctxt_2 = custom_ops.Context(
    #         x0,
    #         v0,
    #         box,
    #         intg_2,
    #         bps
    #     )

    #     test_du_dls = ctxt_2.multiple_steps(lambda_schedule, store_du_dl_freq)

    #     np.testing.assert_equal(ref_du_dls, test_du_dls)


if __name__ == "__main__":
    unittest.main()
