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
from common import prepare_lj_system, prepare_es_system

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

        lj_params, ref_lj_fn, test_lj = prepare_lj_system(
            x0,
            E,
            lambda_plane_idxs,
            lambda_offset_idxs,
            p_scale=3.0,
            cutoff=100.0,
            precision=np.float64       
        )

        # charge_params, ref_es_fn, test_es_ctor = prepare_es_system(
        #     x0,
        #     E,
        #     lambda_plane_idxs,
        #     lambda_offset_idxs,
        #     p_scale=10.0,
        #     cutoff=1000.0,
        #     precision=precision       
        # )


        # test_es = test_es_ctor()

        masses = np.random.rand(N)

        v0 = np.random.rand(x0.shape[0], x0.shape[1])
        N = len(masses)

        num_steps = 5
        # num_steps = 1
        lambda_schedule = np.random.rand(num_steps)
        ca = np.random.rand()
        cbs = -np.random.rand(len(masses))/1
        ccs = np.zeros_like(cbs)

        dt = np.random.rand()
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
            lj_params):

            # ref_es_impl = functools.partial(ref_es_fn, charge_params=charge_params)
            ref_lj_impl = ref_lj_fn

            def ref_total_nrg_fn(*args):
                nrgs = []
                # print("args", args)
                # for fn in [ref_bond_impl, ref_restr_impl, ref_lj_impl]:
                # for fn in [ref_lj_impl, ref_es_impl]:
                for fn in [ref_lj_impl]:
                    nrgs.append(fn(*args))
                return jnp.sum(nrgs)

            dU_dx_fn = jax.grad(ref_total_nrg_fn, argnums=(0,))
            dU_dp_fn = jax.grad(ref_total_nrg_fn, argnums=(1,))
            dU_dl_fn = jax.grad(ref_total_nrg_fn, argnums=(3,))

            all_du_dls = []
            all_du_dps = []
            all_xs = []
            all_du_dxs = []
            for step in range(num_steps):
                du_dl = dU_dl_fn(x_t, lj_params, box, lamb)[0]
                all_du_dls.append(du_dl)
                du_dp = dU_dp_fn(x_t, lj_params, box, lamb)[0]
                all_du_dps.append(du_dp)
                du_dx = dU_dx_fn(x_t, lj_params, box, lamb)[0]
                all_du_dxs.append(du_dx)
                v_t = ca*v_t + np.expand_dims(cbs, axis=-1)*du_dx
                x_t = x_t + v_t*dt
                all_xs.append(x_t)
                # note that we do not calculate the du_dl of the last frame.

            return all_xs, all_du_dxs, all_du_dps, all_du_dls

        box = np.eye(3)*1.5

        # when we have multiple parameters, we need to set this up correctly
        ref_all_xs, ref_all_du_dxs, ref_all_du_dps, ref_all_du_dls = integrate_once_through(
            x0,
            v0,
            box,
            # charge_params,
            lj_params
        )

        intg = custom_ops.LangevinIntegrator(
            dt,
            ca,
            cbs,
            ccs,
            1234
        )

        lj_bp = test_lj.bind(lj_params).bound_impl()

        bps = [lj_bp]

        ctxt = custom_ops.Context(
            x0,
            v0,
            box,
            intg,
            bps
        )

        lj_obs = custom_ops.AvgPartialUPartialParam(lj_bp, 1)
        lj_obs_f2 = custom_ops.AvgPartialUPartialParam(lj_bp, 2)

        lj_obs_du_dl = custom_ops.AvgPartialUPartialLambda(bps, 1)
        lj_obs_f2_du_dl = custom_ops.AvgPartialUPartialLambda(bps, 2)

        obs = [lj_obs, lj_obs_f2, lj_obs_du_dl, lj_obs_f2_du_dl]

        for o in obs:
            ctxt.add_observable(o)

        for step in range(num_steps):
            ctxt.step(lamb)
            test_x_t = ctxt.get_x_t()
            test_v_t = ctxt.get_v_t()
            test_du_dx_t = ctxt.get_du_dx_t()

            np.testing.assert_allclose(test_du_dx_t, ref_all_du_dxs[step])
            np.testing.assert_allclose(test_x_t, ref_all_xs[step])


        ref_avg_du_dps = np.mean(ref_all_du_dps, axis=0)
        ref_avg_du_dps_f2 = np.mean(ref_all_du_dps[::2], axis=0)

        np.testing.assert_allclose(lj_obs.avg_du_dp(), ref_avg_du_dps)
        np.testing.assert_allclose(lj_obs_f2.avg_du_dp(), ref_avg_du_dps_f2)

        ref_avg_du_dls = np.mean(ref_all_du_dls, axis=0)
        ref_avg_du_dls_f2 = np.mean(ref_all_du_dls[::2], axis=0)

        np.testing.assert_allclose(lj_obs_du_dl.avg_du_dl(), ref_avg_du_dls)
        np.testing.assert_allclose(lj_obs_f2_du_dl.avg_du_dl(), ref_avg_du_dls_f2)

if __name__ == "__main__":
    unittest.main()