import functools
import unittest

import numpy as np

import jax
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)

from timemachine.lib import custom_ops
from timemachine.lib import ops
from timemachine.potentials import bonded, nonbonded, alchemy

from common import GradientTest
from common import prepare_bonded_system, prepare_nonbonded_system

class TestContext(unittest.TestCase):

    def setup_system(self, N):

        np.random.seed(125)

        P_bonds = 4
        P_angles = 6
        P_torsions = 13
 
        B = 35
        A = 36
        T = 37

        D = 3

        x0 = np.random.rand(N,D).astype(dtype=np.float64)*2

        precision = np.float64

        params, ref_bonded_potentials_0, test_bonded_potentials_0 = prepare_bonded_system(
            x0,
            P_bonds,
            P_angles,
            P_torsions,
            B,
            A,
            T,
            precision
        )

        params, ref_bonded_potentials_1, test_bonded_potentials_1 = prepare_bonded_system(
            x0,
            P_bonds,
            P_angles,
            P_torsions,
            B,
            A,
            T,
            precision,
            params=params
        )


        masses = np.random.rand(N)

        E = 1
        P_charges = 4
        P_lj = 5
        P_exc = 7

        cutoff = 1.5

        params, ref_nonbonded_potentials_0, test_nonbonded_potentials_0 = prepare_nonbonded_system(
            x0,
            E,
            P_charges,
            P_lj,
            P_exc,
            params=params,
            p_scale=10.0,
            cutoff=cutoff,
            precision=precision
        )

        params, ref_nonbonded_potentials_1, test_nonbonded_potentials_1 = prepare_nonbonded_system(
            x0,
            E,
            P_charges,
            P_lj,
            P_exc,
            params=params,
            p_scale=10.0,
            cutoff=cutoff,
            precision=precision
        )

        ref_potentials = [] #((ref_bonded_potentials_0, ref_bonded_potentials_1), (ref_nonbonded_potentials_0, ref_nonbonded_potentials_1))
        test_potentials = [] #((test_bonded_potentials_0, test_bonded_potentials_1), (test_nonbonded_potentials_0, test_nonbonded_potentials_1))

        for a, b in zip(ref_bonded_potentials_0, ref_bonded_potentials_1):
            ref_potentials.append((a,b))
        for a, b in zip(ref_nonbonded_potentials_0, ref_nonbonded_potentials_1):
            ref_potentials.append((a,b))

        for a, b in zip(test_bonded_potentials_0, test_bonded_potentials_1):
            test_potentials.append((a,b))
        for a, b in zip(test_nonbonded_potentials_0, test_nonbonded_potentials_1):
            test_potentials.append((a,b))


        return x0, params, masses, ref_potentials, test_potentials
        # return total_potential, x0, params, masses, test_bonded_potentials + new_nonbonded_potentials


    def test_reverse_mode_lambda(self):
        """
        This test ensures that we can reverse-mode differentiate
        observables that are dU_dlambdas of each state. We provide
        adjoints with respect to each computed dU/dLambda.
        """
        np.random.seed(4321)

        N = 4

        x0, params, masses, ref_tuples, test_tuples = self.setup_system(N)

        ref_alchemical_fns = []
        for ref_a, ref_b in ref_tuples:
            ref_alchemical_fns.append(jax.partial(alchemy.linear_rescale, fn0=ref_a, fn1=ref_b))

        def ref_total_nrg_fn(*args):
            nrgs = []
            for fn in ref_alchemical_fns:
                nrgs.append(fn(*args))
            return jnp.sum(nrgs)

        test_alchemical_fns = []
        for test_a, test_b in test_tuples:

            test_alchemical_fns.append(ops.AlchemicalGradient(
                N,
                len(params),
                test_a,
                test_b
            ))

        v0 = np.random.rand(x0.shape[0], x0.shape[1])
        N = len(masses)

        num_steps = 5
        lambda_schedule = np.random.rand(num_steps)
        cas = np.random.rand(num_steps)
        cbs = np.random.rand(len(masses))/10
        ccs = np.zeros_like(cbs)


        step_sizes = np.random.rand(num_steps)

        dU_dx_fn = jax.grad(ref_total_nrg_fn, argnums=(0,))
        dU_dl_fn = jax.grad(ref_total_nrg_fn, argnums=(2,))

        def loss_fn(du_dls):
            return jnp.sum(du_dls*du_dls)/du_dls.shape[0]

        def integrate_once_through(x_t, v_t, pp):
            all_du_dls = []
            for step in range(num_steps):
                lamb = lambda_schedule[step]
                du_dl = dU_dl_fn(x_t, pp, lamb)[0]
                all_du_dls.append(du_dl)
                dt = step_sizes[step]
                cb_tmp = np.expand_dims(cbs, axis=-1)                
                v_t = cas[step]*v_t + cb_tmp*dU_dx_fn(x_t, pp, lamb)[0]
                x_t = x_t + v_t*dt
                # note that we do not calculate the du_dl of the last frame.

            all_du_dls = jnp.stack(all_du_dls)
            return loss_fn(all_du_dls)

        ref_loss = integrate_once_through(x0, v0, params)

        grad_fn = jax.grad(integrate_once_through, argnums=(2,))
        ref_dl_dp = grad_fn(x0, v0, params)
        stepper = custom_ops.AlchemicalStepper_f64(
            test_alchemical_fns,
            lambda_schedule
        )

        seed = 1234

        ctxt = custom_ops.ReversibleContext_f64(
            stepper,
            x0,
            v0,
            cas,
            cbs,
            ccs,
            step_sizes,
            params,
            seed
        )

        # run 5 steps forward
        ctxt.forward_mode()
        test_du_dls = stepper.get_du_dl()
        test_loss = loss_fn(test_du_dls)
        loss_grad_fn = jax.grad(loss_fn, argnums=(0,))
        dl_du_adjoint = loss_grad_fn(test_du_dls)[0]

        # limit of precision is due to the settings in fixed_point.hpp
        # np.testing.assert_almost_equal(test_loss, ref_loss, decimal=7)
        np.testing.assert_allclose(test_loss, ref_loss, rtol=1e-6)
        stepper.set_du_dl_adjoint(dl_du_adjoint)
        ctxt.set_x_t_adjoint(np.zeros_like(x0))
        ctxt.backward_mode()
        test_dl_dp = ctxt.get_param_adjoint_accum()
        np.testing.assert_allclose(test_dl_dp, ref_dl_dp[0], rtol=1e-6)

    # def test_reverse_mode(self):
    #     """
    #     This tests a straightforward reverse-mode derivative, whose adjoint is only
    #     provided for the last frame.
    #     """
    #     np.random.seed(4321)
    #     ref_total_nrg_fn, x0, params, masses, test_energies = self.setup_charge_system(N=32*3)
    #     v0 = np.random.rand(x0.shape[0], x0.shape[1])
    #     N = len(masses)
    #     num_steps = 1
    #     cas = np.random.rand(num_steps)
    #     cbs = np.random.rand(len(masses))/30
    #     cbs = np.expand_dims(cbs, axis=-1)
    #     step_sizes = np.random.rand(num_steps)/20

    #     ref_grad = jax.grad(ref_total_nrg_fn, argnums=(0,))
    #     ref_hess = jax.hessian(ref_total_nrg_fn, argnums=(0,))
    #     ref_mp = jax.jacfwd(ref_grad, argnums=(1,))

    #     def integrate_once_through(x_t, v_t, pp):
    #         for step in range(num_steps):
    #             dt = step_sizes[step]
    #             v_t = cas[step]*v_t + cbs*ref_grad(x_t, pp)[0]
    #             x_t = x_t + v_t*dt
    #         return x_t

    #     def integrate_all(x_t, v_t, pp):
    #         all_x_t = [x_t]
    #         for step in range(num_steps):
    #             dt = step_sizes[step]
    #             v_t = cas[step]*v_t + cbs*ref_grad(x_t, pp)[0]
    #             x_t = x_t + v_t*dt
    #             all_x_t.append(np.copy(x_t))

    #         return all_x_t

    #     ref_coords_once = integrate_once_through(x0, v0, params)
    #     x_adjoint = np.random.rand(x0.shape[0], x0.shape[1])
    #     x_adjoint.setflags(write=False)

    #     primals, vjp_fn = jax.vjp(integrate_once_through, x0, v0, params)
    #     _, _, ref_vjp_dp = vjp_fn(x_adjoint)
    #     ref_coords_all = integrate_all(x0, v0, params)

    #     x_t_p = x_adjoint.copy()
    #     v_t_p = np.zeros_like(x_t_p)
    #     all_theta_ps = []

    #     # although we have 6 frames (due to 5 steps), we compute the adjoint
    #     # starting from the 5th frame.
    #     reverse_idxs = list(reversed(range(len(step_sizes))))

    #     for idx in reverse_idxs:
    #         x_t = ref_coords_all[idx]
    #         hess_t = ref_hess(x_t, params)[0][0]
    #         mp_t = ref_mp(x_t, params)[0][0]
    #         v_t_p += step_sizes[idx]*x_t_p
    #         x_t_p += np.einsum('ij,ijkl->kl', cbs*v_t_p, hess_t)
    #         theta_p = np.einsum('ij,ijp->p', cbs*v_t_p, mp_t)
    #         all_theta_ps.append(theta_p)
    #         v_t_p = v_t_p*cas[idx]

    #     np.testing.assert_almost_equal(np.sum(all_theta_ps, axis=0), ref_vjp_dp, decimal=12)

    #     stepper = custom_ops.BasicStepper_f64(test_energies)

    #     ctxt = custom_ops.ReversibleContext_f64_3d(
    #         stepper,
    #         x0,
    #         v0,
    #         cas,
    #         cbs,
    #         step_sizes,
    #         params
    #     )

    #     ctxt.forward_mode()
    #     test_coords = ctxt.get_all_coords() # gets the trajectory

    #     # note that the timemachine uses unsigned long longs to deal with double->uint64 casts
    #     # so the precision is at least equal to that of single precision. The relative error should still 
    #     # be on the order of 1e-8, fix for double precision later by doing uint128 atomicAdds
    #     np.testing.assert_almost_equal(ref_coords_all, test_coords, decimal=11)

    #     ctxt.set_x_t_adjoint(x_adjoint) # x_adjoint: [N,3]
    #     ctxt.backward_mode()

    #     # current_p + learning_rate*p_grad
    #     test_vjp_dp = ctxt.get_param_adjoint_accum() # dL/dp, shape = [P]
    #     np.testing.assert_almost_equal(ref_vjp_dp, test_vjp_dp, decimal=8)

    #     test_x_t_adjoint = ctxt.get_x_t_adjoint()
    #     test_v_t_adjoint = ctxt.get_v_t_adjoint()
    #     np.testing.assert_almost_equal(test_x_t_adjoint, x_t_p, decimal=8)
    #     np.testing.assert_almost_equal(test_v_t_adjoint, v_t_p, decimal=8)


if __name__ == "__main__":
    unittest.main()