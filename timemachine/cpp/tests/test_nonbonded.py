import functools
import unittest
import scipy.linalg


import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)
import functools

from common import GradientTest
from common import prepare_nonbonded_system

from timemachine.lib import custom_ops

np.set_printoptions(linewidth=500)

class TestNonbonded(GradientTest):

    def get_water_system(self,
        D,
        P_charges,
        P_lj,
        sort=False):

        x = np.load("water.npy").astype(np.float64)
        # x = x[:2976, :D]
        if sort:
            perm = hilbert_sort(x, D)
            x = x[perm, :]

        N = x.shape[0]

        params = np.random.rand(P_charges).astype(np.float64)
        params = np.zeros_like(params)
        param_idxs = np.random.randint(low=0, high=P_charges, size=(N), dtype=np.int32)

        lj_params = np.random.rand(P_lj)/10 # we want these to be pretty small for numerical stability reasons
        lj_param_idxs = np.random.randint(low=0, high=P_lj, size=(N,2), dtype=np.int32)
        lj_param_idxs = lj_param_idxs + len(params) # offset 

        return x, np.concatenate([params, lj_params]), param_idxs, lj_param_idxs

    def get_ref_mp(self, x, params, param_idxs, cutoff):
        ref_mp = mixed_fn(x, params, param_idxs, cutoff)[0][0]
        ref_mp = np.transpose(ref_mp, (2,0,1))
        return ref_mp

    def test_fast_nonbonded(self):
        np.random.seed(125)
        N = 65
        D = 4
        E = 5
        P_charges = 4
        P_lj = 5
        P_exc = 7
 
        x = self.get_random_coords(N, D)

        for cutoff in [100.0, 0.5, 0.1]:
            params, ref_forces, test_forces = prepare_nonbonded_system(
                x,
                E,
                P_charges,
                P_lj,
                P_exc,
                p_scale=8.0,
                cutoff=cutoff
            )

            for r, t in zip(ref_forces, test_forces):
                self.compare_forces(x, params, r, t)



        # self.compare_system(*args, cutoff=100.0)
        # self.compare_system(*args, cutoff=0.5)
        # self.compare_system(*args, cutoff=0.1)

    # @unittest.skip("debug")
    def test_water_box(self):
        
        np.random.seed(123)
        P_charges = 4
        P_lj = 5
        P_exc = 7

        for dim in [3, 4]:
            x = self.get_water_coords(dim)
            E = x.shape[0] # each water 2 bonds and 1 angle constraint, so we remove them.

            for cutoff in [1000.0, 0.9, 0.5, 0.4, 0.001]:

                params, ref_forces, test_forces = prepare_nonbonded_system(
                    x,
                    E,
                    P_charges,
                    P_lj,
                    P_exc,
                    p_scale=8.0,
                    cutoff=cutoff
                )

                for r, t in zip(ref_forces, test_forces):
                    self.compare_forces(x, params, r, t)

    @unittest.skip("debug")
    def test_lambda(self):

        np.random.seed(4321)

        N = 64
        D = 3
        E = 32
        P_charges = 4
        P_lj = 5
        P_exc = 7

        x = self.get_random_coords(N, D)
        cutoff = 100.0
        params, ref_forces, test_forces = prepare_nonbonded_system(
            x,
            E,
            P_charges,
            P_lj,
            P_exc,
            p_scale=8.0,
            cutoff=cutoff,
            custom_D=4 # override
        )

        potential_fn = ref_forces[0]

        def lambda_to_w(lamb, lamb_flags, exponent):
            """
            """
            lk = jnp.power(lamb, exponent)
            d4 = lk/(1-lk)*np.ones_like(lamb_flags)
            d4 = jnp.power(d4, lamb_flags)
            d4 = jnp.where(lamb_flags != 0, d4, 0)
            return d4

        def potential_lambda(x3, lamb, params, lamb_flags, exponent):
            """
            Convert a 3 dimensional set of points into a 4 dimensional set of points using a lambda schedule.

            Lambda Idxs can be either positive or negative depending on if we want to do decoupling (1) or coupling (-1)
            """
            # generate 4th dimensional coordinates
            d4 = lambda_to_w(lamb, lamb_flags, exponent)
            d4 = jnp.expand_dims(d4, axis=-1)
            x4 = jnp.concatenate((x3, d4), axis=1)
            return potential_fn(x4, params)

        lamb_flags = np.random.randint(low=0, high=3, size=(N,)) - 1

        potential_lambda = functools.partial(
            potential_lambda,
            lamb_flags=lamb_flags,
            exponent=2)

        lambda_v = 0.5
        dw_dl_fn = jax.jacfwd(lambda_to_w, argnums=(0,))
        dw_dl = dw_dl_fn(lambda_v, lamb_flags, 2)[0]

        t4 = jnp.concatenate((np.zeros_like(x), np.expand_dims(dw_dl, axis=-1)), axis=1)
        d4 = np.expand_dims(lambda_to_w(lambda_v, lamb_flags, 2), axis=1)
        d4 = np.concatenate((x, d4), axis=1)
        potential_grad_fn = jax.grad(potential_fn, argnums=(0,))
        dU_dw = potential_grad_fn(d4, params)[0]
        ref_grad = np.dot(dU_dw[:, -1], t4[:, -1])
        res_p, res_t = jax.jvp(potential_fn,  (d4, params), (t4, np.zeros_like(params)))

        np.testing.assert_almost_equal(ref_grad, res_t) # clearly this works fine!

        dU_dl_fn = jax.grad(potential_lambda, argnums=(1,))
        primals = (x, lambda_v, params)
        dU_dl = dU_dl_fn(*primals)

        np.testing.assert_almost_equal(dU_dl, res_t)

        d2U_dldx_fn = jax.jacrev(dU_dl_fn, argnums=(0,))
        ref_d2U_dldx = d2U_dldx_fn(x, lambda_v, params)[0][0]

        _primal, test_d2U_dldx = jax.jvp(potential_grad_fn, (d4, params), (t4, np.zeros_like(params)))
        np.testing.assert_almost_equal(np.asarray(test_d2U_dldx[0][:, :3]), ref_d2U_dldx)

        # reference d2U/dlambda_dtheta
        d2U_dldp_fn = jax.jacrev(dU_dl_fn, argnums=(2,))
        ref_d2U_dldp = d2U_dldp_fn(*primals)

        # d2U_dldt = the jvp the dU/dp with the 4th dimension tangent set to t4_tagent
        potential_dtheta_fn = jax.grad(potential_fn, argnums=(1,))
        _primal, test_d2U_dldp = jax.jvp(potential_dtheta_fn, (d4, params), (t4, np.zeros_like(params)))

        post_t4 = t4.copy()
        pre_t4 = jnp.concatenate((np.random.rand(x.shape[0], x.shape[1]), np.expand_dims(np.zeros_like(dw_dl), axis=-1)), axis=1)
        full_t4 = pre_t4 + post_t4

        _, a = jax.jvp(potential_dtheta_fn, (d4, params), (pre_t4, np.zeros_like(params)))
        _, b = jax.jvp(potential_dtheta_fn, (d4, params), (post_t4, np.zeros_like(params)))
        _, c = jax.jvp(potential_dtheta_fn, (d4, params), (full_t4, np.zeros_like(params)))
        np.testing.assert_almost_equal(np.asarray(a[0]+b[0]), np.asarray(c[0]))
        np.testing.assert_almost_equal(np.asarray(test_d2U_dldp[0]), np.asarray(ref_d2U_dldp[0][0]))

        # custom_energies = [custom_ops.Nonbonded_f64_4d(charge_param_idxs, lj_param_idxs, exc_idxs, exc_charge_idxs, exc_lj_idxs, cutoff)]
        custom_energies = test_forces

        ls = custom_ops.LambdaStepper_f64(
            custom_energies,
            [lambda_v],
            lamb_flags,
            2
        )

        dx = ls.forward_step(x, params)

        np.testing.assert_almost_equal(dx, dU_dw[:, :3])

        x_tangent = np.random.rand(x.shape[0], x.shape[1])

        du_dl_adjoint = np.random.rand()

        ls.set_du_dl_adjoint(np.array([du_dl_adjoint]));
        test_dx_adjoint, test_dp_adjoint = ls.backward_step(x, params, x_tangent)

        dU_dx_fn = jax.grad(potential_lambda, argnums=(0,))
        dU_dl_fn = jax.grad(potential_lambda, argnums=(1,))
        def dudl_and_force(coords, lamb, params):
            return dU_dx_fn(coords, lamb, params)[0], dU_dl_fn(coords, lamb, params)[0]

        df = dudl_and_force(x, lambda_v, params)
        _, vjp_fn = jax.vjp(dudl_and_force, x, lambda_v, params)

        ref_dx_adjoint, _, ref_dp_adjoint = vjp_fn((x_tangent, du_dl_adjoint)) # middle element is dlambda
        res = vjp_fn((x_tangent, du_dl_adjoint))

        np.testing.assert_almost_equal(test_dx_adjoint, ref_dx_adjoint)
        np.testing.assert_almost_equal(test_dp_adjoint, ref_dp_adjoint)

if __name__ == "__main__":
    unittest.main()
