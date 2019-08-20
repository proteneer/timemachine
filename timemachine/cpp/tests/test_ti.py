import functools
import unittest

import numpy as np
import jax.numpy as jnp

from timemachine.lib import custom_ops
from timemachine.potentials import bonded
from timemachine.potentials import nonbonded

import jax
import jax.ops
from jax.config import config; config.update("jax_enable_x64", True)

# testing thermodynamic integration

class ReferenceLangevin():

    def __init__(self, dt, ca, cb, cc):
        self.dt = dt
        self.coeff_a = ca
        self.coeff_bs = cb
        self.coeff_cs = cc

    def step(self, x_t, v_t, dE_dx):
        noise = np.random.normal(size=(x_t.shape[0], x_t.shape[1]))
        v_t_1 = self.coeff_a*v_t - jnp.expand_dims(self.coeff_bs, axis=-1)*dE_dx + jnp.expand_dims(self.coeff_cs, axis=-1)*noise
        x_t_1 = x_t + v_t_1*self.dt
        final_X = jnp.concatenate([x_t_1[:, :3], x_t[:, 3:]], axis=1)
        final_V = jnp.concatenate([v_t_1[:, :3], v_t[:, 3:]], axis=1)
        return final_X, final_V

class TestTI(unittest.TestCase):

    def test_du_dlamba(self):


        param_idxs = np.array([
            [0, 3],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]], dtype=np.int32)

        scale_matrix = np.array([
            [  0,  0,  1,0.5,  0],
            [  0,  0,  0,  1,  1],
            [  1,  0,  0,  0,0.2],
            [0.5,  1,  0,  0,  1],
            [  0,  1,0.2,  1,  0],
        ], dtype=np.float64)

        params = np.array([3.0, 2.0, 1.0, 1.4], dtype=np.float64)

        energy_fn = functools.partial(nonbonded.lennard_jones,
            scale_matrix=scale_matrix,
            param_idxs=param_idxs,
            box=None,
            cutoff=None)

        def potential_lambda(l, pp):
            # extra_dim = np.zeros(5)
            # extra_dim[3:, 4] = l


            x0 = jnp.array([
                [ 1.0,  0.5, -0.5, 0.0],
                [ 0.2,  0.1, -0.3, 0.0], 
                [ 0.5,  0.4,  0.3, 0.0],  
                [-1.1, -0.5, -0.3, l],
                [ 0.7, -0.2,  2.3, l],
            ], dtype=np.float64)
            return energy_fn(x0, pp)

        lamb = 0.1

        x0 = jnp.array([
            [ 1.0,  0.5, -0.5, 0.0],
            [ 0.2,  0.1, -0.3, 0.0], 
            [ 0.5,  0.4,  0.3, 0.0],  
            [-1.1, -0.5, -0.3, lamb],
            [ 0.7, -0.2,  2.3, lamb],
        ], dtype=np.float64)

        dudl_fn = jax.grad(potential_lambda, argnums=(0,))
        ref_val = dudl_fn(lamb, params)

        # d2u_dldp = jax.jacfwd(dudl_fn, argnums=(1,))

        dudx = jax.grad(energy_fn, argnums=(0,))
        test_grads = dudx(x0, params)[0]

        test_val = np.sum(test_grads[3:, 3:])

        np.testing.assert_almost_equal(ref_val, test_val)

    def test_4d_ti(self):

        # first three atoms are bonded
        # remaining two particle are nonbonded
        masses = np.array([1.0, 12.0, 4.0, 5.0, 5.0])
        x0 = np.array([
            [ 1.0,  0.5, -0.5, 0.0],
            [ 0.2,  0.1, -0.3, 0.0], 
            [ 0.5,  0.4,  0.3, 0.0],  
            [-1.1, -0.5, -0.3, 0.7],
            [ 0.7, -0.2,  2.3, 0.7],
        ], dtype=np.float64)
        x0.setflags(write=False)

        np.random.seed(1337)

        v0 = np.random.rand(x0.shape[0], x0.shape[1])
        v0[:, -1] = 0 # this needs to be set else the velocities will probably a small delta in the 4th dimension

        num_atoms = x0.shape[0]

        #                  bond  bond angle angle   lj   lj   lj   lj
        params = np.array([100.0, 2.0, 75.0, 1.81, 3.0, 2.0, 1.0, 1.4], np.float64)

        # bond_params = np.array([100.0, 2.0], dtype=np.float64)
        bond_idxs = np.array([[0, 1], [1, 2],[3,4]], dtype=np.int32)
        bond_param_idxs = np.array([[0, 1], [0, 1],[0,1]], dtype=np.int32)

        angle_idxs = np.array([[0,1,2]], dtype=np.int32)
        angle_param_idxs = np.array([[2,3]], dtype=np.int32)

        # 1. Reference integration.
        ref_hb = functools.partial(bonded.harmonic_bond,
            bond_idxs=bond_idxs,
            param_idxs=bond_param_idxs,
            box=None
        )

        lj_param_idxs = np.array([
            [0, 3],
            [1, 2],
            [1, 2],
            [1, 2],
            [0, 3]], dtype=np.int32) + 4 # offset

        scale_matrix = np.array([
            [  0,  0,   0, 0.5, 0.5],
            [  0,  0,   0,   1,   1],
            [  0,  0,   0, 0.5, 0.5],
            [0.5,  1, 0.5,   0, 0.5],
            [0.5,  1, 0.5, 0.5,   0]
        ], dtype=np.float64)

        ref_ha = functools.partial(bonded.harmonic_angle,
            angle_idxs=angle_idxs,
            param_idxs=angle_param_idxs,
            box=None
        )

        ref_lj = functools.partial(nonbonded.lennard_jones,
            scale_matrix=scale_matrix,
            param_idxs=lj_param_idxs,
            box=None,
            cutoff=None)

        def total_nrg(conf, params):
            return ref_hb(conf, params) + ref_ha(conf, params) + ref_lj(conf, params)
            # return ref_hb(conf, params)

        test_hb = custom_ops.HarmonicBond_f64(
            bond_idxs,
            bond_param_idxs
        )

        test_ha = custom_ops.HarmonicAngle_f64(
            angle_idxs,
            angle_param_idxs
        )

        test_lj = custom_ops.LennardJones_f64(
            scale_matrix,
            lj_param_idxs
        )

        ref_energies = [ref_hb, ref_ha, ref_lj]
        test_energies = [test_hb, test_ha, test_lj]

        dt = 0.002
        ca = 0.95
        cb = np.random.rand(num_atoms)
        cc = np.zeros(num_atoms, dtype=np.float64)

        lo = custom_ops.LangevinOptimizer_f64(
            dt,
            4,
            ca,
            cb,
            cc
        )

        dp_idxs = np.arange(len(params)).astype(dtype=np.int32)

        ctxt = custom_ops.Context_f64(
            test_energies,
            lo,
            params,
            x0,
            v0,
            dp_idxs
        )

        n_steps = 5

        for i in range(n_steps):
            ctxt.step()

        test_dx_dp = ctxt.get_dx_dp()

        # n_params, n_atoms, n_dims
        assert test_dx_dp.shape == (8, 5, 4)

        intg = ReferenceLangevin(dt, ca, cb, cc)

        ref_dE_dx_fn = jax.grad(total_nrg, argnums=(0,))
        ref_dE_dx_fn = jax.jit(ref_dE_dx_fn)

        ref_hessian = jax.jacfwd(ref_dE_dx_fn, argnums=(0,))
        ref_mixed_partial = jax.jacfwd(ref_dE_dx_fn, argnums=(1,))

        def integrate(x_t, v_t, params):
            for _ in range(n_steps):
                x_t, v_t = intg.step(x_t, v_t, ref_dE_dx_fn(x_t, params)[0])
            return x_t, v_t

        x_f, v_f = integrate(x0, v0, params)

        np.testing.assert_almost_equal(x_f, ctxt.get_x())
        np.testing.assert_almost_equal(v_f, ctxt.get_v())

        grad_fn = jax.jacfwd(integrate, argnums=(2))

        dx_dp_f, dv_dp_f = grad_fn(x0, v0, params)
        dx_dp_f = np.asarray(np.transpose(dx_dp_f, (2,0,1)))
        dv_dp_f = np.asarray(np.transpose(dv_dp_f, (2,0,1)))

        np.testing.assert_almost_equal(dx_dp_f, ctxt.get_dx_dp())
        np.testing.assert_almost_equal(dv_dp_f, ctxt.get_dv_dp())

        # one step behind.
        ctxt.step()

        np.testing.assert_almost_equal(ref_dE_dx_fn(x_f, params)[0], ctxt.get_dE_dx())

        # now we add tests for computing analytic d2u_dldp
        def dudl(x_t, pp):
            return np.sum(ref_dE_dx_fn(x_t, pp)[0][3:, 3:])

        def simulate_ref(l, pp):

            x0 = np.array([
                [ 1.0,  0.5, -0.5, 0.0],
                [ 0.2,  0.1, -0.3, 0.0], 
                [ 0.5,  0.4,  0.3, 0.0],  
                [-1.1, -0.5, -0.3, l],
                [ 0.7, -0.2,  2.3, l],
            ], dtype=np.float64)

            # x_t = x0
            # v_t = v0

            # dudl_fn = jax.grad(potential_lambda, argnums=(0,))

            def integrate(x_t, v_t, ppp):
                for _ in range(n_steps):
                    x_t, v_t = intg.step(x_t, v_t, ref_dE_dx_fn(x_t, ppp)[0])
                return x_t

            x_f = integrate(x0, v0, pp)

            def total_nrg_lambda(c, params):
                return ref_hb(c, params) + ref_ha(c, params) + ref_lj(c, params)

            # dxdp_fn = jax.jacrev(integrate, argnums=(0,))

            return x_f, dudl(x_f, pp)

        ref_x, ref_du_dl = simulate_ref(0.1, params)
        d2u_dldp_fn = jax.jacrev(simulate_ref, argnums=(1,))
        ref_grad = d2u_dldp_fn(0.1, params)

        ref_dx_dp = ref_grad[0][0]
        ref_d2u_dldp = ref_grad[1][0]

        def simulate_test(l, pp):

            x0 = np.array([
                [ 1.0,  0.5, -0.5, 0.0],
                [ 0.2,  0.1, -0.3, 0.0], 
                [ 0.5,  0.4,  0.3, 0.0],  
                [-1.1, -0.5, -0.3, l],
                [ 0.7, -0.2,  2.3, l],
            ], dtype=np.float64)

            ctxt = custom_ops.Context_f64(
                test_energies,
                lo,
                params,
                x0,
                v0,
                dp_idxs
            )

            n_steps = 5

            for i in range(n_steps):
                ctxt.step()

            x_f = ctxt.get_x()

            test_mixed_partials = []

            # we need to compute this separately since the context's sgemm call overwrites
            # the values of d2u_dxdp
            for p, r in zip(test_energies, ref_energies):
                _, _, ph, _, pmp  = p.derivatives(np.expand_dims(x_f, 0), params, dp_idxs)
                test_mixed_partials.append(pmp)

            # (ytz): again, need to integrate once more since we're one step behind
            # this is *super tricky*
            dx_dp = ctxt.get_dx_dp() # P,N,4

            ctxt.step()

            de_dx = ctxt.get_dE_dx() # N,4

            hess = ctxt.get_d2E_dx2()
            ref_hess = ref_hessian(x_f, params)[0][0]

            hess_idxs = jax.ops.index[3:, 3:, :, :3]
            dx_dp_idxs = jax.ops.index[:, :, :3]
            mp_idxs = jax.ops.index[:, 3:, 3:]
            np.testing.assert_almost_equal(hess[hess_idxs], ref_hess[hess_idxs])

            # mixed_part = ctxt.get_d2E_dxdp() 
            mixed_part = np.sum(test_mixed_partials, axis=0)[0]
            ref_mp = np.transpose(ref_mixed_partial(x_f, params)[0][0], [2,0,1])
            np.testing.assert_almost_equal(ref_mp, mixed_part)

            du_dl = np.sum(de_dx[3:, 3:])

            # (ytz): leave these two lines commented, it's the simplest expression of the correct result.
            # lhs = np.einsum('ijkl,mkl->mij',ref_hess, dx_dp)
            # d2u_dldp = np.sum((lhs+rhs)[:,3:,3:], axis=(1,2))

            h_true = hess[hess_idxs]
            dx_dp_true = dx_dp[dx_dp_idxs]
            mp_true = ref_mp[mp_idxs]
            lhs = np.einsum('ijkl,mkl->mij', h_true, dx_dp_true) # correct only up to main hessian
            rhs = mp_true
            # lhs + rhs has shape [P, 2, 1] 
            d2u_dldp = np.sum(lhs+rhs, axis=(1,2)) # P N 4

            return du_dl, dx_dp, d2u_dldp

        test_du_dl, test_dx_dp, test_d2u_dldp = simulate_test(0.1, params)
        np.testing.assert_almost_equal(ref_du_dl, test_du_dl)
        np.testing.assert_almost_equal(np.transpose(ref_dx_dp, [2,0,1]), test_dx_dp) # so this is also correct
        # print(type(ref_d2u_dldp), type(test_d2u_dldp))
        np.testing.assert_almost_equal(np.asarray(ref_d2u_dldp), np.asarray(test_d2u_dldp))

