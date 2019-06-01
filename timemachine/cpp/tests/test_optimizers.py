import functools
import unittest

import numpy as np

from timemachine.lib import custom_ops
from timemachine.potentials import bonded

import jax
from jax.config import config; config.update("jax_enable_x64", True)


class ReferenceLangevin():

    def __init__(self, dt, ca, cb, cc):
        self.dt = dt
        self.coeff_a = ca
        self.coeff_bs = cb
        self.coeff_cs = cc

    def step(self, x_t, v_t, dE_dx):
        noise = np.random.rand(x_t.shape[0], x_t.shape[1])
        # (ytz): * operator isn't defined for sparse grads (resulting from tf.gather ops), hence the tf.multiply
        v_t_1 = self.coeff_a*v_t - np.expand_dims(self.coeff_bs, axis=-1)*dE_dx + np.expand_dims(self.coeff_cs, axis=-1)*noise
        x_t_1 = x_t + v_t_1*self.dt
        return x_t_1, v_t_1


class TestOptimizers(unittest.TestCase):

    def test_context(self):

        masses = np.array([1.0, 12.0, 4.0])
        x0 = np.array([
            [1.0, 0.5, -0.5],
            [0.2, 0.1, -0.3],
            [0.5, 0.4, 0.3],
        ], dtype=np.float64)
        x0.setflags(write=False)

        num_atoms = x0.shape[0]

        params = np.array([100.0, 2.0, 75.0, 1.81], np.float64)
        bond_params = np.array([100.0, 2.0], dtype=np.float64)
        bond_idxs = np.array([[0, 1], [1, 2]], dtype=np.int32)
        bond_param_idxs = np.array([[0, 1], [0, 1]], dtype=np.int32)

        angle_idxs = np.array([[0,1,2]], dtype=np.int32)
        angle_param_idxs = np.array([[2,3]], dtype=np.int32)

        # 1. Reference integration.
        ref_hb = functools.partial(bonded.harmonic_bond,
            bond_idxs=bond_idxs,
            param_idxs=bond_param_idxs,
            box=None
        )

        ref_ha = functools.partial(bonded.harmonic_angle,
            angle_idxs=angle_idxs,
            param_idxs=angle_param_idxs,
            box=None
        )

        def total_nrg(conf, params):
            return ref_hb(conf, params) + ref_ha(conf, params)

        ref_dE_dx_fn = jax.grad(total_nrg, argnums=(0,))
        ref_dE_dx_fn = jax.jit(ref_dE_dx_fn)
        # print(ref_dE_dx_fn(x0, params))

        dt = 0.002
        ca = 0.95
        cb = np.random.rand(num_atoms)
        # coeff_cs = np.random.rand(num_atoms)
        cc = np.zeros(num_atoms, dtype=np.float64)

        intg = ReferenceLangevin(dt, ca, cb, cc)

        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        def integrate(x_t, v_t, params):
            for _ in range(100):
                x_t, v_t = intg.step(x_t, v_t, ref_dE_dx_fn(x_t, params)[0])
            return x_t, v_t

        x_f, v_f = integrate(x0, v0, params)
        
        grad_fn = jax.jacfwd(integrate, argnums=(2))

        dx_dp_f, dv_dp_f = grad_fn(x0, v0, params)
        # asarray is so we can index into them
        dx_dp_f = np.asarray(np.transpose(dx_dp_f, (2,0,1)))
        dv_dp_f = np.asarray(np.transpose(dv_dp_f, (2,0,1)))

        # 2. Custom Ops Integration

        lo = custom_ops.LangevinOptimizer_f64(
            dt,
            ca,
            cb,
            cc
        )

        test_hb = custom_ops.HarmonicBond_f64(
            bond_idxs,
            bond_param_idxs
        )

        test_ha = custom_ops.HarmonicAngle_f64(
            angle_idxs,
            angle_param_idxs
        )

        dp_idxs = np.arange(len(params)).astype(dtype=np.int32)

        ctxt = custom_ops.Context_f64(
            [test_hb, test_ha],
            lo,
            params,
            x0,
            v0,
            dp_idxs
        )

        for i in range(100):
            ctxt.step()

        np.testing.assert_almost_equal(x_f, ctxt.get_x())
        np.testing.assert_almost_equal(v_f, ctxt.get_v())

        np.testing.assert_almost_equal(dx_dp_f, ctxt.get_dx_dp())
        np.testing.assert_almost_equal(dv_dp_f, ctxt.get_dv_dp())

        # test partial indices
        dp_idxs = np.random.permutation(np.arange(len(params)))[:np.random.randint(len(params))]

        ctxt = custom_ops.Context_f64(
            [test_hb, test_ha],
            lo,
            params,
            x0,
            v0,
            dp_idxs
        )

        for i in range(100):
            ctxt.step()

        np.testing.assert_almost_equal(x_f, ctxt.get_x())
        np.testing.assert_almost_equal(v_f, ctxt.get_v())

        np.testing.assert_almost_equal(dx_dp_f[dp_idxs], ctxt.get_dx_dp())
        np.testing.assert_almost_equal(dv_dp_f[dp_idxs], ctxt.get_dv_dp())


    def test_langevin_step(self):
        """
        Test that we correctly step through a couple of langevin steps.
        """

        # num_params = 23
        # num_atoms = 68

        num_params = 5
        num_atoms = 4

        coeff_a = 0.95
        coeff_bs = np.random.rand(num_atoms)
        coeff_cs = np.random.rand(num_atoms)

        for _ in range(10):

            dE_dx = np.random.rand(num_atoms, 3)
            d2E_dx2 = np.random.rand(num_atoms*3, num_atoms*3)
            d2E_dx2 = np.tril(d2E_dx2) + np.tril(d2E_dx2, -1).T
            d2E_dx2 = np.reshape(d2E_dx2, (num_atoms, 3, num_atoms, 3))
            d2E_dxdp = np.random.rand(num_params, num_atoms, 3)

            dt = 1e-3

            lo = custom_ops.LangevinOptimizer_f64(
                dt,
                coeff_a,
                coeff_bs,
                coeff_cs
            )

            x_t = np.random.rand(num_atoms, 3)
            v_t = np.random.rand(num_atoms, 3)

            dx_dp_t = np.random.rand(num_params, num_atoms, 3)
            dv_dp_t = np.random.rand(num_params, num_atoms, 3)

            noise = np.random.rand(num_atoms, 3)

            ref_v_t = coeff_a*v_t - np.expand_dims(coeff_bs, axis=-1)*dE_dx + np.expand_dims(coeff_cs, axis=-1)*noise
            ref_x_t = x_t + ref_v_t*dt

            hmp = np.einsum('ijkl,mkl->mij', d2E_dx2, dx_dp_t) + d2E_dxdp
            ref_dv_dp_t = coeff_a*dv_dp_t - np.reshape(coeff_bs, (1, -1, 1))*hmp
            ref_dx_dp_t = dx_dp_t + dt*ref_dv_dp_t

            lo.step(
                dE_dx,
                d2E_dx2,
                d2E_dxdp,
                x_t,
                v_t,
                dx_dp_t,
                dv_dp_t,
                noise
            )

            np.testing.assert_almost_equal(ref_v_t, v_t)
            np.testing.assert_almost_equal(ref_x_t, x_t)

            np.testing.assert_almost_equal(ref_dv_dp_t, dv_dp_t)
            np.testing.assert_almost_equal(ref_dx_dp_t, dx_dp_t)
