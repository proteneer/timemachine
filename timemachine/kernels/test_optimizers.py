import functools
import unittest

import numpy as np
import custom_ops

import jax
from jax.config import config; config.update("jax_enable_x64", True)

class ReferenceLangevin():

    def __init__(self, masses, dt=0.0025, friction=1.0, temp=300.0, disable_noise=False):
        self.dt = dt
        self.v_t = None
        self.friction = friction # dissipation speed (how fast we forget)
        self.temperature = temp           # temperature

        self.disable_noise = disable_noise
        self.vscale = np.exp(-self.dt*self.friction)

        if self.friction == 0:
            self.fscale = self.dt
        else:
            self.fscale = (1-self.vscale)/self.friction
        kT = BOLTZ * self.temperature
        self.nscale = np.sqrt(kT*(1-self.vscale*self.vscale)) # noise scale
        self.normal = tf.distributions.Normal(loc=0.0, scale=1.0)
        self.invMasses = (1.0/masses).reshape((-1, 1))
        self.sqrtInvMasses = np.sqrt(self.invMasses)

        self.coeff_a = self.vscale
        self.coeff_bs = self.fscale*self.invMasses
        self.coeff_cs = self.nscale*self.sqrtInvMasses

    def step(self, grads):
        num_atoms = len(self.invMasses)
        num_dims = 3

        if self.v_t is None:
            self.v_t = np.zeros((num_atoms, num_dims))

        noise = self.normal.sample((num_atoms, num_dims))
        noise = tf.cast(noise, dtype=grads.dtype)

        if self.disable_noise:
            noise = tf.zeros(noise.shape, dtype=grads.dtype)

        # (ytz): * operator isn't defined for sparse grads (resulting from tf.gather ops), hence the tf.multiply
        self.v_t = self.vscale*self.v_t - tf.multiply(self.fscale*self.invMasses, grads) + self.nscale*self.sqrtInvMasses*noise
        dx = self.v_t * self.dt
        return dx

class TestOptimizers(unittest.TestCase):

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