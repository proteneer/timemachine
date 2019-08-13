import functools
import unittest

import numpy as np
import jax.numpy as jnp

from timemachine.lib import custom_ops
from timemachine.potentials import bonded
from timemachine.potentials import nonbonded

import jax
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

        v0 = np.random.rand(x0.shape[0], x0.shape[1])
        v0[:, -1] = 0 # this needs to be set else the velocities will probably a small delta in the 4th dimension

        num_atoms = x0.shape[0]

        #                  bond  bond angle angle   lj   lj   lj   lj
        params = np.array([100.0, 2.0, 75.0, 1.81, 3.0, 2.0, 1.0, 1.4], np.float64)

        # bond_params = np.array([100.0, 2.0], dtype=np.float64)
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
