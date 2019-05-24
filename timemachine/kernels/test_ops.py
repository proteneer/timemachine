import unittest

import numpy as np
import custom_ops

# (ytz): TBD test in both modes.
from jax.config import config; config.update("jax_enable_x64", True)
import functools

import jax
from timemachine.potentials import bonded

def batch_mult_jvp(grad_fn, x, p, dxdp):
    dpdp = np.eye(p.shape[0])
    def apply_one(dxdp_i, dpdp_i):
        return jax.jvp(
            grad_fn,
            (x, p),
            (dxdp_i, dpdp_i)
        )
    a, b = jax.vmap(apply_one)(dxdp, dpdp)
    return a[0], b

class CustomOpsTest(unittest.TestCase):

    def assert_derivatives(self, conf, params, ref_nrg, test_nrg):

        # This is a messy unit test that tests for:
        # correctness of the 4 derivatives against the reference implementation
        # sparse scattered indices
        # using an empty null array to denote zero dx_dp
        all_dxdps = [
            np.random.rand(params.shape[0], conf.shape[0], conf.shape[1]),
            np.zeros(shape=(params.shape[0], conf.shape[0], conf.shape[1]))
        ]

        all_dp_idxs = [
            np.array([]),
            np.random.permutation(np.arange(len(params)))[:np.random.randint(len(params))],
            np.arange(len(params))
        ]

        for dx_dp in all_dxdps:
            ref_e, ref_de_dp = batch_mult_jvp(ref_nrg, conf, params, dx_dp)
            grad_fn = jax.grad(ref_nrg, argnums=(0,))
            ref_de_dx, ref_d2e_dx2 = batch_mult_jvp(grad_fn, conf, params, dx_dp)

            for dp_idxs in all_dp_idxs:
                dp_idxs = dp_idxs.astype(np.int32)

                test_e, test_de_dx, test_de_dp, test_d2e_dxdp = test_nrg.derivatives(
                    conf,
                    params,
                    dx_dp=dx_dp,
                    dp_idxs=dp_idxs
                )
                np.testing.assert_almost_equal(test_e, ref_e)
                np.testing.assert_almost_equal(test_de_dp, ref_de_dp[dp_idxs])
                np.testing.assert_almost_equal(test_de_dx, ref_de_dx[0])
                np.testing.assert_almost_equal(test_d2e_dxdp, ref_d2e_dx2[0][dp_idxs])

                if np.prod(dx_dp) == 0:

                    dx_dp = np.empty(shape=(0,))

                    test_e, test_de_dx, test_de_dp, test_d2e_dxdp = test_nrg.derivatives(
                        conf,
                        params,
                        dx_dp=dx_dp,
                        dp_idxs=dp_idxs
                    )
                    np.testing.assert_almost_equal(test_e, ref_e)
                    np.testing.assert_almost_equal(test_de_dp, ref_de_dp[dp_idxs])
                    np.testing.assert_almost_equal(test_de_dx, ref_de_dx[0])
                    np.testing.assert_almost_equal(test_d2e_dxdp, ref_d2e_dx2[0][dp_idxs])


class TestHarmonicBond(CustomOpsTest):

    def test_derivatives(self):

        x0 = np.array([
            [1.0, 0.2, 3.3], # H 
            [-0.5,-1.1,-0.9], # C
            [3.4, 5.5, 0.2], # H 
        ], dtype=np.float64)

        params = np.array([10.0, 3.0, 5.5], dtype=np.float64)
        param_idxs = np.array([
            [0,1],
            [1,2],
        ], dtype=np.int32)

        bond_idxs = np.array([
            [0,1],
            [1,2]
        ], dtype=np.int32)

        hb = custom_ops.HarmonicBond_f64(
            bond_idxs,
            param_idxs
        )

        energy_fn = functools.partial(
            bonded.harmonic_bond,
            box=None,
            param_idxs=param_idxs,
            bond_idxs=bond_idxs
        )

        self.assert_derivatives(
            x0,
            params,
            energy_fn,
            hb
        )

class TestHarmonicAngle(CustomOpsTest):
    
    def test_derivatives(self):

        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203], # C
            [ 1.0573,  -0.2011,   1.2864], # H
            [ 2.3928,   1.2209,  -0.2230], # H
            [-0.6891,   1.6983,   0.0780], # H
            [-0.6312,  -1.6261,  -0.2601], # H
        ], dtype=np.float64)
        num_atoms = x0.shape[0]
        params = np.array([75, 1.91, 0.45], dtype=np.float64)

        angle_idxs = np.array([[1,0,2],[1,0,3],[1,0,4],[2,0,3],[2,0,4],[3,0,4]], dtype=np.int32)
        param_idxs = np.array([[0,1],[0,1],[0,2],[0,1],[0,1],[0,2]], dtype=np.int32)

        # enable cos angles
        energy_fn = functools.partial(
            bonded.harmonic_angle,
            box=None,
            angle_idxs=angle_idxs,
            param_idxs=param_idxs,
            cos_angles=True)

        ha = custom_ops.HarmonicAngle_f64(
            angle_idxs,
            param_idxs
        )

        self.assert_derivatives(
            x0,
            params,
            energy_fn,
            ha
        )

# test_derivatives(np.random.rand(3, 3, 3).astype(np.float64))
# test_derivatives(np.zeros(shape=(3, 3, 3)).astype(np.float64))