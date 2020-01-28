import unittest
import numpy as np


import functools
import jax
from jax.config import config; config.update("jax_enable_x64", True)
from jax.test_util import check_grads

from timemachine.potentials import implicit
from tests.invariances import assert_potential_invariance

class TestGBSA(unittest.TestCase):

    def test_gbsa(self):
        """ Testing the GBSA OBC model. """
        conf = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        params = np.array([
            .1984, .115, .85, # H
            0.0221, .19, .72  # C
        ])

        param_idxs = np.array([
            [3, 4, 5],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ])

        # gb_nrg = implicit.gbsa(param_idxs)

        # atomic_radii = params[param_idxs[:, 1]]
        # scale_factors = params[param_idxs[:, 2]]
        energy_fn = functools.partial(implicit.gbsa, param_idxs=param_idxs)

        # assert_potential_invariance(energy_fn, conf, params)

        # test 4d

        conf4d = np.zeros((conf.shape[0], 4), dtype=np.float64)
        conf4d[:, :3] = conf

        e0 = energy_fn(conf4d, params, box=None)
        e1 = energy_fn(conf, params, box=None)
        np.testing.assert_almost_equal(e0, e1)
        conf4d[:, -1] = 1000000.0
        e2 = energy_fn(conf4d, params, box=None)
        np.testing.assert_almost_equal(e0, e2)

        conf4d[0, 3] = 0.0
        conf4d[1, 3] = 0.0
        conf4d[2, 3] = 0.0

        combined_nrg = energy_fn(conf4d, params, box=None)
        grad_fn = jax.grad(energy_fn, argnums=(0,))

        conf4d[3, 3] = 1.0
        conf4d[4, 3] = 1.0

        dudl = np.sum(grad_fn(conf4d, params, box=None)[0][3:, -1])

        assert dudl < 1.0 # less than 1 kJ/mol

        # linearly separable 
        conf0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230]
        ])

        param_idxs0 = np.array([
            [3, 4, 5],
            [0, 1, 2],
            [0, 1, 2]
        ])

        # energy_fn_0 = functools.partial(implicit.gbsa, param_idxs=param_idxs)
        e0 = implicit.gbsa(conf0, params, box=None, param_idxs=param_idxs0)

        conf1 = np.array([
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        param_idxs1 = np.array([
            [0, 1, 2],
            [0, 1, 2],
        ])

        e1 = implicit.gbsa(conf1, params, box=None, param_idxs=param_idxs1)

        np.testing.assert_almost_equal(combined_nrg, e0 + e1, decimal=3)


if __name__ == "__main__":
    unittest.main()