import unittest
import numpy as np
import functools

from jax.config import config; config.update("jax_enable_x64", True)
import jax
from jax.test_util import check_grads

from tests.invariances import assert_potential_invariance
from timemachine.potentials import bonded

from jax.experimental import optimizers


class TestOptimizeGeometry(unittest.TestCase):

    def test_minimize_CH2(self):

        opt_init, opt_update, get_params = optimizers.sgd(1e-2)

        x0 = np.array([
            [1.0, 0.2, 3.3], # H 
            [-0.5,-1.1,-0.9], # C
            [3.4, 5.5, 0.2], # H 
        ], dtype=np.float64)

        # ideal bond lengths are 3.0 and 5.5
        b0 = 3.0
        b1 = 5.5
        params = np.array([10.0, b0, b1], dtype=np.float64)

        param_idxs = np.array([
            [0,1],
            [1,2],
        ], dtype=np.int32)

        bond_idxs = np.array([
            [0,1],
            [1,2]
        ], dtype=np.int32)

        energy_fn = functools.partial(
            bonded.harmonic_bond,
            params=params,
            box=None,
            param_idxs=param_idxs,
            bond_idxs=bond_idxs)

        grad_fn = jax.jit(jax.grad(energy_fn, argnums=(0,)))

        opt_state = opt_init(x0)

        # minimize geometries
        for i in range(100):
            g = grad_fn(get_params(opt_state))[0]
            opt_state = opt_update(i, g, opt_state)

        x_final = get_params(opt_state)
        test_b0 = np.linalg.norm(x_final[1] - x_final[0])
        test_b1 = np.linalg.norm(x_final[2] - x_final[1])

        np.testing.assert_almost_equal(b0, test_b0, decimal=2)
        np.testing.assert_almost_equal(b1, test_b1, decimal=2)
