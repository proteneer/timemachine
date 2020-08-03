import unittest
from timemachine.observables import rmsd

from jax.config import config; config.update("jax_enable_x64", True)

from scipy.stats import special_ortho_group

import numpy as onp
import jax

from jax.test_util import check_grads
from jax.experimental import optimizers

class TestRMSD(unittest.TestCase):


    def test_rmsd(self):

        x0 = onp.array([
            [1.0, 0.2, 3.3], # H
            [-0.6,-1.1,-0.9],# C
            [3.4, 5.5, 0.2], # H
            [3.6, 5.6, 0.6], # H
        ], dtype=onp.float64)

        x1 = onp.array([
            [1.0, 0.2, 3.3], # H
            [-0.6,-1.1,-0.9],# C
            [3.4, 5.5, 0.2], # H
            [3.6, 5.6, 0.6], # H
        ], dtype=onp.float64)

        onp.testing.assert_almost_equal(rmsd.opt_rot_rmsd(x0, x1), 0)

        # test random translation
        for _ in range(10):
            offset = onp.random.rand(1, 3)*10
            onp.testing.assert_almost_equal(rmsd.opt_rot_rmsd(x0+offset, x1), 0)
            onp.testing.assert_almost_equal(rmsd.opt_rot_rmsd(x0, x1+offset), 0)

        # generate random rotation matrix
        for _ in range(10):
            rot_x = special_ortho_group.rvs(3)
            rot = onp.dot(rot_x, rot_x.T)
            onp.testing.assert_almost_equal(rmsd.opt_rot_rmsd(onp.dot(x0, rot), x1), 0)
            onp.testing.assert_almost_equal(rmsd.opt_rot_rmsd(x0, onp.dot(x1, rot)), 0)

        assert rmsd.opt_rot_rmsd(x0 + onp.random.rand(x0.shape[0],3)*10, x1) > 1e-1

        check_grads(rmsd.opt_rot_rmsd, (x0 + onp.random.rand(x0.shape[0],3)*10, x1), order=1, eps=1e-5)
        check_grads(rmsd.opt_rot_rmsd, (x0 + onp.random.rand(x0.shape[0],3)*10, x1), order=2, eps=1e-5)

        check_grads(rmsd.opt_rot_rmsd, (x0, x1 + onp.random.rand(x0.shape[0],3)*10), order=1, eps=1e-5)
        check_grads(rmsd.opt_rot_rmsd, (x0, x1 + onp.random.rand(x0.shape[0],3)*10), order=2, eps=1e-5)

    def test_optimize_rotation(self):

        opt_init, opt_update, get_params = optimizers.sgd(5e-2)

        x0 = onp.array([
            [1.0, 0.2, 3.3], # H
            [-0.6,-1.1,-0.9],# C
            [3.4, 5.5, 0.2], # H
            [3.6, 5.6, 0.6], # H
        ], dtype=onp.float64)

        x1 = onp.array([
            [1.0, 0.2, 3.3], # H
            [-0.6,-1.1,-0.9],# C
            [3.4, 5.5, 0.2], # H
            [3.6, 5.6, 0.6], # H
        ], dtype=onp.float64) + onp.random.rand(x0.shape[0],3)*10

        grad_fn = jax.jit(jax.grad(rmsd.opt_rot_rmsd, argnums=(0,)))
        opt_state = opt_init(x0)

        for i in range(1500):
            g = grad_fn(get_params(opt_state), x1)[0]
            opt_state = opt_update(i, g, opt_state)

        x_final = get_params(opt_state)

        assert rmsd.opt_rot_rmsd(x_final, x1) < 0.1
