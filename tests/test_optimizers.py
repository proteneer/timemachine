import unittest
import numpy as onp
import jax.numpy as jnp
import functools

import jax
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import optimizers
from jax.test_util import check_grads

from timemachine.potentials import bonded

class TestOptimizeGeometry(unittest.TestCase):

    def test_minimize_CH2(self):

        opt_init, opt_update, get_params = optimizers.sgd(5e-2)

        x0 = onp.array([
            [1.0, 0.2, 3.3], # H 
            [-0.5,-1.1,-0.9], # C
            [3.4, 5.5, 0.2], # H 
        ], dtype=onp.float64)

        # ideal bond lengths are 3.0 and 5.5
        k = 10.0
        b0 = 3.0
        b1 = 5.5
        initial_params = onp.array([k, b0, b1], dtype=onp.float64)

        param_idxs = onp.array([
            [0,1],
            [1,2],
        ], dtype=onp.int32)

        bond_idxs = onp.array([
            [0,1],
            [1,2]
        ], dtype=onp.int32)

        def minimize_structure(test_params):
            energy_fn = functools.partial(
                bonded.harmonic_bond,
                params=test_params,
                box=None,
                param_idxs=param_idxs,
                bond_idxs=bond_idxs)

            grad_fn = jax.jit(jax.grad(energy_fn, argnums=(0,)))
            opt_state = opt_init(x0)

            # use lax.scan, way faster compilation times.
            def apply_carry(carry, _):
                i, x = carry
                g = grad_fn(get_params(x))[0]
                new_state = opt_update(i, g, x)
                new_carry = (i+1, new_state)
                return new_carry, _

            carry_final, _ = jax.lax.scan(apply_carry, (jnp.array(0), opt_state), jnp.zeros((75, 0)))

            trip, opt_final = carry_final

            assert trip == 75

            x_final = get_params(opt_final)
            test_b0 = jnp.linalg.norm(x_final[1] - x_final[0])
            test_b1 = jnp.linalg.norm(x_final[2] - x_final[1])

            return test_b0, test_b1

        # this is fine
        tb0, tb1 = minimize_structure(initial_params)

        onp.testing.assert_almost_equal(b0, tb0, decimal=2)
        onp.testing.assert_almost_equal(b1, tb1, decimal=2)

        # test meta-optimizer by optimizing through an optimizer
        # ground-truth real bond lengths
        ib0, ib1 = 1.0, 2.0

        def loss(test_params):
            t0, t1 = minimize_structure(test_params)
            loss = (t0 - ib0)**2 + (t1 - ib1)**2
            return loss

        loss_opt_init, loss_opt_update, loss_get_params = optimizers.sgd(5e-2)

        # minimze loss
        # very slow for long loops (> 80s), since JITing is super-linear
        # (ytz): we can speed this up significantly when jax.while_loop is
        # differentiable.
        # loss_grad_fn = jax.jit(jax.grad(loss, argnums=(0,)))
        loss_grad_fn = jax.grad(loss, argnums=(0,))
        loss_opt_state = loss_opt_init(initial_params)

        for epoch in range(1000):
            # print("epoch", epoch, "parameters", loss_get_params(loss_opt_state))
            loss_grad = loss_grad_fn(loss_get_params(loss_opt_state))[0]
            loss_opt_state = loss_opt_update(epoch, loss_grad, loss_opt_state)

        # final parameters
        fb = loss_get_params(loss_opt_state)

        onp.testing.assert_almost_equal(k, fb[0], decimal=2)
        onp.testing.assert_almost_equal(ib0, fb[1], decimal=2)
        onp.testing.assert_almost_equal(ib1, fb[2], decimal=2)
        onp.testing.assert_almost_equal(0.0, loss(fb), decimal=3)
