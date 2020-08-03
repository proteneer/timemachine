from jax.config import config; config.update("jax_enable_x64", True)
import jax
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import pymbar
from fe import bar
import unittest

def finite_difference_bar(w, delta):
    fd_pymbar = np.zeros_like(w)
    for i in range(2):
        for j in range(len(w[0])):
            original = pymbar.BAR(w[0],w[1])[0]
            # central difference
            w[i][j] += 0.5*delta
            left_edge = pymbar.BAR(w[0],w[1])[0]
            w[i][j] -= delta
            right_edge = pymbar.BAR(w[0],w[1])[0]
            fd = (left_edge - right_edge)/delta
            fd_pymbar[i][j] = fd
            w[i][j] += 0.5*delta
    return fd_pymbar

def finite_difference_exp(w, delta):
    fd_exp = np.zeros_like(w)
    for i in range(2):
        for j in range(len(w[0])):
            original = bar.EXP(w)
            # central difference
            w[i][j] += 0.5*delta
            left_edge = bar.EXP(w)
            w[i][j] -= delta
            right_edge = bar.EXP(w)
            fd = (left_edge - right_edge)/delta
            fd_exp[i][j] = fd
            w[i][j] += 0.5*delta
    return fd_exp

class TestFreeEnergyDerivatives(unittest.TestCase):

    def test_bar_gradient(self):
        delta = 1e-6
        w = np.random.rand(2,50)
        np.testing.assert_allclose(
                finite_difference_bar(w, delta),
                bar.dG_dw(w),
                rtol=1e-6
        )

    def test_exp_gradient(self):
        delta = 1e-6
        w = np.random.rand(2,50)
        dEXP_dw = jax.grad(bar.EXP,argnums=(0,))
        np.testing.assert_allclose(
                finite_difference_exp(w, delta),
                dEXP_dw(w)[0],
                rtol=1e-6
        )

