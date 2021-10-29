import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

from quadpy import quad
from thermo_deriv.lj_non_periodic import lennard_jones
import numpy as np
import functools
from timemachine.constants import BOLTZ

BOLTZ = RGAS / 1000


class ThermodynamicEngine:
    def __init__(self, U_fn, O_fn, temperature):

        # self.temperature = temperature
        self.U_fn = U_fn
        self.O_fn = O_fn  # (R^1 -> R^N)

        raw_dU_dp_fn = jax.jit(jax.grad(lennard_jones, argnums=(1,)))

        def dU_dp_fn(*args, **kwargs):
            res = raw_dU_dp_fn(*args, **kwargs)[0]
            return res

        def O_dot_dU_dp_fn(*args, **kwargs):
            return O_fn(*args, **kwargs) * dU_dp_fn(*args, **kwargs)

        self.dU_dp_fn = dU_dp_fn
        self.O_dot_dU_dp_fn = O_dot_dU_dp_fn

        self.kT = BOLTZ * temperature

        self.int_lower = 0.005
        self.int_upper = 0.995

        def pdf_fn(particle_coords, rv_fn, lj_params):
            probs = []
            for x in particle_coords:
                xs = np.linspace(0, 1.0, 3, endpoint=True)
                xs[1] = x
                conf = np.expand_dims(xs, axis=1)
                U = lennard_jones(conf, lj_params)
                p = rv_fn(conf, lj_params) * np.exp(-U / self.kT)
                probs.append(p)

            probs = np.asarray(probs)
            probs = np.moveaxis(probs, 0, -1)

            return probs

        self.pdf_fn = pdf_fn

        def quad_fn(x):
            v, e = quad(x, a=self.int_lower, b=self.int_upper)

            assert np.all(e < 1e-6)
            return v

        self.quad_fn = quad_fn

    def Z(self, params):

        integrand = functools.partial(self.pdf_fn, rv_fn=lambda conf, params: 1.0, lj_params=params)
        return self.quad_fn(integrand)

    def O_Z(self, params):

        integrand = functools.partial(self.pdf_fn, rv_fn=self.O_fn, lj_params=params)
        # Z = self.Z(params)
        return self.quad_fn(integrand)

    def dU_dp_Z(self, params):

        integrand = functools.partial(self.pdf_fn, rv_fn=self.dU_dp_fn, lj_params=params)
        # Z = self.Z(params)
        return self.quad_fn(integrand)

    def O_dot_dU_dp_Z(self, params):
        integrand = functools.partial(self.pdf_fn, rv_fn=self.O_dot_dU_dp_fn, lj_params=params)
        # Z = self.Z(params)
        return self.quad_fn(integrand)

    def O_and_dO_dp(self, params):
        Z = self.Z(params)
        avg_O = self.O_Z(params) / Z
        avg_dU_dp = self.dU_dp_Z(params) / Z
        avg_O_dot_dU_dp = self.O_dot_dU_dp_Z(params) / Z

        dO_dp = (avg_O * avg_dU_dp - avg_O_dot_dU_dp) / self.kT

        return avg_O, dO_dp


U_fn = jax.jit(lennard_jones)
O_fn = lambda conf, params: conf[1][0]


te = ThermodynamicEngine(U_fn, O_fn, 300.0)

sigma = [0.1, 0.2, 0.3]
eps = [1.0, 1.2, 1.3]
lj_params = np.stack([sigma, eps], axis=1)


def loss_fn(O_pred):
    O_true = 0.5
    return jnp.abs(O_pred - O_true)


loss_grad_fn = jax.grad(loss_fn)

for epoch in range(10):
    O_pred, dO_dp = te.O_and_dO_dp(lj_params)
    loss = loss_fn(O_pred)
    dL_dO = loss_grad_fn(O_pred)
    dL_dp = dL_dO * dO_dp
    print("epoch", epoch, "params", lj_params, "loss", loss, "O", O_pred)
    lj_params -= 0.1 * dL_dp


# print("Z", te.Z(lj_params))
# print("<O>", te.avg_O(lj_params))
# print("<dU/dp>", te.avg_dU_dp(lj_params))
# print("<O.dU/dp>", te.avg_O_dot_dU_dp(lj_params))
