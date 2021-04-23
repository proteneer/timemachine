import pymbar

from jax import config

config.update("jax_enable_x64", True)

from jax import vmap, numpy as np
from jax.scipy.special import logsumexp

from typing import List


class ReweightingLayer:
    def __init__(self, x_k: List[np.array], u_fxn: callable, params_0: np.array, lambdas: np.array):
        """Assume samples x_k[k] are drawn from e^{-u_fxn(x, lambdas[k], params_0)}"""
        self.x_k = x_k  # list of arrays of snapshots, of length len(lambdas)
        self.N_k = list(map(len, x_k))
        N, K = sum(self.N_k), len(self.N_k)
        assert len(lambdas) == K

        self.xs = np.vstack(x_k)
        assert len(self.xs) == sum(self.N_k)

        self.u_fxn = u_fxn  # signature u_fxn(x, lam, params) -> energy
        self.params_0 = params_0
        self.lambdas = lambdas

        assert (min(self.N_k) > 0)  # assume samples from each lambda window

        # compute u_fxn(x, lam, params_0) for x in xs, lam in lambdas
        self.vmapped_u_fxn = vmap(self.u_fxn, in_axes=(0, None, None))
        u_kn = self._compute_u_kn(self.xs)

        assert (u_kn.shape == (len(lambdas), sum(self.N_k)))

        # compute free energies among all K lambda windows at fixed params_0
        self.mbar = pymbar.MBAR(u_kn, self.N_k)

        # compute mixture weights for samples collected at params_0
        self.log_q_k = self.mbar.f_k - self.mbar.u_kn.T
        self.log_denominator_n = logsumexp(self.log_q_k, b=self.mbar.N_k, axis=1)

        assert self.log_denominator_n.shape == (len(self.xs),)

    def _compute_u_kn(self, xs: np.array) -> np.array:
        u_kn = []
        for lam in self.lambdas:
            u_kn.append(self.vmapped_u_fxn(xs, lam, self.params_0))
        return np.array(u_kn)

    def compute_delta_f(self, params: np.array) -> float:
        """Compute an estimate of Delta f_{0 \to 1} , differentiable w.r.t. params"""

        u_0 = self.vmapped_u_fxn(self.xs, 0.0, params)
        u_1 = self.vmapped_u_fxn(self.xs, 1.0, params)

        log_q_ln = np.stack([- u_0 - self.log_denominator_n, - u_1 - self.log_denominator_n])
        fs = - logsumexp(log_q_ln, axis=1)
        delta_f = fs[1] - fs[0]

        return delta_f
