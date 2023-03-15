import numpy as np


class BarkerProposal:
    def __init__(self, grad_log_q, proposal_sig=0.001):
        """Robust gradient-informed proposal distribution.

        Supports:
        * sampling from proposal y ~ p(. | x)
        * evaluation of proposal density p(y | x)

        Compared to Langevin proposals, more robust to poor initialization and to poor choice of step size.

        References
        ----------
        [Livingstone, Zanella, 2020]
            The Barker proposal: combining robustness and efficiency in gradient-based MCMC
            https://arxiv.org/abs/1908.11812
        """
        self.grad_log_q = grad_log_q
        assert proposal_sig > 0
        self.proposal_sig = proposal_sig

    def _sample(self, x, gaussian_rvs, uniform_rvs):
        """alg. 1"""
        assert x.shape == gaussian_rvs.shape == uniform_rvs.shape

        # sample from base kernel mu_sig
        z = gaussian_rvs * self.proposal_sig

        #  use gradient information to compute probabilities of keeping vs. flipping sign of each component of z
        # p_xz = 1 / (1 + exp(-grad_x * z))
        grad_x = self.grad_log_q(x)
        log_p_xz = -np.logaddexp(0.0, -grad_x * z)
        p_xz = np.exp(log_p_xz)

        b_xz = np.sign(p_xz - uniform_rvs)

        y = x + b_xz * z

        return y

    def sample(self, x):
        """y ~ p(. | x)"""
        gauss = np.random.randn(*x.shape)
        unif = np.random.rand(*x.shape)

        return self._sample(x, gauss, unif)

    def log_density(self, x, y):
        """evaluate log p(y | x) using eq. 16"""

        z = y - x

        base_logpdf_z = np.sum(-0.5 * z ** 2 - np.log(self.proposal_sig * np.sqrt(2 * np.pi)))

        # p_xz = 1 / (1 + exp(-grad_x * z))
        grad_x = -self.grad_log_q(x)
        log_p_xz = -np.logaddexp(0.0, -grad_x * z)

        log_Z = np.log(0.5)  # proposition 3.1

        return base_logpdf_z + np.sum(log_p_xz) - log_Z
