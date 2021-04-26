import pymbar

from jax import config

config.update("jax_enable_x64", True)

from jax import vmap, numpy as np
from jax.scipy.special import logsumexp

from typing import List


class ReweightingLayer:
    def __init__(self, x_k: List[np.array], u_fxn: callable, ref_params: np.array, lambdas: np.array):
        """Assumes samples x_k[k] are drawn from e^{-u_fxn(x, lambdas[k], ref_params)}.

        The constructor will
        * aggregate all the samples into a flat structure xs (dropping information about which lambda window each sample
            came from),
        * precompute a matrix "u_kn" containing u_fxn(x, lam, ref_params) for x in xs, lam in lambdas, and
        * pass this matrix to pymbar.MBAR to form self-consistent estimates for the free energies of all lambda windows.

        These free energy estimates in turn imply statistical weights of each sample x in a weighted mixture of the
        sampled lambda windows. These "importance weights" allow us to re-use the samples xs to compute arbitrary
        expectations in thermodynamic states that have sufficient "overlap" with this weighted mixture.

        Crucially, if we know how to differentiate U_fxn w.r.t. params, then
        we also know how to differentiate the importance weights w.r.t. params.

        In particular, to compute a differentiable estimate for the free energy difference between lam=1.0 and lam=0.0,
        using the same potential energy function u_fxn but a possibly new parameter set params, we need to be able to
        compute u_fxn(x, lam=0, params), u_fxn(x, lam=1, params) and their gradients w.r.t. params on all cached xs.

        References
        ----------
        * Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
            J. Chem. Phys. 129:124105, 2008. http://dx.doi.org/10.1063/1.2978177
        * Shirts MR. Reweighting from the mixture distribution as a better way to describe the Multistate Bennett Acceptance Ratio.
            arXiv preprint, 2017. https://arxiv.org/abs/1704.00891
        * pymbar implementation of computePerturbedFreeEnergies , which has an associated uncertainty estimate as well!
            https://github.com/choderalab/pymbar/blob/3c4262c490261110a7595eec37df3e2b8caeab37/pymbar/old_mbar.py#L1106-L1212
        * Messerly RA, Razavi SM, and Shirts MR. Configuration-Sampling-Based Surrogate Models for Rapid
            Parameterization of Non-Bonded Interactions.
            J. Chem. Theory Comput. 2018, 14, 6, 3144–3162 https://doi.org/10.1021/acs.jctc.8b00223
        * PyTorch implementation of differentiable reweighting in neutromeratio
            https://github.com/choderalab/neutromeratio/blob/2abf29f03e5175a988503b5d6ceeee8ce5bfd4ad/neutromeratio/parameter_gradients.py#L246-L267

        TODO: allow constructor to accept a precomputed u_kn matrix, if available
        """

        self.x_k = x_k  # list of arrays of snapshots, of length len(lambdas)
        self.N_k = list(map(len, x_k))
        N, K = sum(self.N_k), len(self.N_k)
        assert len(lambdas) == K

        self.xs = np.vstack(x_k)

        # double-check vstacking didn't result in an unexpected shape
        assert len(self.xs) == sum(self.N_k)

        self.u_fxn = u_fxn  # signature u_fxn(x, lam, params) -> reduced potential energy (unitless float)
        self.ref_params = ref_params
        self.lambdas = lambdas

        # assume samples from each lambda window
        assert (min(self.N_k) > 0)

        # compute u_fxn(x, lam, ref_params) for x in xs, lam in lambdas
        self.vmapped_u_fxn = vmap(self.u_fxn, in_axes=(0, None, None))
        u_kn = self._compute_u_kn(self.xs)

        # double-check vmapped_u_fxn didn't expand dims along the way
        assert (u_kn.shape == (len(lambdas), sum(self.N_k)))

        # compute free energies among all K lambda windows at fixed ref_params
        self.mbar = pymbar.MBAR(u_kn, self.N_k)

        # compute mixture weights for samples collected at ref_params
        self.log_q_k = self.mbar.f_k - self.mbar.u_kn.T
        self.log_denominator_n = logsumexp(self.log_q_k, b=self.mbar.N_k, axis=1)

        # double-check broadcasts and transposes didn't result in an unexpected shape
        assert self.log_denominator_n.shape == (len(self.xs),)

    def _compute_u_kn(self, xs: np.array) -> np.array:
        u_kn = []
        for lam in self.lambdas:
            u_kn.append(self.vmapped_u_fxn(xs, lam, self.ref_params))
        return np.array(u_kn)

    def compute_delta_f(self, params: np.array) -> float:
        """Compute an estimate of Delta f_{0 \to 1} , differentiable w.r.t. params.

        This function is differentiable w.r.t. params, assuming self.u_fxn(x, lam, params) is differentiable w.r.t.
        params at lam=0.0 and at lam=1.0."""

        u_0 = self.vmapped_u_fxn(self.xs, 0.0, params)
        u_1 = self.vmapped_u_fxn(self.xs, 1.0, params)

        log_q_ln = np.stack([- u_0 - self.log_denominator_n, - u_1 - self.log_denominator_n])
        assert log_q_ln.shape == (2, len(self.xs))

        fs = - logsumexp(log_q_ln, axis=1)
        delta_f = fs[1] - fs[0]

        return delta_f
