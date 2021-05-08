import pymbar

from jax import config

config.update("jax_enable_x64", True)

from jax import vmap, numpy as np
from jax.scipy.special import logsumexp

from typing import List


def normalize(log_weights):
    log_Z = logsumexp(log_weights)
    weights = np.exp(log_weights - log_Z) # sum(weights) == 1
    return weights


def effective_sample_size(log_weights: np.array) -> float:
    """Effective sample size available for reweighting using a collection of log importance weights.

    Notes
    -----
    * This uses the conventional definition ESS(w) = 1 / \sum_i w_i^2, which has some important known limitations!
    * See [Elvira, Martino, Robert, 2018] "Rethinking the effective sample size" https://arxiv.org/abs/1809.04129
        and references therein for some insightful discussion of limitations and possible improvements
    """
    weights = normalize(log_weights)
    return 1 / np.sum(weights ** 2)  # between 1 and len(weights)


class CachedImportanceSamples:
    def __init__(self, xs, log_denominators):
        """Samples from a reference distribution, for use in importance sampling.
        Assume xs[i] ~ e^{log_denominators[i]}."""
        self.xs = xs
        self.log_denominators = log_denominators

    def compute_log_importance_weights(self, logpdf_fxn):
        """logpdf_fxn(xs[i]) - log_denominators[i]"""
        log_numerators = vmap(logpdf_fxn)(self.xs)
        log_importance_weights = log_numerators - self.log_denominators
        return log_importance_weights

    def estimate_expectation(self, logpdf_fxn, observable_fxn):
        """Estimate <observable_fxn(x)>, x ~ logpdf_fxn by importance weighting from reference distribution.

        Example usage
        -------------
        * Estimating <volume>, and gradient w.r.t. logpdf_fxn params
        * Estimating <du/dl>, and gradient w.r.t. both logpdf_fxn params and observable_fxn params
        """
        weights = normalize(self.compute_log_importance_weights(logpdf_fxn))
        observable_values = vmap(observable_fxn)(self.xs)

        return np.sum(weights * observable_values, 0)

    def estimate_free_energy(self, logpdf_fxn):
        """Estimate - log(Z/Z_ref) where Z is the normalizing constant of logpdf_fxn,
        Z_ref is the normalizing constant of reference distribution"""

        log_numerators = vmap(logpdf_fxn)(self.xs)
        log_importance_weights = log_numerators - self.log_denominators

        return - (logsumexp(log_importance_weights) - np.log(len(log_importance_weights)))

    def __repr__(self):
        n_samples, sample_shape = len(self.xs), self.xs[0].shape

        return f"CachedImportanceSamples({n_samples} of shape {sample_shape})"


# options: CachedImportanceSamples using all intermediates vs. just endpoints



class ReweightingLayer:
    def __init__(self, x_k: List[np.array], u_fxn: callable, ref_params: np.array, lambdas: np.array, endpoint_padding=1):
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
            https://github.com/choderalab/pymbar/blob/3c4262c490261110a7595eec37df3e2b8caeab37/pymbar/mbar.py#L1163-L1237
        * Messerly RA, Razavi SM, and Shirts MR. Configuration-Sampling-Based Surrogate Models for Rapid
            Parameterization of Non-Bonded Interactions.
            J. Chem. Theory Comput. 2018, 14, 6, 3144–3162 https://doi.org/10.1021/acs.jctc.8b00223
        * Wieder et al. PyTorch implementation of differentiable reweighting in neutromeratio
            https://github.com/choderalab/neutromeratio/blob/2abf29f03e5175a988503b5d6ceeee8ce5bfd4ad/neutromeratio/parameter_gradients.py#L246-L267
        * Boothroyd et al. Implementations of free energy gradients in OpenFF evaluator, which has used both
            finite-difference reweighting and <dU/dparams>_1 - <dU/dparams>_0
            https://github.com/openforcefield/openff-evaluator/blob/6e6f0a47e1157f8f4b2971b5071c11f2e0291092/docs/releasehistory.rst#030

        Notes
        -----
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
        self.reference_log_weights = logsumexp(self.log_q_k, b=np.array(self.mbar.N_k, dtype=np.float64), axis=1)

        # double-check broadcasts and transposes didn't result in an unexpected shape
        assert self.reference_log_weights.shape == (len(self.xs),)

    def _compute_u_kn(self, xs: np.array) -> np.array:
        u_kn = []
        for lam in self.lambdas:
            u_kn.append(self.vmapped_u_fxn(xs, lam, self.ref_params))
        return np.array(u_kn)

    def _reweighting_inds_from_cutoff(self, endpoint_cutoff=0.5):
        """Get indices of samples self.xs that came from lambda windows within endpoint cutoff of lambda=0 or lambda=1"""
        assert (endpoint_cutoff >= 0.0) and (endpoint_cutoff <= 1.0)
        K = len(self.lambdas)

        endpoint_filter = lambda k: (self.lambdas[k] <= endpoint_cutoff) or (self.lambdas[k] >= (1 - endpoint_cutoff))

        _inds = []
        position = 0
        for k in range(K):
            _inds.append(np.arange(position, position + self.N_k[k]))
            position += self.N_k[k]

        reweighting_inds = np.hstack([_inds[k] for k in range(K) if endpoint_filter(k)])
        assert (len(reweighting_inds) > 0)
        return reweighting_inds

    def compute_delta_f(self, params: np.array, ess_warn_threshold: float = 50.0,
                        endpoint_cutoff: float = 0.5) -> float:
        """Compute an estimate of the free energy difference between lam=0 and lam=1 at a new value of params.

        This function is differentiable w.r.t. params, assuming self.u_fxn(x, lam, params) is differentiable w.r.t.
        params at lam=0.0 and at lam=1.0 on the cached samples in self.xs.

        Prints a warning if the number of effective samples available for reweighting to
        u(self.xs, 0.0, params) or u(self.xs, 1.0, params) is less than ess_warn_threshold.

        Optionally, we may want to compute importance weights using samples from only some of the lambda windows.
        This is controlled by endpoint_cutoff.
        If endpoint_cutoff>=0.5, cached samples from all MBAR intermediates will be used.
        If endpoint_cutoff=0.0, cached samples only from the lambda=0 and lambda=1 endpoints will be used.
        For values of endpoint_cutoff between 0 and 0.5, cached samples from a window lambdas_k will be used if
            lambdas_k <= endpoint_cutoff or lambdas_k >= 1 - endpoint_cutoff
        """
        log_q_0 = lambda x: - self.u_fxn(x, 0.0, params)
        log_q_1 = lambda x: - self.u_fxn(x, 1.0, params)

        reweighting_inds = self._reweighting_inds_from_cutoff(endpoint_cutoff)
        cached_samples = CachedImportanceSamples(self.xs[reweighting_inds],
                                                 self.reference_log_weights[reweighting_inds])

        f_0 = cached_samples.estimate_free_energy(log_q_0)
        f_1 = cached_samples.estimate_free_energy(log_q_1)

        delta_f = f_1 - f_0

        # TODO: compute effective sample size without so much redundant effort
        log_w_0 = cached_samples.compute_log_importance_weights(log_q_0)
        log_w_1 = cached_samples.compute_log_importance_weights(log_q_1)

        ess_0, ess_1 = effective_sample_size(log_w_0), effective_sample_size(log_w_1)
        if min(ess_0, ess_1) < ess_warn_threshold:
            message = f"""
            The number of effective samples is lower than {ess_warn_threshold}! proceed with caution...
                ESS(state 0) = {ess_0:.3f}
                ESS(state 1) = {ess_1:.3f}
            """
            print(UserWarning(message))

        return delta_f
