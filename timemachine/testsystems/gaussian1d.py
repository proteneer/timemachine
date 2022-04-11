import jax.numpy as np
from jax.scipy.stats.norm import logpdf as norm_logpdf
from scipy.stats import norm


def make_gaussian_testsystem():
    """normalized/unnormalized 1D Gaussian with a dependence on lambda and params"""

    def annealed_gaussian_def(lam, params):
        initial_mean, initial_log_sigma = 0.0, 0.0
        target_mean, target_log_sigma = params

        # lam = 0 -> (mean = 0, stddev = 1)
        # lam = 1 -> (mean = target_mean, stddev = target_sigma)
        mean = lam * target_mean - (1 - lam) * initial_mean
        stddev = np.exp(lam * target_log_sigma + (1 - lam) * initial_log_sigma)

        return mean, stddev

    def sample(lam, params, n_samples):
        mean, stddev = annealed_gaussian_def(lam, params)
        return norm.rvs(loc=mean, scale=stddev, size=(n_samples, 1))

    def logpdf(x, lam, params):
        mean, stddev = annealed_gaussian_def(lam, params)
        return np.sum(norm_logpdf(x, loc=mean, scale=stddev))

    def u_fxn(x, lam, params):
        """unnormalized version of -logpdf"""
        mean, stddev = annealed_gaussian_def(lam, params)
        return np.sum(0.5 * ((x - mean) / stddev) ** 2)

    def normalized_u_fxn(x, lam, params):
        return -logpdf(x, lam, params)

    def reduced_free_energy(lam, params):
        mean, stddev = annealed_gaussian_def(lam, params)
        log_z = np.log(stddev * np.sqrt(2 * np.pi))
        return -log_z

    return u_fxn, normalized_u_fxn, sample, reduced_free_energy
