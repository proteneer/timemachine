from jax import config

config.update("jax_enable_x64", True)

from jax import grad, numpy as np
import numpy as onp

from scipy.stats import norm
from jax.scipy.stats.norm import logpdf as norm_logpdf

from fe.reweighting import ReweightingLayer


def annealed_gaussian_def(lam, params):
    initial_mean, initial_log_sigma = 0.0, 0.0
    target_mean, target_log_sigma = params

    # lam = 0 -> stddev = 1
    # lam = 1 -> stddev = exp(log_sigma)
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
    return np.sum(-((x - mean) / stddev) ** 2)


def normalized_u_fxn(x, lam, params):
    return - logpdf(x, lam, params)


n_windows = 20
lambdas = np.linspace(0, 1, n_windows)
n_samples_per_window = 100

# force field parameters
params_0 = np.array([1.0, -1.0])

onp.random.seed(0)
x_k = [sample(lam, params_0, n_samples_per_window) for lam in lambdas]


def test_self_consistent(verbose=True):
    """Assert that reweighting from params_0 to params_0 gives same estimate as MBAR on params_0"""
    reweighter = ReweightingLayer(x_k, u_fxn, params_0, lambdas)

    # assert that reweighting from params_0 to params_0 is self-consistent
    mbar_delta_f = reweighter.mbar.f_k[-1] - reweighter.mbar.f_k[0]
    reweighted_delta_f = reweighter.compute_delta_f(params_0)
    if verbose:
        print('mbar delta f vs. reweighted delta f on same samples and params', mbar_delta_f, reweighted_delta_f)
    assert np.isclose(mbar_delta_f, reweighted_delta_f)


def test_zeros(verbose=True, sim_atol=1e-1):
    """Assert that
    * free energy differences between sampled, normalized states are approximately zero,
    * reweighting estimate of delta_f for unsampled, yet normalized states is approximately zero
    * gradient of delta_f w.r.t. params is approximately zeros, when varying params cannot influence free energy difference"""

    # assert free energy differences between sampled, normalized states are approximately zero
    reweighter = ReweightingLayer(x_k, normalized_u_fxn, params_0, lambdas)
    if verbose:
        print('mbar f_k on finite samples, where true f_k known to be 0', reweighter.mbar.f_k)
    assert np.isclose(reweighter.mbar.f_k, 0, atol=sim_atol).all()

    # assert that reweighting estimate for delta_f is still close to 0
    params_new = params_0 * 0.5  # should have good overlap...
    reweighted_delta_f = reweighter.compute_delta_f(params_new)
    if verbose:
        print('reweighted delta f on new unsampled states, but known to be 0', reweighted_delta_f)
    assert np.isclose(reweighted_delta_f, 0, atol=sim_atol)

    # assert that we can take the gradient of this free energy estimate, and that it is approximately 0s
    g = grad(reweighter.compute_delta_f)(params_new)
    if verbose:
        print('gradient w.r.t. params', g)
    assert np.isclose(g, 0, atol=sim_atol).all()
