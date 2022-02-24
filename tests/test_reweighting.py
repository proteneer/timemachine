from jax import config

config.update("jax_enable_x64", True)

import numpy as onp
from jax import jit
from jax import numpy as np
from jax import value_and_grad, vmap
from jax.scipy.stats.norm import logpdf as norm_logpdf
from scipy.stats import norm

from timemachine.fe.reweighting import construct_endpoint_reweighting_estimator

# TODO: should these be moved into a test fixture? or to the testsystems module?


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


def delta_f(params):
    sigma_0 = 1.0
    _, target_log_sigma = params
    sigma_1 = np.exp(target_log_sigma)
    log_z_0 = np.log(sigma_0 * np.sqrt(2 * np.pi))
    log_z_1 = np.log(sigma_1 * np.sqrt(2 * np.pi))
    log_z_ratio = log_z_1 - log_z_0
    return -log_z_ratio


def test_endpoint_reweighting_1d():
    """for many random parameter sets, assert that the reweighted estimates of
    delta_f(params) and grad(delta_f)(params) are consistent with ground-truth"""
    onp.random.seed(2022)

    ref_params = np.zeros(2)
    ref_delta_f = delta_f(ref_params)

    n_samples = int(1e6)
    n_random_trials = 100

    samples_0 = sample(0, ref_params, n_samples)
    samples_1 = sample(1, ref_params, n_samples)

    vec_u = vmap(u_fxn, in_axes=(0, None, None))
    vec_u_0_fxn = lambda xs, params: vec_u(xs, 0, params)
    vec_u_1_fxn = lambda xs, params: vec_u(xs, 1, params)

    estimate_delta_f = jit(
        construct_endpoint_reweighting_estimator(
            samples_0, samples_1, vec_u_0_fxn, vec_u_1_fxn, ref_params, ref_delta_f
        )
    )

    def sample_random_params():
        mean = ref_params[0] + onp.random.rand()
        log_sigma = ref_params[1] - onp.random.rand()
        return np.array([mean, log_sigma])

    f_hat, g_hat = value_and_grad(estimate_delta_f)(ref_params)

    assert f_hat == ref_delta_f, "estimate at ref_params expected to be == ref_delta_f"
    assert g_hat[1] < 0, "derivative w.r.t. log stddev expected to be < 0"

    f_ref, g_ref = value_and_grad(delta_f)(ref_params)
    assert f_hat == f_ref  # check self-consistent

    atol = 5e-3
    onp.testing.assert_allclose(g_hat, g_ref, atol=atol)

    for _ in range(n_random_trials):
        trial_params = sample_random_params()
        f_hat, g_hat = value_and_grad(estimate_delta_f)(trial_params)
        f_ref, g_ref = value_and_grad(delta_f)(trial_params)

        onp.testing.assert_allclose(f_hat, f_ref, atol=atol)
        onp.testing.assert_allclose(g_hat, g_ref, atol=atol)


def test_endpoint_reweighting_ahfe():
    pass


def test_mixture_reweighting_1d():
    pass


def test_mixture_reweighting_ahfe():
    pass


def test_zeros(sim_atol=1e-1):
    """Assert that
    * free energy differences between sampled, normalized states are approximately zero,
    * reweighting estimate of delta_f for unsampled, yet normalized states is approximately zero
    * gradient of delta_f w.r.t. params is approximately zeros, when varying params cannot influence free energy difference"""
    pass
