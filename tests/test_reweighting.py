from jax import config

config.update("jax_enable_x64", True)

import numpy as onp
from jax import grad, jit
from jax import numpy as np
from jax import value_and_grad, vmap
from jax.scipy.stats.norm import logpdf as norm_logpdf
from pymbar import MBAR
from scipy.stats import norm

from timemachine.fe.reweighting import (
    construct_endpoint_reweighting_estimator,
    construct_mixture_reweighting_estimator,
    interpret_as_mixture_potential,
)

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


def reduced_f(lam, params):
    mean, stddev = annealed_gaussian_def(lam, params)
    log_z = np.log(stddev * np.sqrt(2 * np.pi))
    return -log_z


def delta_f(params):
    return reduced_f(1.0, params) - reduced_f(0.0, params)


def assert_estimator_accurate(estimate_delta_f, ref_params, n_random_trials, atol=5e-3):
    def sample_random_params():
        mean = ref_params[0] + onp.random.rand()
        log_sigma = ref_params[1] - onp.random.rand()
        return np.array([mean, log_sigma])

    f_hat, g_hat = value_and_grad(estimate_delta_f)(ref_params)
    f_ref, g_ref = value_and_grad(delta_f)(ref_params)

    onp.testing.assert_allclose(f_hat, f_ref, atol=atol)
    onp.testing.assert_allclose(g_hat, g_ref, atol=atol)

    for _ in range(n_random_trials):
        trial_params = sample_random_params()
        f_hat, g_hat = value_and_grad(estimate_delta_f)(trial_params)
        f_ref, g_ref = value_and_grad(delta_f)(trial_params)

        onp.testing.assert_allclose(f_hat, f_ref, atol=atol)
        onp.testing.assert_allclose(g_hat, g_ref, atol=atol)


def test_endpoint_reweighting_1d():
    """for many random parameter sets, assert that the reweighted estimates of
    delta_f(params) and grad(delta_f)(params) are consistent with ground-truth"""
    onp.random.seed(2022)

    ref_params = np.zeros(2)
    ref_delta_f = delta_f(ref_params)

    n_samples = int(1e6)

    samples_0 = sample(0, ref_params, n_samples)
    samples_1 = sample(1, ref_params, n_samples)

    vec_u = vmap(u_fxn, in_axes=(0, None, None))
    vec_u_0_fxn = lambda xs, params: vec_u(xs, 0, params)
    vec_u_1_fxn = lambda xs, params: vec_u(xs, 1, params)

    estimate_delta_f = construct_endpoint_reweighting_estimator(
        samples_0, samples_1, vec_u_0_fxn, vec_u_1_fxn, ref_params, ref_delta_f
    )

    assert_estimator_accurate(jit(estimate_delta_f), ref_params, n_random_trials=100)


def test_endpoint_reweighting_ahfe():
    pass


def test_mixture_reweighting_1d():
    ref_params = np.ones(2)
    n_windows = 10
    lambdas = np.linspace(0, 1, n_windows)

    n_samples_per_window = int(1e3)
    N_k = [n_samples_per_window] * n_windows
    n_samples_total = sum(N_k)

    trajs = [sample(lam, ref_params, n_samples_per_window) for lam in lambdas]
    xs = np.vstack(trajs).flatten()
    u_kn = onp.zeros((n_windows, n_samples_total))
    vec_u = vmap(u_fxn, in_axes=(0, None, None))

    # TODO [generality] : change signature to be (lam, params) instead of (xs, lam, params)?
    vec_u_0_fxn = lambda xs, params: vec_u(xs, 0.0, params)
    vec_u_1_fxn = lambda xs, params: vec_u(xs, 1.0, params)

    for k in range(n_windows):
        u_kn[k] = vec_u(xs, lambdas[k], ref_params)

    # f_k estimates can come from any source, as long as they're accurate

    # using analytical f_k at ref_params
    f_k_exact = np.array([reduced_f(lam, ref_params) for lam in lambdas])
    u_mix_exact = interpret_as_mixture_potential(u_kn, f_k_exact, N_k)

    # various approximations to f_k at ref_params

    # MBAR
    mbar = MBAR(u_kn, N_k=N_k)
    f_k_mbar = mbar.f_k
    u_mix_mbar = interpret_as_mixture_potential(u_kn, f_k_mbar, N_k)

    # TI
    vec_du_dl = vmap(grad(u_fxn, 1), (0, None, None))
    mean_du_dls = np.array([np.mean(vec_du_dl(traj, lam, ref_params)) for (traj, lam) in zip(trajs, lambdas)])
    f_k_ti = np.array([np.trapz(mean_du_dls[:k], lambdas[:k]) for k in range(n_windows)])
    u_mix_ti = interpret_as_mixture_potential(u_kn, f_k_ti, N_k)

    # TODO [overkill] : BAR
    # ...
    # u_mix_ti = ...

    # TODO [overkill] : SMC
    # ...
    # u_mix_smc = ...

    u_mixes = dict(exact_f_k=u_mix_exact, mbar_f_k=u_mix_mbar, ti_f_k=u_mix_ti)

    for condition in u_mixes:
        u_mix = u_mixes[condition]

        # TODO [sign convention]: change signature to accept u_mix rather than log_weights?
        estimate_delta_f = construct_mixture_reweighting_estimator(xs, -u_mix, vec_u_0_fxn, vec_u_1_fxn)

        assert_estimator_accurate(jit(estimate_delta_f), ref_params, n_random_trials=100)


def test_mixture_reweighting_ahfe():
    pass


def test_zeros(sim_atol=1e-1):
    """Assert that
    * free energy differences between sampled, normalized states are approximately zero,
    * reweighting estimate of delta_f for unsampled, yet normalized states is approximately zero
    * gradient of delta_f w.r.t. params is approximately zeros, when varying params cannot influence free energy difference"""
    pass
