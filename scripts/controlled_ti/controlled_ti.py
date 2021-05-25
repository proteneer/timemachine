from jax import grad, jit, vmap, value_and_grad, config, numpy as np
config.update("jax_enable_x64", True)

import numpy as onp
onp.random.seed(0)

from control_variates.stein import cv_from_scalar_langevin_stein_operator
from control_variates.trial_fxns.pair import (
    construct_pair_sum_fxn,
    construct_laplacian_of_pair_sum_fxn,
    adaptive_tanh_basis
)
from uncontrolled_ti import f_vec, cutoff, n, log_pi

from tqdm import tqdm
from scipy.optimize import minimize
from functools import partial

import os

ti_results = onp.load('results/uncontrolled_ti.npz')
lambdas, ti_samples, fs = ti_results['lambdas'], ti_results['ti_samples'], ti_results['fs']

# Define a test function in terms of pairwise distances between alchemical atom and other atoms
n_basis = 50
n_params = n_basis * 3

coefficients_0 = np.zeros(n_basis)
offsets_0 = np.linspace(0, cutoff, n_basis)
log_scales_0 = np.zeros(n_basis)
theta0 = np.hstack([coefficients_0, offsets_0, log_scales_0])
print(f'# parameters: {n_params}')


def pair_fxn(r, params):
    return adaptive_tanh_basis(r, params)


alchemical_ind = 0
normal_inds = np.arange(1, n)
pairs = np.array([[alchemical_ind, i] for i in normal_inds], dtype=int)
assert (pairs.shape == (n - 1, 2))

test_fxn = jit(construct_pair_sum_fxn(pair_fxn, pairs))
test_fxn_laplacian = jit(construct_laplacian_of_pair_sum_fxn(pair_fxn, pairs))


def regularize(params, strength=1e+2):
    """penalize pair_fxn(r, params) having wild second derivatives w.r.t. r"""
    r_grid = np.linspace(0, cutoff)
    second_derivs = vmap(grad(grad(pair_fxn)), (0, None))

    flamboyancy = np.mean(second_derivs(r_grid, params) ** 2)
    return strength * flamboyancy


@jit
def g_vec(samples, params, lam):
    g = cv_from_scalar_langevin_stein_operator(
        test_fxn_grad=grad(test_fxn),
        test_fxn_laplacian=test_fxn_laplacian,
        grad_log_pi=grad(partial(log_pi, lam=lam)),
    )
    return vmap(g, in_axes=(0, None))(samples, params)


def controlled_predict_on_samples(params, samples, lam):
    f_on_samples = f_vec(samples, lam)
    g_on_samples = g_vec(samples, params, lam)
    return f_on_samples - g_on_samples


def variance_on_samples(params, samples, lam):
    f_minus_g_on_samples = controlled_predict_on_samples(params, samples, lam)
    return np.std(f_minus_g_on_samples) ** 2


@jit
def loss(params, samples, lam, penalty):
    """variance(f - g) + regularization"""

    return variance_on_samples(params, samples, lam) + regularize(params, penalty)


def learn_control_variate(train_samples, lam, theta0, penalty=1.0):
    # TODO: use stochastic optimizers?
    # TODO: precompute basis functions?
    def fun(params):
        v, g = value_and_grad(loss)(params, train_samples, lam, penalty)
        return float(v), onp.array(g, dtype=onp.float64)

    options = dict(disp=False, maxiter=50)
    # options = None
    result = minimize(fun, theta0, jac=True, options=options, method='L-BFGS-B')
    return result.x


def _cross_validate(train_samples, lam, theta0, penalties=np.logspace(-1, 2, 25), n_splits=3):
    # TODO: only look at penalty parameters within a couple notches of the previously selected penalty?

    # split train_samples
    inds = onp.arange(len(train_samples))
    splits = []
    for _ in range(n_splits):
        onp.random.shuffle(inds)
        train_inds, valid_inds = onp.array(inds[::2]), onp.array(inds[1::2])
        splits.append((train_inds, valid_inds))

    mean_val_losses = []
    for penalty in tqdm(penalties):
        current_val_losses = []
        for (train_inds, valid_inds) in splits:
            params = learn_control_variate(train_samples[train_inds], lam, theta0, penalty)
            current_val_losses.append(variance_on_samples(params, train_samples[valid_inds], lam))
        mean_val_losses.append(np.mean(np.array(current_val_losses)))

    return penalties[onp.argmin(mean_val_losses)]


if __name__ == '__main__':

    # penalties_grid = np.logspace(0,2,10)

    i_s = np.arange(len(lambdas))[::-1]

    optimized_cv_params = onp.zeros((len(i_s), *theta0.shape))
    for i in tqdm(i_s):
        lam, samples = lambdas[i], ti_samples[i]
        train, validate = samples[::2], samples[1::2]

        # penalty = _cross_validate(train, lam, theta0, penalties_grid)
        penalty = 1.0

        theta_star = learn_control_variate(train, lam, theta0, penalty)
        optimized_cv_params[i] = theta_star

        raw_variance = np.std(f_vec(samples, lam)) ** 2
        optimized_variance_train = variance_on_samples(theta_star, train, lam)
        optimized_variance_validate = variance_on_samples(theta_star, validate, lam)
        message = f"""optimized CV for lambda={lam:.3f}
            variance(f):                            {raw_variance:.10f}
            variance(f - g_optimized) (training):   {optimized_variance_train:.10f}
            variance(f - g_optimized) (validation): {optimized_variance_validate:.10f}
            variance reduction:                     {(raw_variance / optimized_variance_validate):.3f}x

            (penalty parameter: {penalty:.6})
        """
        print(message)

    controlled_ti_vals = []
    for lam, samples, params in zip(lambdas, ti_samples, optimized_cv_params):
        controlled_ti_vals.append(controlled_predict_on_samples(params, samples, lam))
    controlled_ti_vals = np.array(controlled_ti_vals)

    result_path = os.path.join(os.path.dirname(__file__), f'results/controlled_ti.npz')
    onp.savez(result_path, lambdas=lambdas, ti_samples=ti_samples, fs=fs, cv_params=optimized_cv_params,
              controlled_ti_vals=controlled_ti_vals)
