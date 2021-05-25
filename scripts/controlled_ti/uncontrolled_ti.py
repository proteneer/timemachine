from jax import numpy as np, grad, jit, vmap, value_and_grad, config

config.update("jax_enable_x64", True)

from testsystem import U, md, default_params

import numpy as onp

onp.random.seed(0)

from scipy.optimize import minimize
from functools import partial
from tqdm import tqdm

import os

sig, eps, cutoff = default_params.sigma, default_params.epsilon, default_params.cutoff
n, d = 50, 3

lambdas = np.hstack([np.linspace(0, 0.6, 10, endpoint=False), np.linspace(0.6, 1.0, 50)])


def unflatten(x):
    return x.reshape((n, d))


@jit
def flat_U(x, lam):
    return U(unflatten(x), lam)


def energy_fun(flat_x, lam):
    v, g = value_and_grad(flat_U)(flat_x, lam)
    return float(v), onp.array(g, dtype=onp.float64)


log_pi = jit(lambda x, lam: -U(x, lam))


@jit
def f_vec(samples, lam):
    return vmap(grad(U, argnums=1), (0, None))(samples, lam)


if __name__ == '__main__':
    ti_samples = []
    fs = []

    # generate uncontrolled estimates first
    for lam in tqdm(lambdas):
        print(f'running MD at lam={lam:.3f}...')
        grad_log_pi = grad(partial(log_pi, lam=lam))
        fun = partial(energy_fun, lam=lam)

        x = onp.random.randn(n, d)
        flat_x = x.flatten()

        result = minimize(fun, flat_x, jac=True)
        min_flat_x = result.x
        x_min = unflatten(min_flat_x)

        x_equil = md(x_min, grad_log_pi, 10_000)[-1]
        samples = np.array(md(x_equil, grad_log_pi, 100_000))[::50]
        du_dl = f_vec(samples, lam)

        ti_samples.append(samples)
        fs.append(du_dl)

        print(f'mean(du_dl) = {np.mean(du_dl)}')
        print(f'variance(du_dl) = {np.std(du_dl) ** 2}')

    ti_samples = np.array(ti_samples)
    fs = np.array(fs)

    result_path = os.path.join(os.path.dirname(__file__), 'results/uncontrolled_ti.npz')
    onp.savez(result_path, lambdas=lambdas, ti_samples=ti_samples, fs=fs)
