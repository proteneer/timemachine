"""
Approach:
* Express u as a differentiable function of x and control_params
* Express control_params as a differentiable function of lam and protocol_params
* Express estimate of stddev(du_dl) @ lam in terms of importance weights of pre-cached samples from some distribution
    that has good overlap with all relevant values of lam.
    (This estimate is differentiable w.r.t. protocol_params if importance weights are
    differentiable w.r.t. control_params.)
* Express estimate of TI protocol quality in terms of stddev(du_dl) @ lam, for lam in linspace(0,1,n_windows)
* Differentiate this estimate w.r.t. protocol_params
"""
import os
import numpy as onp
from jax import grad, value_and_grad, jit, vmap, numpy as np, config
from jax.scipy.special import logsumexp

config.update("jax_enable_x64", True)

from collections import namedtuple
from scipy.optimize import minimize, Bounds

import matplotlib.pyplot as plt

from fe.protocol_optimization import parameterized_protocol, n_basis
from fe.reweighting import CachedImportanceSamples


# TODO: refactor basis expansion


# Express u as a differentiable function of x and control_params
def lennard_jones(r: float, sigma: float, epsilon: float) -> float:
    """https://en.wikipedia.org/wiki/Lennard-Jones_potential"""
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


def coulomb(r: float, charge_product: float) -> float:
    """https://en.wikipedia.org/wiki/Electric_potential"""
    return charge_product / r


Geometry = namedtuple('Geometry', ['r'])
FFParams = namedtuple('FFParams', ['sigma', 'epsilon', 'charge_product'])
ControlDials = namedtuple('ControlDials', ['lj_offset', 'coulomb_offset'])

default_ff = FFParams(sigma=1.0, epsilon=1.0, charge_product=-1.0)
cutoff = 5.0


def u_controllable(x: Geometry, ff: FFParams, dials: ControlDials) -> float:
    """Separately controllable distance offsets for LJ and Coulomb"""
    u_lj = lennard_jones(x.r + dials.lj_offset, ff.sigma, ff.epsilon)
    u_coulomb = coulomb(x.r + dials.coulomb_offset, ff.charge_product)

    return u_lj + u_coulomb


# Express control_params as a differentiable function of lam and protocol_params

def rescale_control_dials(normalized_control_params: np.array) -> ControlDials:
    """n -> ControlDials(cutoff * (1-n[0]), cutoff * (1 - n[1]))"""

    rescaled = cutoff * (1 - normalized_control_params)

    return ControlDials(*rescaled)


def u_dials(r: float, dials: ControlDials):
    return u_controllable(Geometry(r), default_ff, dials)


def u_vec(r: float, dial_vec: np.array) -> float:
    return u_dials(r, rescale_control_dials(dial_vec))


def u_lam(r: float, lam: float, params: np.array) -> float:
    return u_dials(r, compute_dials(lam, params))


def compute_dials(lam: float, params: np.array):
    unscaled_control_dials = parameterized_protocol(lam, params)
    return rescale_control_dials(unscaled_control_dials)


# Express estimate of stddev(du_dl) @ lam in terms of importance weights of pre-cached samples
def stddev_du_dl_on_samples(sample_cache: CachedImportanceSamples, lam: float, params: np.array):
    # get weights for samples at (lam, params)
    dials = compute_dials(lam, params)
    logpdf_fxn = lambda x: - u_dials(x, dials)
    log_w = sample_cache.compute_log_importance_weights(logpdf_fxn)
    w = np.exp(log_w - logsumexp(log_w)).flatten()

    # get du_dls for samples at (lam, params)
    vmapped_du_dl = vmap(grad(u_lam, argnums=1), in_axes=(0, None, None))
    du_dls = vmapped_du_dl(sample_cache.xs, lam, params)

    # compute weighted estimate of stddev(du_dl(x)), x ~ p(x | lam, params)
    mean = np.sum(du_dls * w)
    squared_deviations = (du_dls - mean) ** 2
    stddev = np.sqrt(np.sum(w * squared_deviations))

    return stddev


onp.random.seed(0)
n_samples = 1000
xs = onp.random.rand(n_samples) * cutoff
log_denominators = np.ones(n_samples) * np.log(1 / cutoff)  # uniform
sample_cache = CachedImportanceSamples(xs, log_denominators)

# parameters governing shape of control protocol
n_control_params = 2
params_shape = (n_basis, n_control_params)
params = np.ones(params_shape) + 1e-1 * onp.random.randn(*params_shape)

initial_protocol = params.flatten()
unflatten = lambda x: x.reshape(params.shape)

# evenly spaced lambda
n_windows = 50
lambdas = np.linspace(0, 1, n_windows)


# Express estimate of TI protocol quality in terms of stddev(du_dl) @ lam, for lam in linspace(0,1,n_windows)
@jit
def loss(params):
    stddevs = vmap(stddev_du_dl_on_samples, (None, 0, None))(sample_cache, lambdas, params)
    variances = stddevs ** 2

    goal = np.mean(variances)

    # NOTE: penalizing parameters being much different from 1, rather than much different from 0
    #   if we penalize norm(params) directly, then we can get really tiny values of the parameters (like, 1e-12)
    #   but still have a reasonable-looking function

    # TODO: rather than penalizing the parameters themselves, should penalize how "squiggly" the protocol is...

    penalty = np.mean((params.flatten() - 1.0) ** 2)

    return goal + penalty


def L(flat_protocol):
    return loss(unflatten(flat_protocol))


# Differentiate this estimate w.r.t. protocol_params
def fun(flat_protocol):
    v, g = value_and_grad(L)(flat_protocol)
    return float(v), onp.array(g, dtype=onp.float64)


def discretize(lambdas, flat_params):
    return vmap(parameterized_protocol, (0, None))(lambdas, unflatten(flat_params))


def get_figure_fpath(fname):
    return os.path.join(os.path.dirname(__file__), f'figures/{fname}')


def plot_protocols_and_stddevs(lambdas, initial_protocol, optimized_protocol, sample_cache):
    stddev_du_dl_vec_lambda = vmap(stddev_du_dl_on_samples, (None, 0, None))
    initial_stddevs = stddev_du_dl_vec_lambda(sample_cache, lambdas, unflatten(initial_protocol))
    optimized_stddevs = stddev_du_dl_vec_lambda(sample_cache, lambdas, unflatten(optimized_protocol))

    y_ticks = [0, cutoff]
    y_labels = [0, 'cutoff']
    labels = ['LJ offset', 'Coulomb offset']

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title('initial protocol')
    ys = cutoff * (1 - discretize(lambdas, initial_protocol).T)
    for label, y in zip(labels, ys):
        plt.plot(lambdas, y, label=label)
    plt.legend()
    plt.xlabel('$\lambda$')
    plt.ylabel('control_dial($\lambda$)')
    plt.yticks(y_ticks, y_labels)

    plt.subplot(1, 3, 2)
    plt.title('optimized protocol')
    ys = cutoff * (1 - discretize(lambdas, optimized_protocol).T)
    for label, y in zip(labels, ys):
        plt.plot(lambdas, y, label=label)
    plt.legend()
    plt.xlabel('$\lambda$')
    plt.ylabel('control_dial($\lambda$)')
    plt.yticks(y_ticks, y_labels)

    plt.subplot(1, 3, 3)
    plt.title('variance(du/d$\lambda$)\nbefore and after optimization')
    plt.plot(lambdas, initial_stddevs ** 2, label='initial')
    plt.plot(lambdas, optimized_stddevs ** 2, label='optimized')
    plt.legend()

    plt.xlabel('$\lambda$')
    plt.ylabel('variance(du/d$\lambda$)')

    figure_fpath = get_figure_fpath('distance_decoupling_1d_variance.png')
    print(f'saving figure to {figure_fpath}')
    plt.tight_layout()
    plt.savefig(figure_fpath, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print(f'optimizing! initial loss = {L(initial_protocol):.3f}')
    bounds = Bounds(0, +np.inf)  # keep it positive
    result = minimize(fun, initial_protocol, jac=True, bounds=bounds, method='L-BFGS-B', options=dict(maxiter=200))
    optimized_protocol = result.x  # note: flattened
    print(f'done! final loss = {L(optimized_protocol):.3f}')
    print(f'details:\n{result}')

    plot_protocols_and_stddevs(lambdas, initial_protocol, optimized_protocol, sample_cache)
