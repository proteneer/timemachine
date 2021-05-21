import numpy as onp
from jax import (
    grad, value_and_grad, jit, vmap, numpy as np, config,
)
from jax.scipy.special import logsumexp

config.update("jax_enable_x64", True)

from fe.protocol_optimization import parameterized_protocol, n_basis
from collections import namedtuple
from scipy.optimize import minimize

cutoff = 5.0


# force field terms

def lennard_jones(r: float, sigma: float, epsilon: float) -> float:
    """https://en.wikipedia.org/wiki/Lennard-Jones_potential"""
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


def coulomb(r: float, charge_product: float) -> float:
    """https://en.wikipedia.org/wiki/Electric_potential"""
    return charge_product / r


Geometry = namedtuple('Geometry', ['r'])
FFParams = namedtuple('FFParams', ['sigma', 'epsilon', 'charge_product'])
ControlDials = namedtuple('ControlDials', ['lj_offset', 'coulomb_offset'])

default_ff = FFParams(sigma=1.0, eps=1.0, charge_prod=-1.0)


def u_controllable(x: Geometry, ff: FFParams, dials: ControlDials) -> float:
    """Separately controllable distance offsets for LJ and Coulomb"""
    u_lj = lennard_jones(x.r + dials.lj_offset, ff.sigma, ff.epsilon)
    u_coulomb = coulomb(x.r + dials.coulomb_offset, ff.charge_product)

    return u_lj + u_coulomb


# how to expose derivatives w.r.t. "protocol shape"

def rescale_control_dials(normalized_control_params: np.array) -> ControlDials:
    """n -> ControlDials(cutoff * (1-n[0]), cutoff * (1 - n[1]))"""

    rescaled = cutoff * (1 - normalized_control_params)

    return ControlDials(*rescaled)


ProtocolParams = namedtuple('ProtocolParams', ['offset'])


def u_vec(r: float, dial_vec: np.array) -> float:
    x, dials = Geometry(r), rescale_control_dials(dial_vec)
    return u_controllable(x, default_ff, dials)


def u_lam(r: float, lam: float, phi: ProtocolParams) -> float:
    # compute_
    # dial_vec = parameterized_protocol(lam, params)
    # control_params = convert(dial_vec)

    x, dials = Geometry(r), compute_dials(lam, phi)
    return u_controllable(x, default_ff, dials)


def compute_dials(lam: float, phi: ProtocolParams):
    unscaled_control_dials = np.array([parameterized_protocol(lam, params) for params in phi])
    return rescale_control_dials(unscaled_control_dials)


def log_weights(xs, dials):
    return - vmap(u_vec, (0, None))(xs, dials)


def stddev_du_dl_on_samples(xs, lam: float, params: np.array):
    # get weights for samples at (lam, params)
    dials = parameterized_protocol(lam, params)
    log_w = log_weights(xs, dials)
    w = np.exp(log_w - logsumexp(log_w)).flatten()

    # get du_dls for samples at (lam, params)
    vmapped_du_dl = vmap(grad(u_lam, argnums=1), in_axes=(0, None, None))
    du_dls = vmapped_du_dl(xs, lam, params)

    # compute weighted estimate of stddev(du_dl(x)), x ~ p(x | lam, params)
    mean = np.sum(du_dls * w)
    squared_deviations = (du_dls - mean) ** 2
    stddev = np.sqrt(np.sum(w * squared_deviations))

    return stddev


if __name__ == '__main__':
    x_samples = onp.random.rand(1000) * cutoff
    onp.random.seed(0)
    n_control_params = 2
    params = np.ones((n_control_params, n_basis))
    phi = ProtocolParams(*params)

    lambdas = np.linspace(0, 1, 50)


    @jit
    def loss(params):
        stddevs = vmap(stddev_du_dl_on_samples, (None, 0, None))(x_samples, lambdas, params)
        variances = stddevs ** 2

        goal = np.mean(variances)

        # NOTE: penalizing parameters being much different from 1, rather than much different from 0
        #   if we penalize norm(params) directly, then we can get really tiny values of the parameters (like, 1e-12)
        #   but still have a reasonable-looking function

        # TODO: rather than penalizing the parameters themselves, should penalize how "squiggly" the protocol is...

        penalty = np.mean((params.flatten() - 1.0) ** 2)

        return goal + penalty


    initial_protocol = params.flatten()
    unflatten = lambda x: x.reshape(params.shape)


    def L(flat_protocol):
        return loss(unflatten(flat_protocol))


    def fun(flat_protocol):
        v, g = value_and_grad(L)(flat_protocol)
        return float(v), onp.array(g, dtype=onp.float64)


    result = minimize(fun, initial_protocol, jac=True, tol=0.0)
    opt_params = unflatten(result.x)


    def discretize(lambdas, flat_params):
        return vmap(parameterized_protocol, (0, None))(lambdas, unflatten(flat_params))
