# Trial function as trial_fxns(x, params) = sum([pair_fxn(distance(x[i], x[j]), params) for (i, j) in pairs])
# Also supply laplacian of trial_fxns

from jax import numpy as np, vmap, hessian
from jax.numpy.linalg import norm
from typing import Callable

Distance = float
Conf = Params = Array = np.array
PairFxn = Callable[[Distance, Params], float]
TrialFxn = TrialFxnLaplacian = Callable[[Conf, Params], float]


def construct_pair_sum_fxn(pair_fxn: PairFxn, pairs: Array) -> TrialFxn:
    """Convert a function that depends on pairwise distances into a function that sums over a selection of pairs

    pair_fxn(r, params) -> float
    pairs = array of pairs of inds

    trial_fxns(x, params) = sum([pair_fxn(r_ij, params) for (i,j) in pairs])
        where r_ij = norm(x[j] - x[i])
    """
    i, j = pairs.T
    f_pair = vmap(pair_fxn, in_axes=(0, None))

    def trial_fxn(x: Conf, params: Params) -> float:
        r = norm(x[j] - x[i], axis=1)
        return np.sum(f_pair(r, params))

    return trial_fxn


def construct_laplacian_of_pair_sum_fxn(pair_fxn: PairFxn, pairs: Array) -> TrialFxnLaplacian:
    """For a trial function constructed as a sum of pair functions

    pair_fxn(r, params) -> float
    pairs = array of pairs of inds

    laplacian(x, params) = sum([naive_laplacian(pair_fxn)(pair, params) for pair in pairs])
        where r_ij = norm(x[j] - x[i])

    TODO: consider using https://github.com/google/jax/issues/3801 instead, if needed for performance
    """
    i, j = pairs.T

    def flat_pair_fxn(flat_x: Array, params: Params) -> float:
        """unflatten flat_x into 2 particles a and b, then return pair_fxn(distance(a,b), params)"""
        d = len(flat_x) // 2
        a, b = flat_x[:d], flat_x[d:]
        r = norm(a - b)
        return pair_fxn(r, params)

    naive_laplacian = lambda pair, params: np.trace(hessian(flat_pair_fxn)(pair, params))
    vec_naive_laplacian = vmap(naive_laplacian, in_axes=(0, None))

    def smart_laplacian(x: Conf, params: Params) -> float:
        """compute sum of traces of many small hessians"""
        x_stacked = np.hstack([x[i], x[j]])
        return np.sum(vec_naive_laplacian(x_stacked, params))

    return smart_laplacian


def adaptive_tanh_basis(r: Distance, params: Params) -> np.array:
    """Compute a tanh basis expansion with variable locations and scales, of the form

    f_i(r) = coeff_i * tanh(exp(scales_i) * (r - offsets_i))

    where params contains a flat concatenation of coefficients, offsets, and log_scales
    """
    n_basis = len(params) // 3
    coefficients, offsets, log_scales = params[:n_basis], params[n_basis:2 * n_basis], params[-n_basis:]

    scales = np.exp(log_scales)
    basis_expansion = np.tanh(scales * (r - offsets))

    return np.dot(coefficients, basis_expansion)
