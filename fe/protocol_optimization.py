from jax import numpy as np
from functools import partial

# representing protocols using fixed basis functions

n_basis = 50


def basis(lam: float) -> np.array:
    return np.tanh(lam * n_basis - np.linspace(-0.1, n_basis * 1.1, num=n_basis))


def basis_dot(lam: float, params: np.array) -> np.array:
    return np.dot(basis(lam), params)


# normalizing protocols to start and end at preset locations

def normalize(f_0, f_1, f_lam):
    """given evaluations of a function f_0 = f(0), f_1 = f(1), f_lam = f(lam)
    return an evaluation of a new function g(lam) = (f(lam) - f(0)) / (f(1) - f(0))

    with the property that g(0) = zeros, g(1) = ones

    note: this might be cleaner with signature normalize_01(f: callable, lam: float),
        but this makes life harder on the JIT compiler...
    """
    return (f_lam - f_0) / (f_1 - f_0)


def parameterized_protocol(lam: float, params: np.array) -> np.array:
    """start at zeros, end at ones"""

    f = partial(basis_dot, params=params)
    return normalize(f(0), f(1), f(lam))
