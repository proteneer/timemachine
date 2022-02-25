from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Gaussian:
    mu: float
    sigma: float


def construct_optimal_path(a: Gaussian, b: Gaussian) -> Callable[[float], Gaussian]:
    """eqs. 70-71 of http://www.stat.columbia.edu/~gelman/research/published/path2.pdf"""

    if a.mu == b.mu:  # main code path assumes we can divide by (b.mu - a.mu)
        return construct_special_case_optimal_path(a, b)

    R2 = (
        ((a.mu - b.mu) / 2) ** 2
        + (3.0 / 2) * (a.sigma ** 2 + b.sigma ** 2)
        + (9.0 / 4) * ((b.sigma ** 2 - a.sigma ** 2) / (b.mu - a.mu)) ** 2
    )
    R = np.sqrt(R2)
    C = 0.5 * (a.mu + b.mu) + 1.5 * (b.sigma ** 2 - a.sigma ** 2) / (b.mu - a.mu)

    def mu(t):
        if t in [0, 1]:
            return a.mu if t == 0 else b.mu

        return R * np.tanh(phi(0) * (1 - t) + phi(1) * t) + C

    def phi(t):
        return np.arctanh((mu(t) - C) / R)

    def sech(x):
        return 1 / np.cosh(x)

    def sigma(t):
        return R / (np.sqrt(3)) * sech(phi(0) * (1 - t) + phi(1) * t)

    def optimal_path(t: float) -> Gaussian:
        return Gaussian(mu(t), sigma(t))

    return optimal_path


def construct_special_case_optimal_path(a: Gaussian, b: Gaussian) -> Callable[[float], Gaussian]:
    """special case of a.mu == b.mu"""

    assert a.mu == b.mu

    def log_sigma(t):
        """sigma(t) = sigma_b^t sigma_a^{1-t}"""
        return t * np.log(b.sigma) + (1 - t) * np.log(a.sigma)

    def optimal_path(t: float) -> Gaussian:
        return Gaussian(a.mu, np.exp(log_sigma(t)))

    return optimal_path
