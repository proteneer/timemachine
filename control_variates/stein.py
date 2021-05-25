from jax import numpy as np
from jax.config import config
from typing import Callable

config.update("jax_enable_x64", True)

Conf = Params = Array = np.array

VectorValued = Callable[[Conf, Params], Array]
ScalarValued = Callable[[Conf, Params], float]
GradLogPi = Callable[[Conf], float]


def cv_from_scalar_langevin_stein_operator(
        test_fxn_grad: VectorValued,
        test_fxn_laplacian: ScalarValued,
        grad_log_pi: GradLogPi,
) -> ScalarValued:
    """Applies the Scalar Langevin-Stein equation to form a function g which has zero mean
    E_{x ~ pi}[g(x, params)] = 0
    for any value of params

    and is thus suitable for use as a control variate since
    E_{x ~ pi}[f(x)] = E_{x ~ pi}[f(x) - g(x, params)]
    for any value of params

    References
    ----------
    * [Si, Oates, Duncan, Carin, Briol, 2020] Scalable Control Variates for Monte Carlo Methods
        via Stochastic Optimization https://arxiv.org/abs/2006.07487
    * [Oates, Papamarkou, Girolami, 2014] The Controlled Thermodynamic Integral for Bayesian Model Comparison
        https://arxiv.org/abs/1404.5053
    * [Mira, Solgi, Imparato, 2010] Zero Variance Markov Chain Monte Carlo for Bayesian Estimators
        https://arxiv.org/abs/1012.2983
    * [Anastasiou et al., 2021] Stein's Method Meets Statistics: A Review of Some Recent Developments
        https://arxiv.org/abs/2105.03481
    """

    def g(x: Array, params: Array) -> float:
        """function derived from arbitrary parameterized test_fxn,
        but now zero-mean and suitable for use as a control variate"""
        return test_fxn_laplacian(x, params) + np.sum(test_fxn_grad(x, params) * grad_log_pi(x))

    return g
