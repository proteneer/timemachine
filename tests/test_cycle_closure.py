from jax import config

config.update("jax_enable_x64", True)

from jax import value_and_grad, numpy as np
import numpy as onp

from scipy.optimize import minimize
from fe.cycles import construct_mle_layer

onp.random.seed(2021)


def test_cycle_closure_consistency_triangle():
    """Generate a 3-cycle with random true_fs on the nodes, and assert we can recover predicted fs == true_fs"""

    n_nodes = 3
    true_fs = onp.random.randn(n_nodes)
    true_fs -= true_fs[0]

    comparison_inds = np.array([[0, 1], [1, 2], [2, 0]])
    inds_l, inds_r = comparison_inds.T
    simulated_rbfes = true_fs[inds_r] - true_fs[inds_l]

    predict_fs = construct_mle_layer(n_nodes, comparison_inds)

    fs = predict_fs(simulated_rbfes)
    fs -= fs[0]

    assert (np.isclose(true_fs, fs).all())


def test_cycle_closure_consistency_dense(n_nodes=100):
    """Generate a large dense comparison network with random true_fs on the nodes, and assert we can recover
    predicted fs == true_fs"""

    true_fs = onp.random.randn(n_nodes)
    true_fs -= true_fs[0]

    inds_l, inds_r = np.triu_indices(n_nodes, k=1)
    comparison_inds = np.stack([inds_l, inds_r]).T

    simulated_rbfes = true_fs[inds_r] - true_fs[inds_l]

    predict_fs = construct_mle_layer(n_nodes, comparison_inds)
    fs = predict_fs(simulated_rbfes)
    fs -= fs[0]

    assert (np.isclose(true_fs, fs).all())


def test_optimization_with_cycle_closure(n_nodes=10, verbose=True):
    """Optimize a collection of dense pairwise simulated_rbfes values so that the cycle-closure-corrected estimates they
    imply will equal some known realizable set of true_fs."""

    true_fs = onp.random.randn(n_nodes)
    true_fs -= true_fs[0]

    inds_l, inds_r = np.triu_indices(n_nodes, k=1)
    comparison_inds = np.stack([inds_l, inds_r]).T
    n_comparisons = len(comparison_inds)

    predict_fs = construct_mle_layer(n_nodes, comparison_inds)

    def L(simulated_rbfes):
        """sum((fs - true_fs)^2)"""
        assert (len(simulated_rbfes) == n_comparisons)
        fs = predict_fs(simulated_rbfes)
        assert (len(fs) == n_nodes)

        fs -= fs[0]

        return np.sum((fs - true_fs) ** 2)

    def fun(x):
        l, g = value_and_grad(L)(x)
        return float(l), onp.array(g, dtype=onp.float64)

    x0 = onp.random.randn(n_comparisons)
    if verbose:
        print(f'sum((fs - true_fs)^2) before optimization: {L(x0):.3f}')
    assert L(x0) > 1

    result = minimize(fun, x0=x0, jac=True, tol=0.0)
    if verbose:
        print(f'sum((fs - true_fs)^2) after optimization: {L(result.x):.20f}')

    assert result.fun < 1e-16
