from jax import config

config.update("jax_enable_x64", True)

from jax import grad, value_and_grad, numpy as np
import numpy as onp

from scipy.optimize import minimize, check_grad
from fe.cycles import construct_mle_layer

onp.random.seed(2021)


def test_cycle_closure_consistency_triangle():
    """Generate a 3-cycle with random true_free_energies on the nodes,
    and assert we can recover predicted free_energies == true_free_energies"""

    n_nodes = 3
    true_free_energies = onp.random.randn(n_nodes)
    true_free_energies -= true_free_energies[0]

    rbfe_inds = np.array([[0, 1], [1, 2], [2, 0]])
    inds_l, inds_r = rbfe_inds.T
    simulated_rbfes = true_free_energies[inds_r] - true_free_energies[inds_l]

    predict_free_energies = construct_mle_layer(n_nodes, rbfe_inds)

    free_energies = predict_free_energies(simulated_rbfes)

    assert (np.isclose(true_free_energies, free_energies).all())


def test_cycle_closure_consistency_dense(n_nodes=100):
    """Generate a large dense comparison network with random true_free_energies on the nodes, and assert we can recover
    predicted free_energies == true_free_energies"""

    true_free_energies = onp.random.randn(n_nodes)
    true_free_energies -= true_free_energies[0]

    inds_l, inds_r = np.triu_indices(n_nodes, k=1)
    rbfe_inds = np.stack([inds_l, inds_r]).T

    simulated_rbfes = true_free_energies[inds_r] - true_free_energies[inds_l]

    predict_free_energies = construct_mle_layer(n_nodes, rbfe_inds)
    free_energies = predict_free_energies(simulated_rbfes)

    assert (np.isclose(true_free_energies, free_energies).all())


def test_deadlock_triangle(verbose=True):
    """Construct a 3-cycle with a set of edge labels and initial edge predictions that could result in a cancellation of
    gradients in some situations. Assert that the gradient of the cycle-corrected edge loss is non-zero for this set of
    initial predictions, and assert that this loss can be minimized."""

    n_nodes = 3

    rbfe_inds = np.array([[0, 1], [1, 2], [2, 0]])
    inds_l, inds_r = rbfe_inds.T

    true_free_energies = np.array([0, 2, -1], dtype=np.float64)
    true_rbfes = true_free_energies[inds_r] - true_free_energies[inds_l]
    simulated_rbfes = np.array([4, -1, 5], dtype=np.float64)

    predict_free_energies = construct_mle_layer(n_nodes, rbfe_inds)

    def apply_cycle_correction_to_rbfes(simulated_rbfes):
        """estimate mle_free_energies, then return [mle_free_energies[j] - mle_free_energies[i] for (i,j) in rbfe_inds]"""
        free_energies = predict_free_energies(simulated_rbfes)
        cycle_corrected_rbfes = free_energies[inds_r] - free_energies[inds_l]

        return cycle_corrected_rbfes

    def corrected_relative_loss(simulated_rbfes):
        cycle_corrected_rbfes = apply_cycle_correction_to_rbfes(simulated_rbfes)

        return np.sum((cycle_corrected_rbfes - true_rbfes) ** 2)

    assert np.linalg.norm(grad(corrected_relative_loss)(simulated_rbfes)) > 1  # 3.26598621

    def fun(x):
        l, g = value_and_grad(corrected_relative_loss)(x)
        return float(l), onp.array(g, dtype=onp.float64)

    x0 = simulated_rbfes
    if verbose:
        print(f'sum((edge_predictions - edge_labels)^2) before optimization: {corrected_relative_loss(x0):.3f}')
    assert corrected_relative_loss(x0) > 1

    result = minimize(fun, x0=x0, jac=True, tol=0.0)
    if verbose:
        print(f'sum((edge_predictions - edge_labels)^2) after optimization: {corrected_relative_loss(result.x):.20f}')

    assert result.fun < 1e-12


def test_optimization_with_cycle_closure(n_nodes=10, verbose=True):
    """Optimize a collection of dense pairwise simulated_rbfes values so that the cycle-closure-corrected estimates they
    imply will equal some known realizable set of true_free_energies."""

    true_free_energies = onp.random.randn(n_nodes)
    true_free_energies -= true_free_energies[0]

    inds_l, inds_r = np.triu_indices(n_nodes, k=1)
    rbfe_inds = np.stack([inds_l, inds_r]).T
    n_rbfes = len(rbfe_inds)

    predict_free_energies = construct_mle_layer(n_nodes, rbfe_inds)

    def L(simulated_rbfes):
        """sum((free_energies - true_free_energies)^2)"""
        assert (len(simulated_rbfes) == n_rbfes)
        free_energies = predict_free_energies(simulated_rbfes)
        assert (len(free_energies) == n_nodes)

        return np.sum((free_energies - true_free_energies) ** 2)

    def fun(x):
        l, g = value_and_grad(L)(x)
        return float(l), onp.array(g, dtype=onp.float64)

    x0 = onp.random.randn(n_rbfes)
    if verbose:
        print(f'sum((free_energies - true_free_energies)^2) before optimization: {L(x0):.3f}')
    assert L(x0) > 1

    result = minimize(fun, x0=x0, jac=True, tol=0.0)
    if verbose:
        print(f'sum((free_energies - true_free_energies)^2) after optimization: {L(result.x):.20f}')

    assert result.fun < 1e-12


def test_grad_cycle_closure(n_nodes=5, tol=1e-3, verbose=True):
    """Compare grad(loss_fxn)(x) with finite difference,
    where loss_fxn is in terms of absolute error of cycle-closure-corrected estimates derived from x
    and where x is random.

    Also check that grad(loss_fxn)(the_right_answer) = zeros.
    """

    true_free_energies = np.array(onp.random.randn(n_nodes))
    true_free_energies -= true_free_energies[0]

    inds_l, inds_r = np.triu_indices(n_nodes, k=1)
    rbfe_inds = np.stack([inds_l, inds_r]).T
    n_rbfes = len(rbfe_inds)

    predict_free_energies = construct_mle_layer(n_nodes, rbfe_inds)

    def loss_fxn(simulated_rbfes):
        """sum((free_energies - true_free_energies)^2)"""
        assert (len(simulated_rbfes) == n_rbfes)
        free_energies = predict_free_energies(simulated_rbfes)
        free_energies -= free_energies[0]
        assert (len(free_energies) == n_nodes)

        return np.sum((free_energies - true_free_energies) ** 2)

    f = lambda x: float(loss_fxn(x))
    g = lambda x: onp.array(grad(loss_fxn)(x))

    # check on some random rbfe guesses
    for _ in range(5):
        x = onp.random.randn(n_rbfes)
        x /= np.linalg.norm(x)
        c = check_grad(f, g, x)
        if verbose:
            print('sum((grad(cycle_closure_loss)(x) - finite_difference_grad(cycle_closure_loss)(x))^2) = ', c)
        assert c < tol

    # check on exact rbfes
    exact_rbfes = true_free_energies[inds_r] - true_free_energies[inds_l]
    assert np.isclose(f(exact_rbfes), 0)
    assert np.isclose(g(exact_rbfes), 0).all()


def test_absolute(n_nodes=10, n_absolute=5, n_rbfes=20):
    """Given a collection of relative and absolute measurements derived from a set of random ground-truth free_energies,
    assert that we can exactly reconstruct these free_energies, including their absolute offset."""

    # random ground-truth, with true_free_energies[0] not necessarily = 0
    true_free_energies = np.array(onp.random.randn(n_nodes))

    # absolute measurements for subset of nodes
    # TODO: randomize abfe_inds by shuffling?
    abfe_inds = onp.arange(n_nodes)[:n_absolute]
    exact_partial_abfes = true_free_energies[abfe_inds]

    # relative measurements for subset of pairs
    # TODO: randomize inds_l, inds_r, rbfe_inds by shuffling?
    all_inds_l, all_inds_r = np.triu_indices(n_nodes, k=1)
    inds_l, inds_r = all_inds_l[:n_rbfes], all_inds_r[:n_rbfes]
    rbfe_inds = np.stack([inds_l, inds_r]).T[:n_rbfes]
    exact_partial_rbfes = true_free_energies[inds_r] - true_free_energies[inds_l]

    # all rbfes and abfes get equal weight
    predict_free_energies = construct_mle_layer(n_nodes, rbfe_inds=rbfe_inds, abfe_inds=abfe_inds)

    # reconstruct free_energies from these partial views -- should be able to exactly recover true_free_energies
    reconstructed_free_energies = predict_free_energies(exact_partial_rbfes, exact_partial_abfes)

    assert np.isclose(reconstructed_free_energies, true_free_energies).all()
