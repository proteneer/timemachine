from jax import config

config.update("jax_enable_x64", True)

from jax import grad, numpy as np
import numpy as onp

import cvxpy as cp
from cvxpylayers.jax import CvxpyLayer


def construct_mle_layer(n_nodes: int, comparison_inds: np.array, sigmas: np.array) -> callable:
    """Construct a differentiable function predict_fs(simulated_rbfes) -> fs

    Parameters
    ----------
    n_nodes : int
        number of compounds being compared

    comparison_inds : int array, shape (n_comparisons, 2)
    sigmas : float array, shape (n_comparisons,)
        assume input will be an array of n_comparisons simulated_rbfes, with
        simulated_rbfes = [fs[b] - fs[a] + Normal(0, sigmas[i]) for i, (a, b) in enumerate(comparison_inds)]

    Returns
    -------
    predict_fs : callable

        predict_fs(simulated_rbfes) -> fs

        maps an array containing n_comparisons *relative* calculations -> an array containing n_nodes *absolute* estimates
            (up to an offset)

        this is a jax-transformable function, and in particular this means it can be used to define and differentiate
        losses in terms of "cycle-closure-corrected" free energy estimates

    Example
    -------
    >>> n_nodes = 3
    >>> comparison_inds = np.array([[0,1], [1,2], [2,0]])
    >>> simulated_rbfes = np.array([-1.0, -1.0, +2.0])
    >>> sigmas = np.ones(len(comparison_inds))
    >>> predict_fs = construct_mle_layer(n_nodes, comparison_inds, sigmas)
    >>> fs = predict_fs(simulated_rbfes)
    >>> reconstructed_diffs = onp.array([fs[j] - fs[i] for (i, j) in comparison_inds])
    >>> onp.isclose(reconstructed_diffs, simulated_rbfes).all()
    True
    >>> loss = lambda x : np.sum(predict_fs(x)**2) # random loss defined in terms of cycle-corrected estimates
    >>> grad(loss)(simulated_rbfes) # can take gradients of this loss w.r.t. simulated_rbfes
    DeviceArray([-0.66666667, -0.66666667,  1.33333333], dtype=float64)

    Notes
    -----
    * Here we used an independent Gaussian noise model for the likelihood of simulated_rbfe(i, j) given
        true underlying value of fs[j] - fs[i].
        Other noise models could be plugged in here (e.g. ones that allow heavy-tailed noise or correlated errors),
        as long as log_likelihood still permits a cvxpy-friendly expression.
    * Here we used no prior information about plausible values of fs.
        Rather than returning a maximum likelihood estimate
            argmax_{trial_fs} log_likelihood(trial_fs),
        we could just as well compute a maximum a posteriori estimate
            argmax_{trial_fs} log_prior(trial_fs) + log_likelihood(trial_fs),
        where log_prior(trial_fs) could be informed by a cheminformatics model or similar.
    * Here we do not make use of absolute free energy estimates, which may be available. To incorporate these in the
        future, we would add a cp.Parameter for simulated_abfes, and modify the log_likelihood definition accordingly.
    * predicted_fs only identifiable up to a constant offset, without further information.

    References
    ----------
    * DiffNet implementation
        https://github.com/forcefield/DiffNet
    * DiffNet paper
        Huafeng Xu, Optimal measurement network of pairwise differences,
        J. Chem. Inf. Model. 59, 4720-4728, 2019, https://doi.org/10.1021/acs.jcim.9b00528.
    """

    n_comparisons = len(comparison_inds)
    inds_l, inds_r = comparison_inds.T
    if (inds_l == inds_r).any():
        raise AssertionError(f'invalid comparison_inds: {comparison_inds[(inds_l == inds_r)]}')

    # parameters that define the optimization problem: simulated_rbfes
    simulated_rbfes = cp.Parameter(n_comparisons)

    # the optimization variable is a collection of trial absolute free energies
    trial_fs = cp.Variable(n_nodes)  # up to an offset

    # express the expected values of rbfe calculations in terms of differences of underlying trial abfe values
    implied_rbfes = trial_fs[inds_r] - trial_fs[inds_l]

    # offset so that trial_fs[0] = 0
    constraints = [trial_fs[0] == 0]

    # gaussian log likelihood of simulated_rbfes, compared with the relative free energies implied by trial_abfes
    residuals = implied_rbfes - simulated_rbfes
    log_likelihood = cp.sum(- (residuals / sigmas) ** 2 - np.log(sigmas * np.sqrt(2 * np.pi)))

    # predicted fs are obtained by varying trial_fs to maximize log_likelihood of simulated_rbfes
    objective = cp.Minimize(- log_likelihood)
    problem = cp.Problem(objective, constraints=constraints)
    assert problem.is_dpp()

    # return value of the cvxpylayer_fxn callable is a 1-tuple containing a jax array, (fs,)
    cvxpylayer_fxn = CvxpyLayer(problem, parameters=problem.parameters(), variables=[trial_fs])

    # for convenience, return the jax array rather than a 1-tuple
    predict_fs = lambda simulated_rbfes: cvxpylayer_fxn(simulated_rbfes)[0]

    # return callable function, predict_fs(simulated_rbfes) -> fs
    return predict_fs

