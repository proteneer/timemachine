from jax import config

config.update("jax_enable_x64", True)

from jax import grad, numpy as np
import numpy as onp

import cvxpy as cp
from cvxpylayers.jax import CvxpyLayer

from fe.utils import validate_map

from typing import Optional


class DisconnectedEdgesError(Exception):
    pass


def construct_mle_layer(n_nodes: int,
                        rbfe_inds: np.array, rbfe_sigmas: Optional[np.array] = None,
                        abfe_inds: Optional[np.array] = None, abfe_sigmas: Optional[np.array] = None) -> callable:
    """Construct a differentiable function predict_free_energies(simulated_rbfes, simulated_abfes) -> free_energies

    Parameters
    ----------
    n_nodes : int
        number of compounds being compared

    rbfe_inds : int array, shape (n_rbfes, 2)
    rbfe_sigmas : optional float array, shape (n_rbfes,)
        assume input will be an array of n_rbfes simulated_rbfes, with
        simulated_rbfes = [free_energies[b] - free_energies[a] + Normal(0, rbfe_sigmas[i]) for i, (a, b) in enumerate(rbfe_inds)]

        if rbfe_sigmas not provided, will be assumed = 1.0 for all comparisons

    abfe_inds : optional int array, shape (n_absolute,)
    abfe_sigmas : optional float array, shape (n_absolute,)
        assume input will be an array of n_abfes simulated_abfes, with
        simulated_abfes = [free_energies[i] + Normal(0, abfe_sigmas[i]) for i in abfe_inds]

        if abfe_inds not provided, will add one dummy ABFE "measurement" that free_energies[0] = 0
        if abfe_sigmas not provided, will be assumed = 1.0 for all ABFEs

    TODO: refactor (rbfe_inds, rbfe_sigmas) into an "RBFEInfo" object, (abfe_inds, abfe_sigmas) into an "ABFEInfo" object?

    Returns
    -------
    predict_free_energies : callable

        predict_free_energies(simulated_rbfes, Optional[simulated_abfes]) -> free_energies

        maps (an array containing n_rbfes *relative* calculations, and optionally
              an array containing n_absolute *absolute* calculations)
            -> an array containing n_nodes *absolute* estimates

        this is a jax-transformable function, and in particular this means it can be used to define and differentiate
        losses in terms of "cycle-closure-corrected" free energy estimates

    Example
    -------
    >>> n_nodes = 3
    >>> rbfe_inds = np.array([[0,1], [1,2], [2,0]])
    >>> simulated_rbfes = np.array([-1.0, -1.0, +2.0])
    >>> rbfe_sigmas = np.ones(len(rbfe_inds))
    >>> predict_free_energies = construct_mle_layer(n_nodes, rbfe_inds, rbfe_sigmas)
    >>> free_energies = predict_free_energies(simulated_rbfes, np.array([0.0]))
    >>> reconstructed_diffs = onp.array([free_energies[j] - free_energies[i] for (i, j) in rbfe_inds])
    >>> onp.isclose(reconstructed_diffs, simulated_rbfes).all()
    True
    >>> loss = lambda x : np.sum(predict_free_energies(x)**2) # random loss defined in terms of cycle-corrected estimates
    >>> grad(loss)(simulated_rbfes) # can take gradients of this loss w.r.t. simulated_rbfes
    DeviceArray([-2.66666667, -0.66666667,  3.33333333], dtype=float64)

    Notes
    -----
    * Here we used an independent Gaussian noise model for the likelihood of simulated_rbfe(i, j) given
        true underlying value of free_energies[j] - free_energies[i].
        Other noise models could be plugged in here (e.g. ones that allow heavy-tailed noise or correlated errors),
        as long as log_likelihood still permits a cvxpy-friendly expression.
    * Here we used no prior information about plausible values of free_energies.
        Rather than returning a maximum likelihood estimate
            argmax_{trial_free_energies} log_likelihood(trial_free_energies),
        we could just as well compute a maximum a posteriori estimate
            argmax_{trial_free_energies} log_prior(trial_free_energies) + log_likelihood(trial_free_energies),
        where log_prior(trial_free_energies) could be informed by a cheminformatics model or similar.

    References
    ----------
    * DiffNet implementation
        https://github.com/forcefield/DiffNet
    * DiffNet paper
        Huafeng Xu, Optimal measurement network of pairwise differences,
        J. Chem. Inf. Model. 59, 4720-4728, 2019, https://doi.org/10.1021/acs.jcim.9b00528.
    """

    # get RBFE info
    n_rbfes = len(rbfe_inds)
    inds_l, inds_r = rbfe_inds.T
    if (inds_l == inds_r).any():
        raise AssertionError(f'invalid rbfe_inds -- includes self-comparisons: {rbfe_inds[(inds_l == inds_r)]}')

    if rbfe_sigmas is None:
        rbfe_sigmas = np.ones(n_rbfes)

    # get ABFE info
    no_abfe = abfe_inds is None
    if no_abfe:
        abfe_inds = onp.array([0])
    n_absolute = len(abfe_inds)
    if abfe_sigmas is None:
        abfe_sigmas = np.ones(n_absolute)

    # check that the "map" is connected
    valid = validate_map(n_nodes, relative_inds=rbfe_inds, absolute_inds=abfe_inds)
    if not valid:
        raise DisconnectedEdgesError(f'invalid map -- disconnected!')

    # parameters that define the optimization problem: simulated_rbfes and simulated_abfes
    simulated_rbfes = cp.Parameter(n_rbfes)
    simulated_abfes = cp.Parameter(n_absolute)

    # the optimization variable is a collection of trial absolute free energies
    trial_free_energies = cp.Variable(n_nodes)

    # express the expected values of rbfe calculations in terms of differences of underlying trial abfe values
    implied_rbfes = trial_free_energies[inds_r] - trial_free_energies[inds_l]
    implied_abfes = trial_free_energies[abfe_inds]

    # gaussian log likelihood of simulated_rbfes, compared with the relative free energies implied by trial_free_energies
    rbfe_residuals = implied_rbfes - simulated_rbfes
    log_likelihood_rbfes = cp.sum(- (rbfe_residuals / rbfe_sigmas) ** 2 - np.log(rbfe_sigmas * np.sqrt(2 * np.pi)))

    # gaussian log likelihood of simulated_abfes, compared with the absolute free energies implied by trial_free_energies
    abfe_residuals = implied_abfes - simulated_abfes
    log_likelihood_abfes = cp.sum(- (abfe_residuals / abfe_sigmas) ** 2 - np.log(abfe_sigmas * np.sqrt(2 * np.pi)))

    # likelihood of rbfes and abfes jointly
    log_likelihood = log_likelihood_rbfes + log_likelihood_abfes

    # predicted free_energies are obtained by varying trial_free_energies to maximize log_likelihood of
    #   simulated_rbfes and simulated_abfes
    objective = cp.Minimize(- log_likelihood)
    problem = cp.Problem(objective)
    assert problem.is_dpp()

    # return value of the cvxpylayer_fxn callable is a 1-tuple containing a jax array, (free_energies,)
    cvxpylayer_fxn = CvxpyLayer(problem, parameters=problem.parameters(), variables=[trial_free_energies])

    # for convenience, return the jax array rather than a 1-tuple, and make simulated_abfes argument optional
    #   if no ABFE inds were specified
    if no_abfe:
        def predict_free_energies(simulated_rbfes, simulated_abfes=None):
            return cvxpylayer_fxn(simulated_rbfes, np.zeros(1))[0]
    else:
        def predict_free_energies(simulated_rbfes, simulated_abfes):
            return cvxpylayer_fxn(simulated_rbfes, simulated_abfes)[0]

    # return callable function, predict_free_energies(simulated_rbfes, Optional[simulated_abfes]) -> free_energies
    return predict_free_energies
