# various utility functions for minimizing structures.
import numpy as np
from scipy.optimize import minimize


def minimize_newton_cg(nrgs, x0, num_params):
    """
    Minimzes a structure using a Newton-CG method. This requires a
    hopefully fully invertible analytic Hessian that will be used
    to minimize geometries.

    Parameters
    ----------
    nrgs: [list of functionals]
        Energy functions used to compute the energy, hessian, and mixed partials.

    x0: np.array
        Structure of the molecule to be minimized.

    num_params: int
        total number of parameters of the model. (ytz): this should be refactored out.

    """
    assert x0.shape[1] == 3

    N = x0.shape[0]

    def hessian(conf):
        conf = conf.reshape((N,3))
        hess = None
        for e in nrgs:
            _, _, test_hessians, _ = e.total_derivative(conf, num_params)
            if hess is None:
                hess = test_hessians
            else:
                hess += test_hessians
        return hess.reshape((N*3, N*3))

    def gradient(conf):
        conf = conf.reshape((N,3))
        grads = np.zeros_like(conf)
        for e in nrgs:
            _, test_grads, _, _ = e.total_derivative(conf, num_params)
            grads += test_grads
        return grads.reshape(-1)

    def energy(conf):
        conf = conf.reshape((N,3))
        grads = np.zeros_like(conf)
        nrg = 0
        for e in nrgs:
            test_nrg, test_grads, _, _ = e.total_derivative(conf, num_params)
            nrg += test_nrg
            grads += test_grads
        return nrg, grads.reshape(-1)

    res = minimize(
        energy,
        x0.reshape(-1),
        # method='Newton-CG',
        method='L-BFGS-B',
        jac=True,
        # hess=hessian,
        # options={'xtol': 1e-8, 'disp': False}
    )

    # print("before and after")
    # print(x0)
    # print(np.array(res.x).reshape((N,3)))

    return res.x.reshape((N,3))
    # print(energy(x0), gradient(x0), hessian(x0).shape)