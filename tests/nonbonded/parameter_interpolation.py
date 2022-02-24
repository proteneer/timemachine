import numpy as np
import numpy.typing as npt


def gen_params(params_initial: npt.NDArray, rng: np.random.Generator, dcharge=0.01, dlogsig=0.1, dlogeps=0.1):
    """Given an initial set of nonbonded parameters, generate random
    final parameters and return the concatenation of the initial and
    final parameters"""

    num_atoms, _ = params_initial.shape
    charge_init, sig_init, eps_init = params_initial[:].T

    charge_final = charge_init + rng.normal(0, dcharge, size=(num_atoms,))

    # perturb LJ parameters in log space to avoid negative result
    sig_final = np.where(sig_init, np.exp(np.log(sig_init) + rng.normal(0, dlogsig, size=(num_atoms,))), 0)
    eps_final = np.where(eps_init, np.exp(np.log(eps_init) + rng.normal(0, dlogeps, size=(num_atoms,))), 0)

    params_final = np.stack((charge_final, sig_final, eps_final), axis=1)

    return np.concatenate((params_initial, params_final))
