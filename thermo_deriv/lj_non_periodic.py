import jax.numpy as np

def distance(ri, rj, box=None):
    dxdydz = np.power(ri - rj, 2)
    dij = np.sqrt(np.sum(dxdydz, axis=-1))
    return dij

def lennard_jones(conf, lj_params):
    """
    Implements a non-periodic LJ612 potential using the Lorentz−Berthelot combining
    rules, where sig_ij = (sig_i + sig_j)/2 and eps_ij = sqrt(eps_i * eps_j).

    Parameters
    ----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_params,] np.array
        unique parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    """

    sig = lj_params[:, 0]
    eps = lj_params[:, 1]

    sig_i = np.expand_dims(sig, 0)
    sig_j = np.expand_dims(sig, 1)
    sig_ij = (sig_i + sig_j)/2
    sig_ij_raw = sig_ij

    eps_i = np.expand_dims(eps, 0)
    eps_j = np.expand_dims(eps, 1)
    eps_ij = np.sqrt(eps_i * eps_j)
    eps_ij_raw = eps_ij

    ri = np.expand_dims(conf, 0)
    rj = np.expand_dims(conf, 1)

    dij = distance(ri, rj)
    N = conf.shape[0]
    keep_mask = np.ones((N,N)) - np.eye(N)

    # (ytz): this avoids a nan in the gradient in both jax and tensorflow
    sig_ij = np.where(keep_mask, sig_ij, np.zeros_like(sig_ij))
    eps_ij = np.where(keep_mask, eps_ij, np.zeros_like(eps_ij))

    sig2 = sig_ij/dij
    sig2 *= sig2
    sig6 = sig2*sig2*sig2

    eij = 4*eps_ij*(sig6-1.0)*sig6

    eij = np.where(keep_mask, eij, np.zeros_like(eij))
    return np.sum(eij/2)



    # N = conf.shape[0]
    # D = conf.shape[-1]

    # sig_ij = lj_params[0]
    # eps_ij = lj_params[1]

    # ri = np.expand_dims(conf, 0)
    # rj = np.expand_dims(conf, 1)

    # dij = distance(ri, rj)

    # N = conf.shape[0]
    # keep_mask = np.ones((N,N)) - np.eye(N)

    # sig2 = sig_ij/dij
    # sig2 *= sig2
    # sig6 = sig2*sig2*sig2

    # eij = 4*eps_ij*(sig6-1.0)*sig6

    # eij = np.where(keep_mask, eij, np.zeros_like(eij))
    # return np.sum(eij/2)