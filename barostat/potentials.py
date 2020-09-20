import jax.numpy as np

def delta_r(ri, rj, box=None):

    diff = ri - rj # this can be either N,N,3 or B,3
    dims = ri.shape[-1]

    if box is not None:
        for d in range(dims):
            diff -= box[d]*np.floor(np.expand_dims(diff[...,d], axis=-1)/box[d][d]+0.5)

    return diff

def distance(ri, rj, box=None):
    dxdydz = np.power(delta_r(ri, rj, box), 2)
    dij = np.sqrt(np.sum(dxdydz, axis=-1))
    return dij

def lennard_jones(conf, lj_params, volume):
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

    param_idxs: shape [num_atoms, 2] np.array
        each tuple (sig, eps) is used as part of the combining rules

    scale_matrix: shape [num_atoms, num_atoms] np.array
        scale mask denoting how we should scale interaction e[i,j].
        The elements should be between [0, 1]. If e[i,j] is 1 then the interaction
        is fully included, 0 implies it is discarded.

    cutoff: float
        Whether or not we apply cutoffs to the system. Any interactions
        greater than cutoff is fully discarded.
    
    """   
    N = conf.shape[0]
    D = conf.shape[-1]
    box_length = np.sqrt(volume)
    # box_length = volume
    box = np.eye(D) * box_length


    sig_ij = lj_params[0]
    eps_ij = lj_params[1]

    ri = np.expand_dims(conf, 0)
    rj = np.expand_dims(conf, 1)

    dij = distance(ri, rj, box)

    N = conf.shape[0]
    keep_mask = np.ones((N,N)) - np.eye(N)

    sig2 = sig_ij/dij
    sig2 *= sig2
    sig6 = sig2*sig2*sig2

    eij = 4*eps_ij*(sig6-1.0)*sig6

    eij = np.where(keep_mask, eij, np.zeros_like(eij))

    return np.sum(eij/2)