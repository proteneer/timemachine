import jax.numpy as np

from timemachine.potentials.jax_utils import distance


def overlap(conf, params, a_idxs, b_idxs):
    """
    Compute the 2 body gaussian overlap volume.
    """
    idxs = np.concatenate([a_idxs, b_idxs])

    conf = conf[[idxs]]

    ci = np.expand_dims(conf, axis=1)
    cj = np.expand_dims(conf, axis=0)

    ai = np.expand_dims(params[:, 0], axis=1)
    aj = np.expand_dims(params[:, 0], axis=0)

    pi = np.expand_dims(params[:, 1], axis=1)
    pj = np.expand_dims(params[:, 1], axis=0)

    dij = distance(ci, cj)

    d2ij = dij*dij
    aij = ai*aj
    kij = np.exp(-aij*d2ij/(ai+aj))

    pij = pi*pj
    vij = pij*kij*np.power(np.pi/(ai+aj), 1.5)

    self_overlap = np.diag(vij)

    A = len(a_idxs)
    B = len(b_idxs)

    self_overlap_A = self_overlap[:A]
    self_overlap_B = self_overlap[A:]

    cross_overlap = vij[:A, A:]

    nom = 2*cross_overlap
    denom = np.expand_dims(self_overlap_A, axis=1) + np.expand_dims(self_overlap_B, axis=0)

    total_overlap = nom/denom

    return total_overlap


def inverse_overlap(conf, params, box, lamb, a_idxs, b_idxs):

    V = overlap(conf, params, a_idxs, b_idxs)

    return np.sum(1/V)