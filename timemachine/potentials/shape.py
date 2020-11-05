import jax.numpy as np

def d2ij(ci, cj):
    diff = ri - rj
    return np.dot(diff, axis=-1)


# overlap volume between two sets of points
def volume(conf_a, params_a, conf_b, params_b):

    ci = np.expand_dims(conf_a, axis=1) # N, 1, 3
    cj = np.expand_dims(conf_b, axis=0) # 1, N, 3

    dij = distance(ci, cj)
    d2ij = dij*dij

    ai = np.expand_dims(params_a[:, 0], axis=1)
    aj = np.expand_dims(params_b[:, 0], axis=0)

    pi = np.expand_dims(params_a[:, 1], axis=1)
    pj = np.expand_dims(params_b[:, 1], axis=0)

    aij = ai*aj
    kij = np.exp(-(aij*d2ij)/(ai+aj))
    pij = pi*pj
    vij = pij*kij*np.power(np.pi/(ai+aj), 3/2)

    return np.sum(vij)

def overlap(conf, params, a_idxs, b_idxs):
    """
    Compute the 2 body gaussian overlap volume.

    Parameters
    ----------

    conf: [N, 3]
    """
    conf_a = conf[a_idxs]
    conf_b = conf[b_idxs]
    params_a = params[a_idxs]
    params_b = params[b_idxs]

    vij = volume(conf_a, params_a, conf_b, params_b)
    vii = volume(conf_a, params_a, conf_a, params_a)
    vjj = volume(conf_b, params_b, conf_b, params_b)

    return 2*vij/(vii+vjj)

def inverse_overlap(conf, params, box, lamb, a_idxs, b_idxs):

    V = overlap(conf, params, a_idxs, b_idxs)
    # much more stable than 1/V variants
    return 200*(V-1)**2
