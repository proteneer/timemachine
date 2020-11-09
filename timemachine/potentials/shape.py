import jax.numpy as np

def squared_distance(ci, cj):
    diff = ci - cj
    return np.sum(diff*diff, axis=-1)


def volume(conf_a, params_a, conf_b, params_b):
    """
    Given two molecules, compute the second order volume Vij between them. It's
    important the units in the coordinates conf_x are consistent with alpha
    coefficients (params_x[:, 0]). This will likely overestimate the true
    volume as the third and higher order terms are ignored.

    Parameters
    ----------
    conf_a: np.array [A, 3]
        point cloud of coordinates for molecule A

    params_a: np.array [A, 2]
        parameters where [:, 0] are alphas in Grant's paper, and [:, 1] 
        are the weights/prefactors outside of the exponential

    conf_b: np.array [B, 3]
        point cloud of coordinates for molecule B

    params_b: np.array [A, 2]
        identical to params_a in semantics

    """
    ci = np.expand_dims(conf_a, axis=1) # N, 1, 3
    cj = np.expand_dims(conf_b, axis=0) # 1, N, 3

    d2ij = squared_distance(ci, cj)

    ai = np.expand_dims(params_a[:, 0], axis=1)
    aj = np.expand_dims(params_b[:, 0], axis=0)

    pi = np.expand_dims(params_a[:, 1], axis=1)
    pj = np.expand_dims(params_b[:, 1], axis=0)

    kij = np.exp(-(ai*aj*d2ij)/(ai+aj))
    vij = pi*pj*kij*np.power(np.pi/(ai+aj), 3/2)

    return np.sum(vij)


def normalized_overlap(conf_a, params_a, conf_b, params_b):
    """
    Compute the normalized overlap volume defined by Grant 1996,
    equation 20:

    S_AB =  2*int dr rho_A * rho_B
           ------------------------
           int dr rho_A^2 + rho_B^2

    Guaranteeing that 0 <= S_AB <= 1. An alternative would be
    something like the Tanimoto. 

    Parameters
    ----------
    conf_a: np.array [A, 3]
        point cloud of coordinates for molecule A

    params_a: np.array [A, 2]
        parameters where [:, 0] are alphas in Grant's paper, and [:, 1] 
        are the weights/prefactors outside of the exponential

    conf_b: np.array [B, 3]
        point cloud of coordinates for molecule B

    params_b: np.array [A, 2]
        identical to params_a in semantics

    """
    vij = volume(conf_a, params_a, conf_b, params_b)
    vii = volume(conf_a, params_a, conf_a, params_a)
    vjj = volume(conf_b, params_b, conf_b, params_b)

    # (ytz): try tanimoto etc. later as well
    return 2*vij/(vii+vjj)


def harmonic_overlap(conf, params, box, lamb, a_idxs, b_idxs, alphas, weights, k):
    """
    Compute the shape potential. The derivative of this function
    w.r.t. conf generates a non-rigid force.

    Parameters
    ----------
    conf: np.array [N, 3]
        Conformation of the system

    params: None
        unused - dummy parameter

    box: None
        unused - dummy parameter

    lamb: None
        unused - dummy parameter

    a_idxs: np.array [A]
        molecule A's indices into the conformation

    b_idxs: np.array [B]
        molecule B's indices into the conformation

    alphas: np.array float64 [N]
        factor inside exponential

    weights: np.array float64 [N]
        factor outside exponential

    """

    conf_a = conf[a_idxs]
    conf_b = conf[b_idxs]

    params_c = np.stack([alphas, weights], axis=1)

    params_a = params_c[a_idxs]
    params_b = params_c[b_idxs]

    V = normalized_overlap(conf_a, params_a, conf_b, params_b)
    return k*(V-1)**2

from timemachine.potentials.jax_utils import convert_to_4d

def harmonic_4d_overlap(conf, params, box, lamb, a_idxs, b_idxs, alphas, weights, k):

    S = len(a_idxs) + len(b_idxs)

    lambda_plane_idxs = np.zeros(S, dtype=np.int32)
    lambda_offset_idxs = np.concatenate([
        np.zeros(len(a_idxs), dtype=np.int32),
        np.ones(len(b_idxs), dtype=np.int32)
    ])
    cutoff = 100.0

    conf_a = conf[a_idxs]
    conf_b = conf[b_idxs]

    conf_a = convert_to_4d(conf_a, lamb, np.zeros(len(a_idxs)), np.zeros(len(a_idxs)), cutoff)
    conf_b = convert_to_4d(conf_b, lamb, np.zeros(len(b_idxs)), np.ones(len(b_idxs)), cutoff)

    params_c = np.stack([alphas, weights], axis=1)

    params_a = params_c[a_idxs]
    params_b = params_c[b_idxs]

    V = normalized_overlap(conf_a, params_a, conf_b, params_b)
    # return k*(1/V)
    return k*(V-1)**2
    # return -k*V**2
    # return -k*V