from http.client import CONFLICT
import jax.numpy as jnp
import numpy as np


def centroid_restraint(conf, params, box, lamb, group_a_idxs, group_b_idxs, kb, b0):
    """Computes kb  * (r - b0)**2 where r is the distance between the centroids of group_a and group_b

    Notes
    ------
    * Geometric centroid, not mass-weighted centroid
    * Gradient undefined when `(r - b0) == 0` and `b0 != 0` (explicitly stabilized in case `b0 == 0`)
    """
    xi = conf[group_a_idxs]
    xj = conf[group_b_idxs]

    avg_xi = jnp.mean(xi, axis=0)
    avg_xj = jnp.mean(xj, axis=0)

    dx = avg_xi - avg_xj
    d2ij = jnp.sum(dx * dx)
    d2ij = jnp.where(d2ij == 0, 0, d2ij)  # stabilize derivative
    dij = jnp.sqrt(d2ij)
    delta = dij - b0

    # when b0 == 0 and dij == 0
    return jnp.where(b0 == 0, kb * d2ij, kb * delta ** 2)


def restraint(conf, lamb, params, lamb_flags, box, bond_idxs):
    r"""
    Compute the harmonic bond energy given a collection of molecules.

    This implements a harmonic bond potential:
        V(conf; lamb) = \sum_bond kbs[bond] term[bond]^2

        where term[bond] = 1 - exp(-a0s[bond]*(dij[bond] - b0s[bond]))
        and where
            dij[bond] = distance[bond] + f_lambda^2
            and where
                f_lambda = lamb * lamb_flags

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    lamb: float
        lambda value for 4d decoupling

    params: shape [num_params,] np.array
        unique parameters

    lamb_flags: np.array
        equivalent to offset_idxs, adjusts how much we offset by

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    bond_idxs: [num_bonds, 2] np.array
        each element (src, dst) is a unique bond in the conformation

    Notes:
    ------
    * box argument is unused
    """
    f_lambda = lamb * lamb_flags

    ci = conf[bond_idxs[:, 0]]
    cj = conf[bond_idxs[:, 1]]

    dij = jnp.sqrt(jnp.sum(jnp.power(ci - cj, 2), axis=-1) + f_lambda * f_lambda)
    kbs = params[:, 0]
    b0s = params[:, 1]
    a0s = params[:, 2]

    term = 1 - jnp.exp(-a0s * (dij - b0s))

    energy = jnp.sum(kbs * term * term)

    return energy


# lamb is *not used* it is used in the alchemical stuff after
def harmonic_bond(conf, params, box, lamb, bond_idxs, lamb_mult=None, lamb_offset=None):
    r"""
    Compute the harmonic bond energy given a collection of molecules.

    This implements a harmonic bond potential:
        V(conf) = \sum_bond kbs[bond] * (distance[bond] - r0s[bond])^2

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_bonds, 2] np.array
        parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    lamb: float
        alchemical lambda parameter, linearly rescaled

    lamb_mult: None, or broadcastable to bond_idxs.shape[0]
        prefactor = (lamb_offset + lamb_mult * lamb)

    lamb_offset: None, or broadcastable to bond_idxs.shape[0]
        prefactor = (lamb_offset + lamb_mult * lamb)

    bond_idxs: [num_bonds, 2] np.array
        each element (src, dst) is a unique bond in the conformation

    Notes:
    ------
    * lamb argument is unused

    """

    bond_idxs = np.array(bond_idxs)
    params = np.array(params)

    assert params.shape == bond_idxs.shape

    if bond_idxs.shape[0] == 0:
        return 0.0

    if lamb_mult is None or lamb_offset is None or lamb is None:
        assert lamb_mult is None
        assert lamb_offset is None
        prefactor = 1.0
    else:
        assert lamb_mult is not None
        assert lamb_offset is not None
        prefactor = lamb_offset + lamb_mult * lamb

    ci = conf[bond_idxs[:, 0]]
    cj = conf[bond_idxs[:, 1]]

    cij = ci - cj
    d2ij = jnp.sum(cij * cij, axis=-1)
    d2ij = jnp.where(d2ij == 0, 0, d2ij)  # stabilize derivative
    dij = jnp.sqrt(d2ij)
    kbs = params[:, 0]
    r0s = params[:, 1]

    # this is here to prevent a numerical instability
    # when b0 == 0 and dij == 0
    energy = jnp.where(r0s == 0, prefactor * kbs / 2 * d2ij, prefactor * kbs / 2 * jnp.power(dij - r0s, 2.0))

    return jnp.sum(energy)

def get_centroid_cos_angles(conf, angle_idxs):

    vij = []
    vik = []


    for idxs, j, k in angle_idxs:

        x_j = conf[j]
        # compute v-site from unit vectors relative to i
        if len(idxs) == 2:
            a, b = idxs
            x_a, x_b = conf[a], conf[b]
            x_ia = (x_a - x_j)
            x_ia = x_j + x_ia/jnp.linalg.norm(x_ia)
            x_ib = (x_b - x_j)
            x_ib = x_j + x_ib/jnp.linalg.norm(x_ib)
            x_i = (x_ia + x_ib)/2.0
        elif len(idxs) == 3:
            a, b, c = idxs
            x_a, x_b, x_c = conf[a], conf[b], conf[c]
            x_ia = (x_a - x_j)
            x_ia = x_j + x_ia/jnp.linalg.norm(x_ia)
            x_ib = (x_b - x_j)
            x_ib = x_j + x_ib/jnp.linalg.norm(x_ib)
            x_ic = (x_c - x_j)
            x_ic = x_j + x_ic/jnp.linalg.norm(x_ic)
            x_i = (x_ia + x_ib + x_ic)/3.0
        else:
            assert 0

        vij.append(x_i-x_j)
        x_k = conf[k]
        vik.append(x_k-x_j)

    vij = jnp.array(vij)
    vik = jnp.array(vik)

    top = jnp.sum(jnp.multiply(vij, vik), -1)
    bot = jnp.linalg.norm(vij, axis=-1) * jnp.linalg.norm(vik, axis=-1)

    tb = top / bot

    return tb


def harmonic_x_angle(conf, params, box, lamb, angle_idxs):
    if len(angle_idxs) == 0:
        return 0.0

    params = jnp.array(params)
    v_ij = []
    v_ik = []

    for (j, a), (j, b), (j, c) in angle_idxs:
        x_a = conf[a]
        x_b = conf[b]
        x_c = conf[c]
        x_j = conf[j]
        v_ja = x_a - x_j
        v_jb = x_b - x_j
        v_jc = x_c - x_j
        v_ij.append(jnp.cross(v_ja, v_jb))
        v_ik.append(v_jc)

    v_ij = jnp.array(v_ij)
    v_ik = jnp.array(v_ik)
    top = jnp.sum(jnp.multiply(v_ij, v_ik), -1)
    bot = jnp.linalg.norm(v_ij, axis=-1) * jnp.linalg.norm(v_ik, axis=-1)
    cos_angles = top / bot
    kas = params[:, 0]
    cos_2_angles = 2*cos_angles**2 - 1 # double angle - symmetrized to both ends
    energies = kas / 2 * jnp.power(cos_2_angles - 1, 2)
    return jnp.sum(energies)


def harmonic_c_angle(conf, params, box, lamb, angle_idxs):
    if len(angle_idxs) == 0:
        return 0.0

    params = jnp.array(params)
    tbs = get_centroid_cos_angles(conf, angle_idxs)
    kas = params[:, 0]
    a0s = params[:, 1]

    # we have to use the cos_angle form here since we often set a0s to pi
    energies = kas / 2 * jnp.power(tbs - jnp.cos(a0s), 2)
    return jnp.sum(energies)
    

def harmonic_angle(conf, params, box, lamb, angle_idxs, lamb_mult=None, lamb_offset=None, cos_angles=True):
    """
    Compute the harmonic angle energy given a collection of molecules.

    This implements a harmonic angle potential:
        V(t) = k*(t - t0)^2
            if cos_angles=False
        or
        V(t) = k*(cos(t)-cos(t0))^2
            if cos_angles=True


    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_angles, 2] np.array
        parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    lamb: float
        alchemical lambda parameter, linearly rescaled

    lamb_mult: None, or broadcastable to angle_idxs.shape[0]
        prefactor = (lamb_offset + lamb_mult * lamb)

    lamb_offset: None, or broadcastable to angle_idxs.shape[0]
        prefactor = (lamb_offset + lamb_mult * lamb)

    angle_idxs: shape [num_angles, 3] np.array
        each element (a, b, c) is a unique angle in the conformation. atom b is defined
        to be the middle atom.

    cos_angles: True (default)
        if True, then this instead implements V(t) = k*(cos(t)-cos(t0))^2. This is far more
        numerically stable when the angle is pi.

    Notes:
    ------
    * lamb argument unused
    """

    angle_idxs = np.array(angle_idxs)
    params = np.array(params)

    if angle_idxs.shape[0] == 0:
        return 0.0

    if lamb_mult is None or lamb_offset is None or lamb is None:
        assert lamb_mult is None
        assert lamb_offset is None
        prefactor = 1.0
    else:
        assert lamb_mult is not None
        assert lamb_offset is not None
        prefactor = lamb_offset + lamb_mult * lamb

    ci = conf[angle_idxs[:, 0]]
    cj = conf[angle_idxs[:, 1]]
    ck = conf[angle_idxs[:, 2]]

    kas = params[:, 0]
    a0s = params[:, 1]

    vij = ci - cj
    vjk = ck - cj

    top = jnp.sum(jnp.multiply(vij, vjk), -1)
    bot = jnp.linalg.norm(vij, axis=-1) * jnp.linalg.norm(vjk, axis=-1)

    tb = top / bot

    # (ytz): we use the squared version so that the energy is strictly positive
    if cos_angles:
        energies = prefactor * kas / 2 * jnp.power(tb - jnp.cos(a0s), 2)
    else:
        angle = jnp.arccos(tb)
        energies = prefactor * kas / 2 * jnp.power(angle - a0s, 2)

    return jnp.sum(energies, -1)  # reduce over all angles


def signed_torsion_angle(ci, cj, ck, cl):
    """
    Batch compute the signed angle of a torsion angle.  The torsion angle
    between two planes should be periodic but not necessarily symmetric.

    Parameters
    ----------
    ci: shape [num_torsions, 3] np.array
        coordinates of the 1st atom in the 1-4 torsion angle

    cj: shape [num_torsions, 3] np.array
        coordinates of the 2nd atom in the 1-4 torsion angle

    ck: shape [num_torsions, 3] np.array
        coordinates of the 3rd atom in the 1-4 torsion angle

    cl: shape [num_torsions, 3] np.array
        coordinates of the 4th atom in the 1-4 torsion angle

    Returns
    -------
    shape [num_torsions,] np.array
        array of torsion angles.
    """

    # Taken from the wikipedia arctan2 implementation:
    # https://en.wikipedia.org/wiki/Dihedral_angle

    # We use an identical but numerically stable arctan2
    # implementation as opposed to the OpenMM energy function to
    # avoid a singularity when the angle is zero.

    rij = cj - ci
    rkj = cj - ck
    rkl = cl - ck

    n1 = jnp.cross(rij, rkj)
    n2 = jnp.cross(rkj, rkl)

    y = jnp.sum(jnp.multiply(jnp.cross(n1, n2), rkj / jnp.linalg.norm(rkj, axis=-1, keepdims=True)), axis=-1)
    x = jnp.sum(jnp.multiply(n1, n2), -1)

    return jnp.arctan2(y, x)


def periodic_torsion(conf, params, box, lamb, torsion_idxs, lamb_mult=None, lamb_offset=None):
    """
    Compute the periodic torsional energy.

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_torsions, 3] np.array
        parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    lamb: float
        alchemical lambda parameter, linearly rescaled

    lamb_mult: None, or broadcastable to torsion_idxs.shape[0]
        prefactor = (lamb_offset + lamb_mult * lamb)

    lamb_offset: None, or broadcastable to torsion_idxs.shape[0]
        prefactor = (lamb_offset + lamb_mult * lamb)

    torsion_idxs: shape [num_torsions, 4] np.array
        indices denoting the four atoms that define a torsion

    Notes:
    ------
    * box argument unused
    * lamb argument unused
    * if conf has more than 3 dimensions, this function only depends on the first 3
    """

    torsion_idxs = np.array(torsion_idxs)
    params = np.array(params)

    if torsion_idxs.shape[0] == 0:
        return 0.0

    if lamb_mult is None:
        lamb_mult = np.zeros(torsion_idxs.shape[0])
    if lamb_offset is None:
        lamb_offset = np.ones(torsion_idxs.shape[0])

    conf = conf[:, :3]  # this is defined only in 3d

    ci = conf[torsion_idxs[:, 0]]
    cj = conf[torsion_idxs[:, 1]]
    ck = conf[torsion_idxs[:, 2]]
    cl = conf[torsion_idxs[:, 3]]

    ks = params[:, 0]
    phase = params[:, 1]
    period = params[:, 2]
    angle = signed_torsion_angle(ci, cj, ck, cl)

    prefactor = lamb_offset + lamb_mult * lamb

    nrg = ks * (1 + jnp.cos(period * angle - phase))
    return jnp.sum(prefactor * nrg, axis=-1)
