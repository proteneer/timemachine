import jax.numpy as np

# from timemachine.potentials.jax_utils import distance, delta_r, convert_to_4d

def centroid_restraint(conf, params, box, lamb, masses, group_a_idxs, group_b_idxs, kb, b0):

    xi = conf[group_a_idxs]
    xj = conf[group_b_idxs]

    avg_xi = np.average(xi, axis=0, weights=masses[group_a_idxs])
    avg_xj = np.average(xj, axis=0, weights=masses[group_b_idxs])

    dx = avg_xi - avg_xj
    dij = np.sqrt(np.sum(dx*dx))
    delta = dij - b0

    return kb*delta*delta

def restraint(conf, lamb, params, lamb_flags, box, bond_idxs):
    """
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
    f_lambda = lamb*lamb_flags

    ci = conf[bond_idxs[:, 0]]
    cj = conf[bond_idxs[:, 1]]

    dij = np.sqrt(np.sum(np.power(ci - cj, 2), axis=-1) + f_lambda*f_lambda)
    kbs = params[:, 0]
    b0s = params[:, 1]
    a0s = params[:, 2]

    term = 1 - np.exp(-a0s*(dij - b0s))

    energy = np.sum(kbs * term*term)

    return energy

# lamb is *not used* it is used in the alchemical stuff after
def harmonic_bond(conf, params, box, lamb, bond_idxs):
    """
    Compute the harmonic bond energy given a collection of molecules.

    This implements a harmonic bond potential:
        V(conf) = \sum_bond kbs[bond] * (distance[bond] - r0s[bond])^2

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_params, 2] np.array
        unique parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    lamb: float

    bond_idxs: [num_bonds, 2] np.array
        each element (src, dst) is a unique bond in the conformation

    Notes:
    ------
    * lamb argument is unused

    """
    assert params.shape == bond_idxs.shape

    ci = conf[bond_idxs[:, 0]]
    cj = conf[bond_idxs[:, 1]]
    dij = np.linalg.norm(ci-cj, axis=-1)
    kbs = params[:, 0]
    r0s = params[:, 1]

    energy = np.sum(kbs/2 * np.power(dij - r0s, 2.0))
    return energy


def harmonic_angle(conf, params, box, lamb, angle_idxs, cos_angles=True):
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

    params: shape [num_params,] np.array
        unique parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    lamb: float

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

    ci = conf[angle_idxs[:, 0]]
    cj = conf[angle_idxs[:, 1]]
    ck = conf[angle_idxs[:, 2]]

    kas = params[:, 0]
    a0s = params[:, 1]

    vij = ci - cj
    vjk = ck - cj

    top = np.sum(np.multiply(vij, vjk), -1)
    bot = np.linalg.norm(vij, axis=-1)*np.linalg.norm(vjk, axis=-1)

    tb = top/bot

    # (ytz): we use the squared version so that the energy is strictly positive
    if cos_angles:
        energies = kas/2*np.power(tb - np.cos(a0s), 2)
    else:
        angle = np.arccos(tb)
        energies = kas/2*np.power(angle - a0s, 2)

    return np.sum(energies, -1)  # reduce over all angles


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

    n1 = np.cross(rij, rkj)
    n2 = np.cross(rkj, rkl)

    y = np.sum(np.multiply(np.cross(n1, n2), rkj/np.linalg.norm(rkj, axis=-1, keepdims=True)), axis=-1)
    x = np.sum(np.multiply(n1, n2), -1)

    return np.arctan2(y, x)


def periodic_torsion(conf, params, box, lamb, torsion_idxs):
    """
    Compute the periodic torsional energy.

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_params,] np.array
        unique parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    lamb: float

    torsion_idxs: shape [num_torsions, 4] np.array
        indices denoting the four atoms that define a torsion

    param_idxs: shape [num_torsions, 3] np.array
        indices into the params array denoting the force constant, phase, and period

    Notes:
    ------
    * box argument unused
    * lamb argument unused
    * if conf has more than 3 dimensions, this function only depends on the first 3
    """

    conf = conf[:, :3] # this is defined only in 3d

    ci = conf[torsion_idxs[:, 0]]
    cj = conf[torsion_idxs[:, 1]]
    ck = conf[torsion_idxs[:, 2]]
    cl = conf[torsion_idxs[:, 3]]

    ks = params[:, 0]
    phase = params[:, 1]
    period = params[:, 2]
    angle = signed_torsion_angle(ci, cj, ck, cl)

    nrg = ks*(1+np.cos(period * angle - phase))
    return np.sum(nrg, axis=-1)
