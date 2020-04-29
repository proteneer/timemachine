import jax.numpy as np

from timemachine.potentials.jax_utils import distance, delta_r


# lamb is *not used* it is used in the alchemical stuffl ater
def harmonic_bond(conf, params, lamb, box, bond_idxs, param_idxs):
    """
    Compute the harmonic bond energy given a collection of molecules.

    This implements a harmonic angle potential: V(t) = k*(t - t0)^2 or V(t) = k*(cos(t)-cos(t0))^2

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_params,] np.array
        unique parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    bond_idxs: [num_bonds, 2] np.array
        each element (src, dst) is a unique bond in the conformation

    param_idxs: [num_bonds, 2] np.array
        each element (k_idx, r_idx) maps into params for bond constants and ideal lengths

    """
    ci = conf[bond_idxs[:, 0]]
    cj = conf[bond_idxs[:, 1]]
    dij = distance(ci, cj, box)
    kbs = params[param_idxs[:, 0]]
    r0s = params[param_idxs[:, 1]]

    energy = np.sum(kbs/2 * np.power(dij - r0s, 2.0))
    return energy


def harmonic_angle(conf, params, lamb, box, angle_idxs, param_idxs, lambda_idxs, cos_angles=True):
    """
    Compute the harmonic bond energy given a collection of molecules.

    This implements a harmonic angle potential: V(t) = k*(t - t0)^2 or V(t) = k*(cos(t)-cos(t0))^2

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_params,] np.array
        unique parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    angle_idxs: shape [num_angles, 3] np.array
        each element (a, b, c) is a unique angle in the conformation. atom b is defined
        to be the middle atom.

    param_idxs: shape [num_angles, 2] np.array
        each element (k_idx, t_idx) maps into params for angle constants and ideal angles

    cos_angles: True (default)
        if True, then this instead implements V(t) = k*(cos(t)-cos(t0))^2. This is far more
        numerically stable when the angle is pi.

    """
    ci = conf[angle_idxs[:, 0]]
    cj = conf[angle_idxs[:, 1]]
    ck = conf[angle_idxs[:, 2]]

    kas = params[param_idxs[:, 0]]
    a0s = params[param_idxs[:, 1]]

    vij = delta_r(ci, cj, box)
    vjk = delta_r(ck, cj, box)

    top = np.sum(np.multiply(vij, vjk), -1)
    bot = np.linalg.norm(vij, axis=-1)*np.linalg.norm(vjk, axis=-1)

    tb = top/bot

    # (ytz): we used the squared version so that we make this energy being strictly positive
    if cos_angles:
        energies = prefactors*kas/2*np.power(tb - np.cos(a0s), 2)
    else:
        angle = np.arccos(tb)
        energies = prefactors*kas/2*np.power(angle - a0s, 2)

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
    # avoid asingularity when the angle is zero.

    rij = delta_r(cj, ci)
    rkj = delta_r(cj, ck)
    rkl = delta_r(cl, ck)

    n1 = np.cross(rij, rkj)
    n2 = np.cross(rkj, rkl)

    lhs = np.linalg.norm(n1, axis=-1)
    rhs = np.linalg.norm(n2, axis=-1)
    bot = lhs * rhs

    y = np.sum(np.multiply(np.cross(n1, n2), rkj/np.linalg.norm(rkj, axis=-1, keepdims=True)), axis=-1)
    x = np.sum(np.multiply(n1, n2), -1)

    return np.arctan2(y, x)


def periodic_torsion(conf, params, lamb, box, torsion_idxs, param_idxs, lambda_idxs):
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

    torsion_idxs: shape [num_torsions, 4] np.array
        indices denoting the four atoms that define a torsion

    param_idxs: shape [num_torsions, 3] np.array
        indices into the params array denoting the force constant, phase, and period
    
    """

    conf = conf[:, :3] # this is defined only in 3d

    ci = conf[torsion_idxs[:, 0]]
    cj = conf[torsion_idxs[:, 1]]
    ck = conf[torsion_idxs[:, 2]]
    cl = conf[torsion_idxs[:, 3]]

    ks = params[param_idxs[:, 0]]
    phase = params[param_idxs[:, 1]]
    period = params[param_idxs[:, 2]]
    angle = signed_torsion_angle(ci, cj, ck, cl)

    prefactors_c = np.where(lambda_idxs ==  0, 1, 0)
    prefactors_a = np.where(lambda_idxs ==  1, lamb, 0)
    prefactors_b = np.where(lambda_idxs == -1, 1-lamb, 0)
    prefactors = np.stack([prefactors_a, prefactors_b, prefactors_c])
    prefactors = np.sum(prefactors, axis=0)

    nrg = prefactors*ks*(1+np.cos(period * angle - phase))
    return np.sum(nrg, axis=-1)