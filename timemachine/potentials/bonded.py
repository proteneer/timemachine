import jax.numpy as np

from timemachine.potentials.jax_utils import distance, delta_r, convert_to_4d


def boresch_torsion(conf, params, torsion_idxs):
    """
    Compute the periodic torsional energy.

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_params,] np.array
        unique parameters

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

    ks = params[:, 0]
    phase = params[:, 1]
    angle = signed_torsion_angle(ci, cj, ck, cl)

    nrg = ks/2*(1-np.cos(angle - phase))
    return np.sum(nrg, axis=-1)

def boresch_like_restraint(
    conf,
    lamb,
    lamb_flag,
    lamb_offset,
    bond_idxs,
    bond_params,
    angle_idxs,
    angle_params,
    torsion_idxs,
    torsion_params):

    assert bond_idxs.shape[0] == 1
    assert bond_idxs.shape[1] == 2

    assert bond_params.shape[0] == 1
    assert bond_params.shape[1] == 2

    assert angle_idxs.shape[0] == 2
    assert angle_idxs.shape[1] == 3

    assert angle_params.shape[0] == 2
    assert angle_params.shape[1] == 2

    assert torsion_idxs.shape[0] == 3
    assert torsion_idxs.shape[1] == 4

    assert torsion_params.shape[0] == 3
    assert torsion_params.shape[1] == 2

    lamb_final = lamb*lamb_flag + lamb_offset
    bond_nrg = harmonic_bond(conf, 1.0, bond_params, None, bond_idxs)
    # warning use the cos angle form when integrating
    angle_nrg = harmonic_angle(conf, 1.0, angle_params, None, angle_idxs, cos_angles=True)
    torsion_nrg = boresch_torsion(conf, torsion_params, torsion_idxs)

    return lamb_final*(angle_nrg + bond_nrg + torsion_nrg)


def centroid_restraint(conf, lamb, params, lamb_flag, lamb_offset, group_a_idxs, group_b_idxs, kb, b0):

    xi = conf[group_a_idxs]
    xj = conf[group_b_idxs]

    avg_xi = np.mean(xi, axis=0)
    avg_xj = np.mean(xj, axis=0)

    dx = avg_xi - avg_xj
    dij = np.sqrt(np.sum(dx*dx))
    delta = dij - b0

    lamb_final = lamb*lamb_flag + lamb_offset

    return lamb_final*kb*delta*delta

def restraint(conf, lamb, params, lamb_flags, box, bond_idxs):
    """
    Compute the harmonic bond energy given a collection of molecules.

    This implements a harmonic angle potential: V(t) = k*(b - b0)^2

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_params,] np.array
        unique parameters

    lamb: float
        lambda value for 4d decoupling

    lamb_flags: np.array
        equivalent to offset_idxs, adjusts how much we offset by

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    bond_idxs: [num_bonds, 2] np.array
        each element (src, dst) is a unique bond in the conformation

    param_idxs: [num_bonds, 2] np.array
        each element (k_idx, r_idx) maps into params for bond constants and ideal lengths

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

# lamb is *not used* it is used in the alchemical stuffl ater
def harmonic_bond(conf, lamb, params, box, bond_idxs):
    """
    Compute the harmonic bond energy given a collection of molecules.

    This implements a harmonic angle potential: V(t) = k*(b - b0)^2

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_params, 2] np.array
        unique parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    bond_idxs: [num_bonds, 2] np.array
        each element (src, dst) is a unique bond in the conformation

    param_idxs: [num_bonds, 2] np.array
        each element (k_idx, r_idx) maps into params for bond constants and ideal lengths

    """
    assert params.shape == bond_idxs.shape

    ci = conf[bond_idxs[:, 0]]
    cj = conf[bond_idxs[:, 1]]
    dij = distance(ci, cj, box)
    kbs = params[:, 0]
    r0s = params[:, 1]

    energy = np.sum(kbs/2 * np.power(dij - r0s, 2.0))
    return energy


def harmonic_angle(conf, lamb, params, box, angle_idxs, cos_angles=True):
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

    kas = params[:, 0]
    a0s = params[:, 1]

    vij = delta_r(ci, cj, box)
    vjk = delta_r(ck, cj, box)

    top = np.sum(np.multiply(vij, vjk), -1)
    bot = np.linalg.norm(vij, axis=-1)*np.linalg.norm(vjk, axis=-1)

    tb = top/bot

    # (ytz): we used the squared version so that we make this energy being strictly positive
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


def periodic_torsion(conf, lamb, params, box, torsion_idxs):
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

    ks = params[:, 0]
    phase = params[:, 1]
    period = params[:, 2]
    angle = signed_torsion_angle(ci, cj, ck, cl)

    nrg = ks*(1+np.cos(period * angle - phase))
    return np.sum(nrg, axis=-1)