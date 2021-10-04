import jax
import jax.numpy as np

def psi(rotation, k):
    cos_theta = (np.trace(rotation) - 1)/2
    return cos_angle_u(cos_theta, k)

def angle_u(theta, k):
    return cos_angle_u(np.cos(theta), k)

def cos_angle_u(cos_theta, k):
    term = cos_theta - 1
    nrg = k*term*term
    return nrg


def get_optimal_rotation(x1, x2):
    # x1, x2 must be centered
    assert x1.shape == x2.shape

    # x1 and x2 must be already mean aligned.
    correlation_matrix = np.dot(x2.T, x1)
    U, S, V_tr = np.linalg.svd(correlation_matrix, full_matrices=False)
    is_reflection = (np.linalg.det(U) * np.linalg.det(V_tr)) < 0.0
    U = jax.ops.index_update(U,
        jax.ops.index[:, -1],
        np.where(is_reflection, -U[:, -1], U[:, -1])
    )
    rotation = np.dot(U, V_tr)

    return rotation

def get_optimal_translation(x1, x2):
    """
    Returns the displacement vector whose tail is at x1 and head its at x2.
    """
    return np.mean(x2, axis=0) - np.mean(x1, axis=0)

def get_optimal_rotation_and_translation(x1, x2):
    """
    Compute the optimal rotation and translation of x2 unto x1.

    Parameters
    ----------
    x1: np.array (K,3)

    x2: np.array (K,3)

    Returns
    -------
    tuple (np.array, np.array)
        Rotation translation pair
    """
    t = get_optimal_translation(x1, x2)
    x1 = x1 - np.mean(x1, axis=0)
    x2 = x2 - np.mean(x2, axis=0)
    return get_optimal_rotation(x1, x2), t

def apply_rotation_and_translation(x, R, t):
    """
    Apply R and t from x.
    """
    x_com = np.mean(x, axis=0)
    aligned_x = (x - x_com)@R - t + x_com
    return aligned_x

def align_x2_unto_x1(x1, x2):
    com1 = np.mean(x1, axis=0)
    com2 = np.mean(x2, axis=0)
    t = com2 - com1
    x1_centered = x1 - com1
    x2_centered = x2 - com2

    R = get_optimal_rotation(x1_centered, x2_centered)

    return x2_centered@R + com2 - t

def rmsd_align(x1, x2):
    """
    Optimally align x1 and x2 via rigid translation and rotations.

    The returned alignment is a proper rotation. Note while it is technically
    possible to find an ever better alignment if we were to allow for reflections,
    there are technical difficulties with defining what the "standard" rotation is,
    i.e. either np.eye(3) or -np.eye(3). We can revisit this at a later time.

    Parameters
    ----------
    x1: np.ndarray (N,3)
        conformation 1

    x1: np.ndarray (N,3)
        conformation 2

    Returns
    -------
    2-tuple
        Return a pair of aligned (N,3) ndarrays. Each conformer has its centroid
        set to the origin.

    """

    assert x1.shape == x2.shape

    x1 = x1 - np.mean(x1, axis=0)
    x2 = x2 - np.mean(x2, axis=0)

    rotation = get_optimal_rotation(x1, x2)

    xa = x1
    xb = x2@rotation

    return xa, xb


def rmsd_restraint(conf, params, box, lamb, group_a_idxs, group_b_idxs, k):
    """
    Compute a rigid RMSD restraint using two groups of atoms. group_a_idxs and group_b_idxs
    must have the same size. a and b can have duplicate indices and need not be necessarily
    disjoint. This function will automatically recenter the two groups of atoms before computing
    the rotation matrix. For relative binding free energy calculations, this restraint
    does not need to be turned off.

    Note that you should add a center of mass restraint as well to accomodate for the translational
    component.

    Parameters
    ----------
    conf: np.ndarray
        N x 3 coordinates

    params: Any
        Unused dummy variable for API consistency

    box: Any
        Unused dummy variable for API consistency

    lamb: Any
        Unused dummy variable for API consistency

    group_a_idxs: list of int
        idxs for the first group of atoms

    group_b_idxs: list of int
        idxs for the second group of atoms

    k: float
        force constant

    """
    assert len(group_a_idxs) == len(group_b_idxs)

    x1 = conf[group_a_idxs]
    x2 = conf[group_b_idxs]
    # recenter
    x1 = x1 - np.mean(x1, axis=0)
    x2 = x2 - np.mean(x2, axis=0)

    rotation = get_optimal_rotation(x1, x2)

    return psi(rotation, k)
