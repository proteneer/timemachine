# This is ported from Matt Harrigan's TF code:
# https://github.com/mdtraj/tftraj/blob/master/tftraj/rmsd.py

import jax.scipy
import jax.numpy as np

@jax.jit
def optimal_rotational_quaternion(r):
    """Just need the largest eigenvalue of this to minimize RMSD over rotations
    
    References
    ----------
    [1] http://dx.doi.org/10.1002/jcc.20110
    """
    # @formatter:off
    return np.array([
        [r[0][0] + r[1][1] + r[2][2], r[1][2] - r[2][1], r[2][0] - r[0][2], r[0][1] - r[1][0]],
        [r[1][2] - r[2][1], r[0][0] - r[1][1] - r[2][2], r[0][1] + r[1][0], r[0][2] + r[2][0]],
        [r[2][0] - r[0][2], r[0][1] + r[1][0], -r[0][0] + r[1][1] - r[2][2], r[1][2] + r[2][1]],
        [r[0][1] - r[1][0], r[0][2] + r[2][0], r[1][2] + r[2][1], -r[0][0] - r[1][1] + r[2][2]],
    ])

@jax.jit
def squared_deviation(frame, target):
    R = np.matmul(np.transpose(frame), target)
    F = optimal_rotational_quaternion(R)
    vals, vecs = jax.scipy.linalg.eigh(F)
    lmax = vals[-1]
    sd = np.sum(frame ** 2 + target ** 2) - 2 * lmax
    # singularities occur when sd is a very small negative number
    return np.maximum(sd, 0)

@jax.jit
def opt_rot_rmsd(x0, x1):
    """
    Compute the optimally rotated root mean squared deviation between two
    sets of points x0 and x1.

    Parameters
    ----------
    x0: shape (num_atoms, 3) np.array
        First set of coordinates

    x1: shape (num_atoms, 3) np.array
        Second set of coordinates

    Returns
    -------
    scalar
        A scalar denoting the distance

    """
    x0_center = x0 - np.mean(x0, axis=0, keepdims=True)
    x1_center = x1 - np.mean(x1, axis=0, keepdims=True)
    n_atoms = x0.shape[0]
    inner = squared_deviation(x0_center, x1_center)/n_atoms
    inner = np.where(inner < 1e-12, 0, inner)
    return np.sqrt(inner)
