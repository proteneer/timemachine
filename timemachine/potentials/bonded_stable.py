import jax.numpy as jnp


def kahan_angle_stable(ci, cj, ck, eps):
    """
    Compute the angle given three points, i,j,k, as defined by the vector j->i, j->k
    """
    rji = ci - cj
    rjk = ck - cj
    rji = jnp.hstack([ci - cj, jnp.expand_dims(eps, axis=-1)])
    rjk = jnp.hstack([ck - cj, jnp.expand_dims(eps, axis=-1)])
    nji = jnp.linalg.norm(rji, axis=-1)
    njk = jnp.linalg.norm(rjk, axis=-1)
    nji = jnp.expand_dims(nji, axis=-1)
    njk = jnp.expand_dims(njk, axis=-1)
    y = jnp.linalg.norm(njk * rji - nji * rjk, axis=-1)
    x = jnp.linalg.norm(njk * rji + nji * rjk, axis=-1)
    angle = 2 * jnp.arctan2(y, x)
    return angle


def harmonic_angle_stable(conf, params, angle_idxs):
    r"""
    Compute the harmonic angle energy using a numerically stable approximation.

    The functional form is identical to :py:func:`potentials.bonded.HarmonicAngle`, except that the following
    approximation is used for the intermediate computation of :math:`\cos(\theta)`:

    :math::

        \cos(\theta) \approx \frac{r_{ij} \cdot r_{kj}}{\sqrt{(r_{ij}^2 + \epsilon^2) (r_{kj}^2 + \epsilon^2)}}

    This reduces to the exact expression when :math:`\epsilon = 0`; When :math:`\epsilon > 0`, this avoids the
    singularities in the exact expression as :math:`r_{ij}` or :math:`r_{kj}` approach zero.

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.ndarray
        atomic coordinates

    params: shape [num_angles, 3] np.ndarray
        force constants, eq angles, epsilons
        (kas, a0s, epsilons = params.T)

    angle_idxs: shape [num_angles, 3] np.ndarray
        each element (i, j, k) is a unique angle in the conformation. Atom j is defined
        to be the middle atom.

    """
    if angle_idxs.shape[0] == 0:
        return 0.0
    ci, cj, ck = conf[angle_idxs.T]
    kas, a0s, eps = params.T
    angle = kahan_angle_stable(ci, cj, ck, eps)
    energies = kas / 2 * jnp.power(angle - a0s, 2)
    return jnp.sum(energies, -1)
