import jax.numpy as jnp


def harmonic_angle_stable(conf, params, angle_idxs, cos_angles=True):
    r"""
    Compute the harmonic angle energy using a numerically stable approximation.

    The functional form is identical to :py:func:`potentials.bonded.HarmonicAngle`, except that the following
    approximation is used for the intermediate computation of :math:`\cos(\theta)`:

    :math::

        \cos(\theta) \approx \frac{r_{ij} \cdot r_{kj} + \epsilon^2}{\sqrt{(r_{ij}^2 + \epsilon^2) (r_{kj}^2 + \epsilon^2)}}

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
        each element (a, b, c) is a unique angle in the conformation. atom b is defined
        to be the middle atom.

    cos_angles: True (default)
        if True, then this instead implements V(t) = k*(cos(t)-cos(t0))^2. This is far more
        numerically stable when the angle is pi.
    """

    if angle_idxs.shape[0] == 0:
        return 0.0

    ci, cj, ck = conf[angle_idxs.T]
    kas, a0s, eps = params.T

    vij = ci - cj
    vkj = ck - cj

    top = jnp.sum(vij * vkj, -1) + eps ** 2
    bot = jnp.sqrt((jnp.sum(vij * vij, axis=-1) + eps ** 2) * (jnp.sum(vkj * vkj, axis=-1) + eps ** 2))

    tb = top / bot

    # (ytz): we use the squared version so that the energy is strictly positive
    if cos_angles:
        energies = kas / 2 * (tb - jnp.cos(a0s)) ** 2
    else:
        angle = jnp.arccos(tb)
        energies = kas / 2 * (angle - a0s) ** 2

    return jnp.sum(energies, -1)  # reduce over all angles
