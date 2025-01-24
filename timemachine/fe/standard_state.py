import functools

import numpy as np
import scipy.integrate

from timemachine.potentials import rmsd


def integrate_radial_Z(u_fn, beta, r_max):
    """
    Evaluate the partition function of a radially symmetric
    restraint.

    Parameters:
    -----------
    u_fn: f: R -> R
        A radial function that takes in a distance r and returns
        a scalar. This function must be integrable.

    beta: float
        1/kT

    r_max: float
        upper bound of integration

    Returns
    -------
    float
        Free energy associated with release into a 1660A^3 volume.

    """

    def integrand(r):
        return 4 * np.pi * (r**2) * np.exp(-beta * u_fn(r))

    r_min = 0.0
    Z, err = scipy.integrate.quad(integrand, r_min, r_max)

    assert err < 1e-5

    return Z


def standard_state_correction(Z_infty, beta):
    """
    Compute the standard state of releasing a ligand into the standard
    molar volume.

    Parameters
    ----------
    Z_infty: float
        Partition function when integrated to infinity

    beta: float
        1/kT

    Returns
    -------
    dG
        Free energy of releasing into the standard state

    """
    return -np.log(1.660 / Z_infty) / beta  # in kJ/mol


def integrate_radial_Z_exact(k, beta):
    k = k * beta
    b = 0.0
    # this is the analytical solution of the integral of:
    # U = (1/kT)*kb*(r-b0)**2
    # int_{r=0}^{r=infty} 4.0 * np.pi * r**2 * np.exp(-U)
    Z_exact = (
        4.0
        * np.pi
        * (
            (b * np.exp(-(b**2) * k)) / (2 * k)
            + ((1 + 2 * b**2 * k) * np.sqrt(np.pi) * (1 + scipy.special.erf(b * np.sqrt(k)))) / (4 * k ** (3 / 2))
        )
    )
    return Z_exact


def integrate_rotation_Z(u_fn, beta):
    """
    Compute the partition function a rotational restraint over SO(3)

    Parameters
    ----------
    u_fn: f: R->R
        Takes in an arbitrary scalar representing an angle relative
        to the identity transformation and returns an energy.

    beta: float
        1/Kt

    Returns
    -------
    scalar
        Value of the partition function

    """
    # Integrating in the quaternion form requires only two integrals as opposed
    # to three. The general technique is outlined here. See "Average Rotation Angle"
    # for a direct analogy. The main difference is that we explicit do not compute the
    # 1/pi^2 normalization constant.

    # https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html

    def integrand(alpha, theta):
        nrg = u_fn(2 * theta)
        assert nrg > 0
        return np.exp(-beta * nrg) * np.sin(theta) ** 2 * np.sin(alpha)

    Z, Z_err = scipy.integrate.dblquad(
        integrand,
        0,  # theta low
        np.pi / 2,  # theta high
        lambda x: 0,  # alpha low
        lambda x: np.pi,  # alpha high
    )

    assert Z_err < 1e-5

    # outer integral
    Z *= 2 * np.pi
    return Z


def release_orientational_restraints(k_t, k_r, beta):
    """
    Convenience function.

    Compute the free energy of releasing orientational restraints
    into the standard state. It assumes that a harmonic translational
    restraint and an rmsd restraint is used. Do not use this function
    if you use any other type of restraint.

    The quantity computed is:

    dG_release = -1/beta ln(Z_T Z_R)

    Parameters
    ----------
    k_t: float
        Force constant of the translational restraint

    k_r: float
        Force constant of the rotational restraint

    beta: float
        1/kT

    Returns
    -------
    float, float
        dG of the translational and rotational restraint

    """

    def harmonic_restraint(r):
        return k_t * r**2

    Z_numeric = integrate_radial_Z(harmonic_restraint, beta, r_max=np.inf)  # i like to live dangerously
    Z_exact = integrate_radial_Z_exact(k_t, beta)

    np.testing.assert_almost_equal(Z_exact, Z_numeric)
    dG_translation = standard_state_correction(Z_numeric, beta)
    u_fn = functools.partial(rmsd.angle_u, k=k_r)
    Z_rotation = integrate_rotation_Z(u_fn, beta)
    # A_ij = (-1/beta)*ln(Z_j/Z_i)
    dG_rotation = (-1 / beta) * np.log(1 / Z_rotation)
    return dG_translation, dG_rotation
