from timemachine.constants import BOLTZ
import numpy as np


def langevin_coefficients(
    temperature,
    dt,
    friction,
    masses):
    """
    Compute coefficients for langevin dynamics

    Parameters
    ----------
    temperature: float
        units of Kelvin

    dt: float
        units of picoseconds

    friction: float
        frequency in picoseconds

    masses: array
        mass of each atom in standard mass units

    Returns
    -------
    tuple (ca, cb, cc)
        ca is scalar, and cb and cc are n length arrays
        that are used during langevin dynamics


    """
    vscale = np.exp(-dt*friction)
    if friction == 0:
        fscale = dt
    else:
        fscale = (1-vscale)/friction
    kT = BOLTZ * temperature
    nscale = np.sqrt(kT*(1-vscale*vscale)) # noise scale
    invMasses = 1.0/masses
    sqrtInvMasses = np.sqrt(invMasses)

    ca = vscale
    cb = fscale*invMasses
    cc = nscale*sqrtInvMasses
    return ca, cb, cc

def brownian_coefficients(
    temperature,
    dt,
    friction,
    masses):
    """
    Compute coefficients for langevin dynamics

    Parameters
    ----------
    temperature: float
        units of Kelvin

    dt: float
        units of picoseconds

    friction: float
        frequency in picoseconds

    masses: array
        mass of each atom in standard mass units

    Returns
    -------
    tuple (ca, cb, cc)
        ca is scalar, and cb and cc are n length arrays
        that are used during langevin dynamics

    """
    fscale = dt/friction
    kT = BOLTZ * temperature

    invMasses = 1.0/masses
    sqrtInvMasses = np.sqrt(invMasses)

    nscale = np.sqrt(2.0*kT*dt/friction);

    ca = 0.0
    cb = fscale*invMasses
    cc = nscale*sqrtInvMasses
    return ca, cb, cc