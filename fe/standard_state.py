
import simtk.unit as unit
import scipy.special
import numpy as np

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

def harmonic_com_vol(kb, b0, T):
    """
    Compute the volume component of the standard state correction.
    """
    kT = kB * (T*unit.kelvin)
    kT = kT.value_in_unit(unit.kilojoules_per_mole)
    k = kb/kT
    b = b0
    # (ytz): don't you love integrals?
    # this is the analytical solution of the integral of:
    # U = (1/kT)*kb*(r-b0)**2
    # int_{r=0}^{r=infty} 4.0 * np.pi * r**2 * np.exp(-U)   
    return 4.0 * np.pi*((b*np.exp(-b**2*k))/(2*k) + ((1 + 2*b**2*k)*np.sqrt(np.pi)*(1 + scipy.special.erf(b*np.sqrt(k))))/(4*k**(3/2)))

def harmonic_com_ssc(kb, b0, T):
    """
    Compute the standard state correction of a harmonic oscillator between two centroids.
    This is derived from the Yank code for RadiallySymmetricRestraints.

    Parameters:
    -----------
    kb: float
        force constant in kJ/mol
    
    b0: float
        ideal length in nanometers
    
    T: temperature

    Returns
    -------
    float
        The analytical restraint correction in kJ/mol.

    """
    kT = kB * (T*unit.kelvin)
    kT = kT.value_in_unit(unit.kilojoules_per_mole)
    restr_vol = harmonic_com_vol(kb, b0, T)
    return -np.log(1.660 / restr_vol)*kT # in kJ/mol

