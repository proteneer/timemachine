import numpy as np
import functools

from simtk import unit
from fe import standard_state

import scipy.integrate

def test_harmonic_com_ssc():

    T = 300.0
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * T*unit.kelvin
    kT = kT.value_in_unit(unit.kilojoules_per_mole)

    def harmonic_com(r, kb, b0):

        e = kb*(r-b0)**2
        e = e/kT
        dI = 4.0 * np.pi * r**2 * np.exp(-e)
        return dI

    test_kb = 1000.0
    test_b0 = 0.02


    u_fn = functools.partial(harmonic_com, kb=test_kb, b0=test_b0) # kJ/mol, nanometers
    numerical_restr_vol, err = scipy.integrate.quad(u_fn, 0, 10.0) # returns nm^3
    analytical_restr_vol = standard_state.harmonic_com_vol(test_kb, test_b0, T)

    np.testing.assert_allclose(numerical_restr_vol, analytical_restr_vol)
  
    numerical_ssc = -np.log(1.660 / numerical_restr_vol)*kT # in kJ/mol
    analytical_ssc = standard_state.harmonic_com_ssc(test_kb, test_b0, T)

    np.testing.assert_allclose(numerical_ssc, analytical_ssc)
