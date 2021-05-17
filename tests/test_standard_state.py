import numpy as np
import functools

from simtk import unit
from fe import standard_state
import rmsd

import scipy.integrate


from timemachine.potentials import rmsd

def test_translational_restraint():
    k = 25.0
    b = 0.0

    def harmonic_restraint(r):
        return k*(r-b)**2

    beta = 0.67
    Z_numeric = standard_state.integrate_radial_Z(
        harmonic_restraint,
        beta,
        r_max = 10.0
    )

    k = k*beta
    Z_exact = 4.0*np.pi*((b*np.exp(-b**2*k))/(2*k) + ((1 + 2*b**2*k)*np.sqrt(np.pi)*(1 + scipy.special.erf(b*np.sqrt(k))))/(4*k**(3/2)))

    np.testing.assert_almost_equal(Z_numeric, Z_exact)

    dG = standard_state.standard_state_correction(Z_exact, beta)

    assert dG < 0

def test_rotational_restraint():

    k = 25.0
    u_fn = functools.partial(rmsd.angle_u, k=k)
    beta = 0.67
    Z_quat = standard_state.integrate_rotation_Z(u_fn, beta)

    def integrand(phi_1,phi_2,psi):
        delta = psi
        alpha = phi_1
        gamma = phi_2
        cos_theta = np.cos(delta/2)**2*np.cos(gamma+alpha) - np.sin(delta/2)**2
        nrg = rmsd.cos_angle_u(cos_theta, k)
        assert nrg > 0
        # constant = 1/(8*np.pi**2) # normalization constant not needed
        constant = 1/8
        return constant*np.sin(psi)*np.exp(-beta*nrg)

    Z_euler, _ = scipy.integrate.tplquad(
        integrand,
        0, # psi low
        np.pi, # psi high
        lambda x: 0, # phi_1 low
        lambda x: 2*np.pi, # phi_1 high
        lambda x,y: 0, # phi_2 low
        lambda x,y: 2*np.pi # phi_2 high
    )

    np.testing.assert_almost_equal(Z_quat, Z_euler)

def test_release_restraints():
    # test the release of orientational restraints.
    k_t = 50.0
    k_r = 25.0
    beta = 0.67
    dG_t, dG_r = standard_state.release_orientational_restraints(k_t, k_r, beta)

    # these should be negative for sensible force constants
    assert dG_t < 0
    assert dG_r < 0