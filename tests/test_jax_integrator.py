import unittest
import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
import jax

from timemachine import jax_integrator
from timemachine.jax_functionals import jax_bonded


class TestIntegrator(unittest.TestCase):


    def test_langevin_integrator_diatom(self):

        # simple diatom, with an ideal
        # separation of 1A/0.1nm
        x0 = np.array([
            [0.00000, 0.00000,  0.00000],
            [0.10000, 0.00000,  0.00000],
        ], dtype=np.float64)

        # initial velocities are set to zero
        v0 = np.zeros_like(x0)

        # normalized masses
        masses = np.array([1.0, 1.0], dtype=np.float64)

        intg = jax_integrator.LangevinIntegrator(
            masses=masses,
            dt=0.0025,
            friction=1.0,
            temperature=300.0)

        # force constant and ideal bond length
        parameters = np.array([25000.0, 0.1])

        harmonic = jax_bonded.HarmonicBond(
            bond_idxs=np.array([[0,1]]),
            param_idxs=np.array([[0,1]])
        )

        # compute dE/dx
        grad_fn = jax.jit(jax.grad(harmonic.energy, argnums=(0,)))
        # grad_fn = jax.grad(harmonic.energy, argnums=(0,))

        num_steps = 2000

        def simulate(params):
            x_t = x0
            v_t = v0
            for step in range(num_steps):
                print(step)
                g_t = grad_fn(x_t, params)[0]
                x_t, v_t = intg.step(x_t, v_t, g_t)
            return x_t, v_t

        # uncomment if you want to run inference
        x_t_f, v_t_f = simulate(parameters) # stable
        assert np.linalg.norm(x_t_f[0] - x_t_f[1]) < 0.2 # molecule is still stable

        # compute dx_t/dp and dv_t/dp
        param_grad_fn = jax.jacfwd(simulate, argnums=(0,))
        dx_dps, dv_dps = param_grad_fn(parameters) # unstable despite simulation being stable

        # return length-1 array per tf/jax convention
        print(dx_dps[0], dv_dps[0])
         #[[[ 2.49075400e-02  5.64793566e+04]
         #  [-3.57013383e-02 -8.11086777e+04]
         #  [ 3.97305464e-05  6.12038129e+01]]

         # [[-2.49075400e-02 -5.64793566e+04]
         #  [ 3.57013383e-02  8.11086777e+04]
         #  [-3.97305464e-05 -6.12038129e+01]]] [[[-9.41169210e-01 -2.13005209e+06]
         #  [-1.00413166e+00 -2.27679643e+06]
         #  [ 2.06595948e+00  4.68856690e+06]]

         # [[ 9.41169210e-01  2.13005209e+06]
         #  [ 1.00413166e+00  2.27679643e+06]
         #  [-2.06595948e+00 -4.68856690e+06]]]
    
        # both coordinate and velocity derivatives are unstable.
        assert not np.any(dx_dps[0] > 10000) # derivatives w.r.t. force constants are stable, but
        assert not np.any(dv_dps[0] > 10000) # derivatives w.r.t. bond lengths unstable

