import time
import numpy as onp
import jax.numpy as np

from timemachine.constants import BOLTZ


class LangevinIntegrator():


    def __init__(self, masses, dt=0.0025, friction=1.0, temperature=300.0):
        """
        Parameters
        ----------
        masses: np.array
            shape [N,] of masses in atomic mass units
    
        dt: float
            timestep in picoseconds

        friction: float
            strength of the friction in 1/picoseconds

        temperature: float
            temperature of the thermostat in Kelvins
            
        """
        self.dt = dt
        self.friction = friction
        self.masses = masses
        self.num_atoms = masses.shape[0]
        self.vscale = np.exp(-dt*friction)

        if friction == 0:
            self.fscale = dt
        else:
            self.fscale = (1-self.vscale)/friction

        kT = BOLTZ * temperature
        self.nscale = np.sqrt(kT*(1-self.vscale*self.vscale)) # noise scale
        self.invMasses = (1.0/masses).reshape((-1, 1))

    def step(self, x_t, v_t, g_t):
        """
        Take an MD step.

        Parameters
        ----------
        x_t: np.array
            [N,3] shape vector of coordinates

        v_t: np.array
            [N,3] shape vector of velocities

        g_t: np.array
            [N,3] shape vector of gradients of the energy w.r.t. coordinates

        Returns
        -------
        tuple of np.array, np.array
            Returns a pair of [N,3], [N,3] coordinates at the next timestep.

        """
        noise = onp.random.normal(size=(self.num_atoms, 3)).astype(x_t.dtype)
        v_t_new = self.vscale*v_t - self.fscale*self.invMasses*g_t + self.nscale*np.sqrt(self.invMasses)*noise
        x_t_new = x_t + v_t_new*self.dt
        return x_t_new, v_t_new
