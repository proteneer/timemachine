import numpy as np

from jankmachine.lib import custom_ops
from jankmachine.integrator import langevin_coefficients

# safe to pickle!

class LangevinIntegrator():

    def __init__(self, temperature, dt, friction, masses, seed):

        self.dt = dt
        self.seed = seed

        ca, cb, cc = langevin_coefficients(temperature, dt, friction, masses)
        cb *= -1
        self.ca = ca
        self.cbs = cb
        self.ccs = cc

    def impl(self):
        return custom_ops.LangevinIntegrator(self.dt, self.ca, self.cbs, self.ccs, self.seed)