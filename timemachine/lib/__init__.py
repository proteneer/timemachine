import numpy as np

from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients

# safe to pickle!

class LangevinIntegrator():

    def __init__(self, temperature, dt, friction, masses, seed):

        self.dt = dt
        self.seed = seed

        ca, _, cc = langevin_coefficients(temperature, dt, friction, masses)

        self.ca = ca
        self.invMasses = 1.0 / masses
        self.ccs = cc

    def impl(self):
        return custom_ops.LangevinIntegrator(self.dt, self.ca, self.invMasses, self.ccs, self.seed)


class MonteCarloBarostat():

    __slots__ = ("N", "temperature", "pressure", "group_idxs", "interval", "seed")

    def __init__(self, N, pressure, temperature, group_idxs, interval, seed):
        self.N = N
        self.pressure = pressure
        self.temperature = temperature
        self.group_idxs = group_idxs
        self.interval = interval
        self.seed = seed

    def impl(self, bound_potentials):
        return custom_ops.MonteCarloBarostat(self.N, self.pressure, self.temperature, self.group_idxs, self.interval, bound_potentials, self.seed)
