import numpy as np
from timemachine.integrator import langevin_coefficients

class System():

    def __init__(self, gradients, integrator):
        # fully contained class that allows simulations to be run forward
        # and backward
        # self.x0 = x0
        # self.v0 = v0
        self.gradients = gradients
        self.integrator = integrator


class Integrator():

    def __init__(self, steps, dt, temperature, friction, masses, lambs, seed):

        minimization_steps = 2000

        ca, cbs, ccs = langevin_coefficients(
            temperature,
            dt,
            friction,
            masses
        )

        complete_cas = np.ones(steps)*ca
        complete_dts = np.concatenate([
            np.linspace(0, dt, minimization_steps),
            np.ones(steps-minimization_steps)*dt
        ])

        assert len(complete_cas) == len(complete_dts)
        assert len(complete_dts) == len(self.lambs)

        self.dts = complete_dts
        self.cas = complete_cas
        self.cbs = -cbs
        self.ccs = ccs
        # self.lambs = np.zeros(steps) + lamb
        self.lambs = lambs
        self.seed = seed