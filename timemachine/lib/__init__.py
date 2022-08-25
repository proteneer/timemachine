from timemachine.integrator import langevin_coefficients
from timemachine.lib import custom_ops

# safe to pickle!


class LangevinIntegrator:
    def __init__(self, temperature, dt, friction, masses, seed):

        self.dt = dt
        self.friction = friction
        self.masses = masses
        self.seed = seed
        self.temperature = temperature

        ca, cbs, ccs = langevin_coefficients(temperature, dt, friction, masses)

    def impl(self):
        return custom_ops.LangevinIntegrator(self.masses, self.temperature, self.dt, self.friction, self.seed)


class VelocityVerletIntegrator:
    def __init__(self, dt, masses):
        self.dt = dt

        cb = dt / masses
        cb *= -1
        self.cbs = cb

    def impl(self):
        return custom_ops.VelocityVerletIntegrator(self.dt, self.cbs)


class MonteCarloBarostat:

    __slots__ = ("N", "temperature", "pressure", "group_idxs", "interval", "seed")

    def __init__(self, N, pressure, temperature, group_idxs, interval, seed):
        self.N = N
        self.pressure = pressure
        self.temperature = temperature
        self.group_idxs = group_idxs
        self.interval = interval
        self.seed = seed

    def impl(self, bound_potentials):
        return custom_ops.MonteCarloBarostat(
            self.N, self.pressure, self.temperature, self.group_idxs, self.interval, bound_potentials, self.seed
        )
