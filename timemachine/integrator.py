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
        collision rate in 1 / picoseconds

    masses: array
        mass of each atom in standard mass units

    Returns
    -------
    tuple (ca, cb, cc)
        ca is scalar, and cb and cc are n length arrays
        that are used during langevin dynamics as follows:

        during heat-bath update
        v -> ca * v + cc * gaussian

        during force update
        v -> v + cb * force
    """
    kT = BOLTZ * temperature
    nscale = np.sqrt(kT / masses)

    ca = np.exp(-friction * dt)
    cb = dt / masses
    cc = np.sqrt(1 - np.exp(-2 * friction * dt)) * nscale

    return ca, cb, cc


class Integrator:

    def step(self, x, v):
        """Return copies x and v, updated by a single timestep"""
        raise NotImplementedError

    def multiple_steps(self, x, v, n_steps=1):
        """Return trajectories of x and v, advanced by n_steps"""
        xs, vs = [x], [v]

        for _ in range(n_steps):
            new_x, new_v = self.step(xs[-1], vs[-1])

            xs.append(new_x)
            vs.append(new_v)

        return np.array(xs), np.array(vs)


class LangevinIntegrator(Integrator):
    def __init__(self, force_fxn, masses, temperature, dt, friction):
        """BAOAB (https://arxiv.org/abs/1203.5428), rotated by half a timestep"""
        self.dt = dt
        self.masses = masses
        ca, cb, cc = langevin_coefficients(temperature, dt, friction, masses)
        self.force_fxn = force_fxn

        # make masses, frictions, etc. (scalar or (N,)) shape-compatible with coordinates (vector or (N,3))
        self.ca, self.cb, self.cc = np.expand_dims(ca, -1), np.expand_dims(cb, -1), np.expand_dims(cc, -1)

    def step(self, x, v):
        """Intended to match https://github.com/proteneer/timemachine/blob/37e60205b3ae3358d9bb0967d03278ed184b8976/timemachine/cpp/src/integrator.cu#L71-L74"""
        v_mid = v + self.cb * self.force_fxn(x)

        new_v = (self.ca * v_mid) + (self.cc * np.random.randn(*x.shape))
        new_x = x + 0.5 * self.dt * (v_mid + new_v)

        return new_x, new_v
