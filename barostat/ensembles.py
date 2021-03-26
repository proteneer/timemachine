import numpy as np
from simtk import unit
from barostat.utils import compute_box_volume

# TODO: should these be imported from a project-wide constants.py file?
ENERGY_UNIT = unit.kilojoule_per_mole
DISTANCE_UNIT = unit.nanometer


class PotentialEnergyModel:
    def __init__(self, sys_params, unbound_potentials, precision=np.float64):
        self.sys_params = sys_params
        self.unbound_potentials = unbound_potentials
        self.bp_cache = dict()
        self.precision = precision
        self._initialize()

    def _initialize(self):
        for component_params, unbound_pot in zip(self.sys_params, self.unbound_potentials):
            key = unbound_pot

            if key not in self.bp_cache:
                impl = unbound_pot.bind(np.asarray(component_params)).bound_impl(self.precision)
                self.bp_cache[key] = impl

    @property
    def all_impls(self):
        """List of impl, e.g. as required by context constructor"""
        return [self.bp_cache[key] for key in self.unbound_potentials]

    def energy_and_gradient(self, x, box, lam):
        Us, dU_dxs = [], []
        for key in self.unbound_potentials:
            dU_dx, dU_dl, U = self.bp_cache[key].execute(x, box, lam)

            Us.append(U)
            dU_dxs.append(dU_dx)

        return np.sum(Us), np.sum(dU_dxs, 0)


class NVTEnsemble:
    def __init__(self, potential_energy: PotentialEnergyModel, temperature: unit.Quantity):
        self.potential_energy = potential_energy
        self.temperature = temperature
        self.beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * self.temperature)

        # given a value assumed to be in units of ENERGY_UNIT, multiply by this to get unitless reduced potential energy
        self._prefactor = self.beta * ENERGY_UNIT / unit.AVOGADRO_CONSTANT_NA

    def reduced_potential_and_gradient(self, x, box, lam):
        U, dU_dx = self.potential_energy.energy_and_gradient(x, box, lam)

        # reduced potential u
        #   (unitless)
        u = self._prefactor * U

        # d reduced potential / dx
        #   (units of 1 / nm, but returned without units, since (x, box) don't have units either)
        du_dx = self._prefactor * dU_dx

        return u, du_dx


class NPTEnsemble:
    def __init__(self, potential_energy: PotentialEnergyModel, temperature: unit.Quantity, pressure: unit.Quantity):
        self.potential_energy = potential_energy
        self.temperature = temperature
        self.pressure = pressure
        self.beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * self.temperature)

    def reduced_potential_and_gradient(self, x, box, lam):
        U, dU_dx = self.potential_energy.energy_and_gradient(x, box, lam)

        volume = compute_box_volume(box) * DISTANCE_UNIT ** 3

        # reduced potential u
        #   (unitless)
        u = self.beta * ((U * ENERGY_UNIT / unit.AVOGADRO_CONSTANT_NA) + self.pressure * volume)

        # d reduced potential / dx
        #   (units of 1 / nm, but returned without units, since (x, box) don't have units either)
        du_dx = self.beta * ((dU_dx * ENERGY_UNIT / unit.AVOGADRO_CONSTANT_NA) + self.pressure * volume)

        return u, du_dx
