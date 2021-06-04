import numpy as np
from jax import numpy as jnp
from simtk import unit
from md.barostat.utils import compute_box_volume
from timemachine.constants import kB, ENERGY_UNIT, DISTANCE_UNIT

from typing import Union

non_unitted = Union[float, np.ndarray, jnp.ndarray]  # raw value without simtk unit attached


class PotentialEnergyModel:
    def __init__(self, sys_params, unbound_potentials, precision=np.float64, guard_threshold=1e6):
        self.sys_params = sys_params
        self.unbound_potentials = unbound_potentials
        self.bp_cache = dict()
        self.precision = precision
        self._initialize()

        # potential energies lower than - abs(guard_threshold) will be treated as +inf
        self.guard_threshold = guard_threshold

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

        U_ = np.sum(Us)
        F_ = np.sum(dU_dxs, 0)
        if np.abs(U) > self.guard_threshold:
            return + np.inf, np.nan * F_
        else:
            return U_, F_


class NVTEnsemble:
    def __init__(self, potential_energy: PotentialEnergyModel, temperature: unit.Quantity):
        self.potential_energy = potential_energy
        self.temperature = temperature
        self.beta = 1.0 / (kB * self.temperature)

        # given a value assumed to be in units of ENERGY_UNIT, multiply by this to get unitless reduced potential energy
        self._prefactor = self.beta * ENERGY_UNIT

    def reduce(self, U: non_unitted):
        """u_nvt = beta * U

        U assumed to be in units of ENERGY_UNIT (kJ/mol), but without simtk unit attached"""
        return self._prefactor * U

    def reduced_potential_and_gradient(self, x, box, lam):
        U, dU_dx = self.potential_energy.energy_and_gradient(x, box, lam)

        # reduced potential u
        #   (unitless)
        u = self.reduce(U)

        # d reduced potential / dx
        #   (units of 1 / nm, but returned without units, since (x, box) don't have units either)
        du_dx = self.reduce(dU_dx)

        return u, du_dx


class NPTEnsemble:
    def __init__(self, potential_energy: PotentialEnergyModel, temperature: unit.Quantity, pressure: unit.Quantity):
        self.potential_energy = potential_energy
        self.temperature = temperature
        self.pressure = pressure
        self.beta = 1.0 / (kB * self.temperature)

    def reduce(self, U: non_unitted, volume: non_unitted):
        """u_npt = beta * (U + pressure * volume)

        U assumed to be in units of ENERGY_UNIT (kJ/mol), but without simtk unit attached
        volume assumed to be in units of DISTANCE_UNIT^3 (nm^3), but without simtk unit attached

        Reference
        ---------
        OpenMMTools thermodynamic states
            https://github.com/choderalab/openmmtools/blob/321b998fc5977a1f8893e4ad5700b1b3aef6101c/openmmtools/states.py#L1904-L1912
        """
        potential_energy = U * ENERGY_UNIT
        volume = volume * DISTANCE_UNIT**3

        reduced_u = self.beta * potential_energy
        reduced_pv = self.beta * self.pressure * volume * unit.AVOGADRO_CONSTANT_NA

        return reduced_u + reduced_pv


    def reduced_potential_and_gradient(self, x, box, lam):
        U, dU_dx = self.potential_energy.energy_and_gradient(x, box, lam)
        volume = compute_box_volume(box)

        # reduced potential u
        #   (unitless)
        u = self.reduce(U, volume)

        # d reduced potential / dx
        #   (units of 1 / nm, but returned without units, since (x, box) don't have units either)
        du_dx = self.reduce(dU_dx, volume)

        return u, du_dx
