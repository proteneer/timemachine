import numpy as np

from timemachine.constants import BOLTZ, AVOGADRO
from timemachine.md.barostat.utils import compute_box_volume


class PotentialEnergyModel:
    def __init__(self, sys_params, unbound_potentials, precision=np.float32, guard_threshold=1e6):
        # TODO: refactor to use summed potential...

        self.sys_params = sys_params
        self.unbound_potentials = unbound_potentials
        self.all_impls = []
        self.precision = precision
        self._initialize()

        # potential energies lower than - abs(guard_threshold) will be treated as +inf
        self.guard_threshold = guard_threshold

    def _initialize(self):
        assert len(self.all_impls) == 0
        for component_params, unbound_pot in zip(self.sys_params, self.unbound_potentials):
            impl = unbound_pot.bind(np.asarray(component_params)).bound_impl(self.precision)
            self.all_impls.append(impl)

    def energy_and_gradient(self, x, box, lam):
        Us, dU_dxs = [], []
        for impl in self.all_impls:
            dU_dx, dU_dl, U = impl.execute(x, box, lam)

            Us.append(U)
            dU_dxs.append(dU_dx)

        U_ = np.sum(Us)
        F_ = np.sum(dU_dxs, 0)
        if np.abs(U) > self.guard_threshold:
            return +np.inf, np.nan * F_
        else:
            return U_, F_


class NVTEnsemble:
    def __init__(self, potential_energy: PotentialEnergyModel, temperature: float):
        self.potential_energy = potential_energy
        self.temperature = temperature
        self.beta = 1.0 / (BOLTZ * temperature)

    def reduce(self, U: float):
        """Convert to reduced units

        Parameters
        ----------
        U: float
            in kJ/mol

        Returns
        -------
        u_nvt: float
            reduced units
        """
        return self.beta * U

    def reduced_potential_and_gradient(self, x, box, lam):
        """

        Parameters
        ----------
        x : array
        box : array
        lam : float

        Returns
        -------
        u : float
            unitless
        du_dx : array
            units of 1 / nm
        """
        U, dU_dx = self.potential_energy.energy_and_gradient(x, box, lam)

        # reduced potential u
        u = self.reduce(U)

        # d reduced potential / dx
        du_dx = self.reduce(dU_dx)

        return u, du_dx


class NPTEnsemble:
    def __init__(self, potential_energy: PotentialEnergyModel, temperature: float, pressure: float):
        self.potential_energy = potential_energy
        self.temperature = temperature
        self.pressure = pressure
        self.beta = 1.0 / (BOLTZ * self.temperature)

    def reduce(self, U: float, volume: float):
        """u_npt = beta * (U + pressure * volume * avogadro)

        Parameters
        ----------
        U : float
            kJ/mol
        volume : float
            nm^3

        Returns
        -------
        u_npt : float
            reduced units

        Reference
        ---------
        OpenMMTools thermodynamic states
            https://github.com/choderalab/openmmtools/blob/321b998fc5977a1f8893e4ad5700b1b3aef6101c/openmmtools/states.py#L1904-L1912
        """
        potential_energy = U

        reduced_u = self.beta * potential_energy
        reduced_pv = self.beta * self.pressure * volume * AVOGADRO

        return reduced_u + reduced_pv

    def reduced_potential_and_gradient(self, x, box, lam):
        """

        Parameters
        ----------
        x : array
        box : array
        lam : float

        Returns
        -------
        u : float
            unitless
        du_dx : array
            units of 1 / nm
        """
        U, dU_dx = self.potential_energy.energy_and_gradient(x, box, lam)
        volume = compute_box_volume(box)

        u = self.reduce(U, volume)
        du_dx = self.reduce(dU_dx, volume)

        return u, du_dx
