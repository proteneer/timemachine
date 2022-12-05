from dataclasses import dataclass
from typing import List

import numpy as np

from timemachine.fe import topology
from timemachine.fe.utils import get_mol_masses, get_romol_conf
from timemachine.ff import ForcefieldParams
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.lib.potentials import CustomOpWrapper


class HostConfig:
    def __init__(self, omm_system, conf, box):
        self.omm_system = omm_system
        self.conf = conf
        self.box = box


@dataclass
class SimulationProtocol:
    n_frames: int
    n_eq_steps: int
    steps_per_frame: int


@dataclass
class InitialState:
    """
    An initial contains everything that is needed to bitwise reproduce a trajectory given a SimulationProtocol

    This object can be pickled safely.
    """

    potentials: List[CustomOpWrapper]
    integrator: LangevinIntegrator
    barostat: MonteCarloBarostat
    x0: np.ndarray
    v0: np.ndarray
    box0: np.ndarray
    lamb: float
    ligand_idxs: np.ndarray


@dataclass
class SimulationResult:
    all_dGs: List[np.ndarray]
    all_errs: List[float]
    dG_errs_by_lambda_by_component: np.ndarray  # (len(U_names), L - 1)
    overlaps_by_lambda: np.ndarray  # (L - 1,)
    overlaps_by_lambda_by_component: np.ndarray  # (len(U_names), L - 1)
    dG_errs_png: bytes
    overlap_summary_png: bytes
    overlap_detail_png: bytes
    frames: List[np.ndarray]
    boxes: List[np.ndarray]
    initial_states: List[InitialState]
    protocol: SimulationProtocol


class BaseFreeEnergy:
    @staticmethod
    def _get_system_params_and_potentials(ff_params: ForcefieldParams, topology, lamb: float):
        params_potential_pairs = [
            topology.parameterize_harmonic_bond(ff_params.hb_params),
            topology.parameterize_harmonic_angle(ff_params.ha_params),
            topology.parameterize_periodic_torsion(ff_params.pt_params, ff_params.it_params),
            topology.parameterize_nonbonded(ff_params.q_params, ff_params.lj_params, lamb),
        ]

        params, potentials = zip(*params_potential_pairs)
        return params, potentials


# this class is serializable.
class AbsoluteFreeEnergy(BaseFreeEnergy):
    def __init__(self, mol, top):
        """
        Compute the absolute free energy of a molecule via 4D decoupling.

        Parameters
        ----------
        mol: rdkit mol
            Ligand to be decoupled

        top: Topology
            topology.Topology to use

        """
        self.mol = mol
        self.top = top

    def prepare_host_edge(self, ff_params: ForcefieldParams, host_system, lamb: float):
        """
        Prepares the host-guest system

        Parameters
        ----------
        ff_params: ForcefieldParams
            forcefield parameters

        host_system: openmm.System
            openmm System object to be deserialized.

        lamb: float
            alchemical parameter controlling 4D decoupling

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses = get_mol_masses(self.mol)

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)
        hgt = topology.HostGuestTopology(host_bps, self.top)

        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, hgt, lamb)
        combined_masses = self._combine(ligand_masses, host_masses)
        return final_potentials, final_params, combined_masses

    def prepare_vacuum_edge(self, ff_params: ForcefieldParams):
        """
        Prepares the vacuum system

        Parameters
        ----------
        ff_params: ForcefieldParams
            forcefield parameters

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses = get_mol_masses(self.mol)
        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, self.top, 0.0)
        return final_potentials, final_params, ligand_masses

    def prepare_combined_coords(self, host_coords=None):
        """
        Returns the combined coordinates.

        Parameters
        ----------
        host_coords: np.array
            Nx3 array of atomic coordinates
            If None, return just the ligand coordinates.

        Returns
        -------
            combined_coordinates
        """
        ligand_coords = get_romol_conf(self.mol)
        return self._combine(ligand_coords, host_coords)

    def _combine(self, ligand_values, host_values=None):
        """
        Combine the values along the 0th axis.
        The host values will be first, if given.
        Then ligand values.

        Parameters
        ----------
        ligand_values: np.array
        host_values: Optional[np.array]

        Returns
        -------
            combined_values
        """
        if host_values is None:
            return ligand_values
        return np.concatenate([host_values, ligand_values])
