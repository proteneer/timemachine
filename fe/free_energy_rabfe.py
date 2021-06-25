from typing import List
from jax.config import config; config.update("jax_enable_x64", True)

import jax
import numpy as np

from fe import topology

from timemachine.lib import potentials, custom_ops
from timemachine.lib import LangevinIntegrator

from ff.handlers import openmm_deserializer

from rdkit.Chem import MolToSmiles

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

class UnsupportedTopology(Exception):
    pass

class BaseFreeEnergy():

    @staticmethod
    def _get_system_params_and_potentials(ff_params, topology):

        ff_tuples = [
            [topology.parameterize_harmonic_bond, (ff_params[0],)],
            [topology.parameterize_harmonic_angle, (ff_params[1],)],
            [topology.parameterize_periodic_torsion, (ff_params[2], ff_params[3])],
            [topology.parameterize_nonbonded, (ff_params[4], ff_params[5])]
        ]

        final_params = []
        final_potentials = []

        for fn, params in ff_tuples:
            combined_params, combined_potential = fn(*params)
            final_potentials.append(combined_potential)
            final_params.append(combined_params)

        return final_params, final_potentials

# this class is serializable.
class AbsoluteFreeEnergy(BaseFreeEnergy):

    def __init__(self, mol, ff):
        """
        Compute the absolute free energy of a molecule via 4D decoupling.

        Parameters
        ----------
        mol: rdkit mol
            Ligand to be decoupled

        ff: ff.Forcefield
            Ligand forcefield

        """
        self.mol = mol
        self.ff = ff
        self.top = topology.BaseTopology(mol, ff)

    def prepare_host_edge(self, ff_params, host_system):
        """
        Prepares the host-edge system

        Parameters
        ----------
        ff_params: tuple of np.array
            Exactly equal to bond_params, angle_params, proper_params, improper_params, charge_params, lj_params

        host_system: openmm.System
            openmm System object to be deserialized

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses = [a.GetMass() for a in self.mol.GetAtoms()]
        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

        hgt = topology.HostGuestTopology(host_bps, self.top)

        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, hgt)

        combined_masses = np.concatenate([host_masses, ligand_masses])

        return final_potentials, final_params, combined_masses


# this class is serializable.
class RelativeFreeEnergy(BaseFreeEnergy):

    def __init__(self, dual_topology: topology.DualTopology, label=None, complex_path=None):
        self.top = dual_topology
        self.label = label
        self.complex_path = complex_path

    @property
    def mol_a(self):
        return self.top.mol_a

    @property
    def mol_b(self):
        return self.top.mol_b

    @property
    def ff(self):
        return self.top.ff

    def prepare_host_edge(self, ff_params, host_system):
        """
        Prepares the host-edge system
        Parameters
        ----------
        ff_params: tuple of np.array
            Exactly equal to bond_params, angle_params, proper_params, improper_params, charge_params, lj_params
        host_system: openmm.System
            openmm System object to be deserialized

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """

        ligand_masses_a = [a.GetMass() for a in self.mol_a.GetAtoms()]
        ligand_masses_b = [b.GetMass() for b in self.mol_b.GetAtoms()]

        # extract the 0th conformer
        ligand_coords_a = get_romol_conf(self.mol_a)
        ligand_coords_b = get_romol_conf(self.mol_b)

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

        hgt = topology.HostGuestTopology(host_bps, self.top)

        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, hgt)

        combined_masses = np.concatenate([host_masses, ligand_masses_a, ligand_masses_b])

        return final_potentials, final_params, combined_masses


def construct_absolute_lambda_schedule(num_windows):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    manually optimized by YTZ
    """

    A = int(.20 * num_windows)
    B = int(.66 * num_windows)
    D = 1 # need only one window from 0.6 to 1.0
    C = num_windows - A - B - D

    # optimizing the overlap based on eyeballing absolute hydration free energies
    # there's probably some better way to deal with this by inspecting the curvature
    lambda_schedule = np.concatenate([
        np.linspace(0.0,  0.08,  A, endpoint=False),
        np.linspace(0.08,  0.27, B, endpoint=False),
        np.linspace(0.27, 0.41,  C, endpoint=True),
        [1.0],
    ])

    assert len(lambda_schedule) == num_windows

    return lambda_schedule

def construct_relative_lambda_schedule(num_windows):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    manually optimized by YTZ
    """

    A = int(.15 * num_windows)
    B = int(.60 * num_windows)
    C = num_windows - A - B

    # optimizing the overlap based on eyeballing absolute hydration free energies
    # there's probably some better way to deal with this by inspecting the curvature
    lambda_schedule = np.concatenate([
        np.linspace(0.00, 0.08, A, endpoint=False),
        np.linspace(0.08, 0.27, B, endpoint=False),
        np.linspace(0.27, 1.00, C, endpoint=True)
    ])

    assert len(lambda_schedule) == num_windows

    return lambda_schedule
