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


class RBFETransformIndex:
    """Builds an index of relative free energy transformations to use
    with construct_mle_layer
    """

    def __len__(self):
        return len(self._indices)

    def __init__(self):
        self._indices = {}

    def build(self, refs: List[RelativeFreeEnergy]):
        for ref in refs:
            self.get_transform_indices(ref)

    def get_transform_indices(self, ref: RelativeFreeEnergy) -> List[int]:
        return self.get_mol_idx(ref.mol_a), self.get_mol_idx(ref.mol_b)

    def get_mol_idx(self, mol):
        hashed = self._mol_hash(mol)
        if hashed not in self._indices:
            self._indices[hashed] = len(self._indices)
        return self._indices[hashed]

    def _mol_hash(self, mol):
        return MolToSmiles(mol)


def construct_absolute_lambda_schedule(num_windows):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    manually optimized by YTZ
    """

    A = int(.2 * num_windows)
    B = int(.6 * num_windows)
    C = num_windows - A - B

    # Empirically, we see the largest variance in std <du/dl> near the endpoints in the nonbonded
    # terms. Bonded terms are roughly linear. So we add more lambda windows at the endpoint to
    # help improve convergence.
    lambda_schedule = np.concatenate([
        np.linspace(0.0, 0.1, A, endpoint=False),
        np.linspace(0.1, 0.35, B, endpoint=False),
        np.linspace(0.35, 1.0, C, endpoint=True)
    ])

    assert len(lambda_schedule) == num_windows

    return lambda_schedule


# def construct_absolute_lambda_schedule(num_windows):
#     A = int(0.70 * num_windows)
#     B = num_windows - A

#     lambda_schedule = np.concatenate([
#         np.linspace(0.0, 0.3, A, endpoint=False),
#         np.linspace(0.3, 1.0, B, endpoint=True)
#     ])

#     # lambda_schedule = np.linspace(0, 1, num_windows)

#     return lambda_schedule

