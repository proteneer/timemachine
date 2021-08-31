from rdkit import Chem
from collections import namedtuple
from jax.config import config; config.update("jax_enable_x64", True)

import numpy as np

from fe import topology
from fe.utils import get_romol_conf

from ff.handlers import openmm_deserializer

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from dataclasses import dataclass


@dataclass
class RABFEResult():
    mol_name: str
    dG_complex_conversion: float
    dG_complex_decouple: float
    dG_solvent_conversion: float
    dG_solvent_decouple: float

    def log(self):
        """print stage summary"""
        print("stage summary for mol:", self.mol_name,
              "dG_complex_conversion (K complex)", self.dG_complex_conversion,
              "dG_complex_decouple (E0 + A0 + A1 + E1)", self.dG_complex_decouple,
              "dG_solvent_conversion (K solvent)", self.dG_solvent_conversion,
              "dG_solvent_decouple (D)", self.dG_solvent_decouple
              )

    @classmethod
    def from_log(cls, log_line):
        """parse log line"""
        mol_name, rest = log_line.split('stage summary for mol: ')[-1].split(' dG_complex_conversion (K complex) ')
        tokens = rest.split()
        value_strings = tokens[0], tokens[9], tokens[13], tokens[16]
        values = list(map(float, value_strings))
        return RABFEResult(mol_name, *values)

    @property
    def dG_complex(self):
        """effective free energy of removing from complex"""
        return self.dG_complex_conversion + self.dG_complex_decouple

    @property
    def dG_solvent(self):
        """effective free energy of removing from solvent"""
        return self.dG_solvent_conversion + self.dG_solvent_decouple

    @property
    def dG_bind(self):
        """the final value we seek is the free energy of moving
        from the solvent into the complex"""
        return self.dG_solvent - self.dG_complex


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

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

        hgt = topology.HostGuestTopology(host_bps, self.top)

        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, hgt)

        combined_masses = np.concatenate([host_masses, ligand_masses_a, ligand_masses_b])

        return final_potentials, final_params, combined_masses


def construct_conversion_lambda_schedule(num_windows):
    return np.linspace(0, 1, num_windows)


def construct_absolute_lambda_schedule_complex(num_windows):
    """Generate a length-num_windows list of lambda values from 0.0 up to 1.0

    Notes
    -----
    manually optimized by YTZ
    """

    A = int(.20 * num_windows)
    B = int(.50 * num_windows)
    C = num_windows - A - B

    lambda_schedule = np.concatenate([
        np.linspace(0.0, 0.1, A, endpoint=False),
        np.linspace(0.1, 0.3, B, endpoint=False),
        np.linspace(0.3, 1.0, C, endpoint=True)
    ])

    return lambda_schedule


def construct_absolute_lambda_schedule_solvent(num_windows):
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
        np.linspace(0.27, 0.50,  C, endpoint=True),
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


def setup_relative_restraints(
    mol_a,
    mol_b):
    """
    Setup restraints between ring atoms in two molecules.

    """
    # setup relative orientational restraints
    # rough sketch of algorithm:
    # find core atoms in mol_a
    # find core atoms in mol_b
    # use the hungarian algorithm to assign matching

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)

    core_idxs_a = []
    for idx, a in enumerate(mol_a.GetAtoms()):
        if a.IsInRing():
            core_idxs_a.append(idx)

    core_idxs_b = []
    for idx, b in enumerate(mol_b.GetAtoms()):
        if b.IsInRing():
            core_idxs_b.append(idx)

    ri = np.expand_dims(ligand_coords_a[core_idxs_a], 1)
    rj = np.expand_dims(ligand_coords_b[core_idxs_b], 0)
    rij = np.sqrt(np.sum(np.power(ri-rj, 2), axis=-1))

    row_idxs, col_idxs = linear_sum_assignment(rij)

    core_idxs = []

    for core_a, core_b in zip(row_idxs, col_idxs):
        core_idxs.append((
            core_idxs_a[core_a],
            core_idxs_b[core_b]
        ))

    core_idxs = np.array(core_idxs, dtype=np.int32)

    return core_idxs

def setup_relative_restraints_by_distance(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    cutoff: float = 0.1,
    terminal: bool = False):
    """
    Setup restraints between atoms in two molecules using
    a cutoff distance between atoms

    Parameters
    ----------
    mol_a: Chem.Mol
        First molecule

    mol_b: Chem.Mol
        Second molecule

    cutoff: float=0.1
        Distance between atoms to consider as a match

    terminal: bool=false
        Map terminal atoms

    Returns
    -------
    np.array (N, 2)
        Atom mapping between atoms in mol_a to atoms in mol_b.
    """

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)
    core_idxs_a = []
    core_idxs_b = []

    for idx, a in enumerate(mol_a.GetAtoms()):
        if not terminal and a.GetDegree() == 1:
            continue
        for b_idx, b in enumerate(mol_b.GetAtoms()):
            if not terminal and b.GetDegree() == 1:
                continue
            if np.linalg.norm(ligand_coords_a[idx] - ligand_coords_b[b_idx]) < cutoff:
                core_idxs_a.append(idx)
                core_idxs_b.append(b_idx)
    assert len(core_idxs_a) == len(core_idxs_b), "Core sizes were inconsistent"

    rij = cdist(ligand_coords_a[core_idxs_a], ligand_coords_b[core_idxs_b])

    row_idxs, col_idxs = linear_sum_assignment(rij)

    core_idxs = []

    for core_a, core_b in zip(row_idxs, col_idxs):
        core_idxs.append((
            core_idxs_a[core_a],
            core_idxs_b[core_b]
        ))

    core_idxs = np.array(core_idxs, dtype=np.int32)

    return core_idxs

def setup_relative_restraints_using_smarts(
    mol_a,
    mol_b,
    smarts):
    """
    Setup restraints between atoms in two molecules using
    a pre-defined SMARTS pattern.

    Parameters
    ----------
    mol_a: Chem.Mol
        First molecule

    mol_b: Chem.Mol
        Second molecule

    smarts: string
        Smarts pattern defining the common core.

    Returns
    -------
    np.array (N, 2)
        Atom mapping between atoms in mol_a to atoms in mol_b.

    """

    # check to ensure the core is connected
    # technically allow for this but we need to do more validation before
    # we can be fully comfortable
    assert "." not in smarts

    core = Chem.MolFromSmarts(smarts)

    # we want *all* possible combinations.
    limit = 1000
    all_core_idxs_a = np.array(mol_a.GetSubstructMatches(core, uniquify=False, maxMatches=limit))
    all_core_idxs_b = np.array(mol_b.GetSubstructMatches(core, uniquify=False, maxMatches=limit))

    assert len(all_core_idxs_a) < limit
    assert len(all_core_idxs_b) < limit

    best_rmsd = np.inf
    best_core_idxs_a = None
    best_core_idxs_b = None

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)

    # setup relative orientational restraints
    # rough sketch of algorithm:
    # find core atoms in mol_a
    # find core atoms in mol_b
    # for all matches in mol_a
    #    for all matches in mol_b
    #       use the hungarian algorithm to assign matching
    #       if sum is smaller than best, then store.

    for core_idxs_a in all_core_idxs_a:
        for core_idxs_b in all_core_idxs_b:

            ri = np.expand_dims(ligand_coords_a[core_idxs_a], 1)
            rj = np.expand_dims(ligand_coords_b[core_idxs_b], 0)
            rij = np.sqrt(np.sum(np.power(ri-rj, 2), axis=-1))

            row_idxs, col_idxs = linear_sum_assignment(rij)

            rmsd = np.linalg.norm(ligand_coords_a[core_idxs_a[row_idxs]] - ligand_coords_b[core_idxs_b[col_idxs]])

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_core_idxs_a = core_idxs_a
                best_core_idxs_b = core_idxs_b


    core_idxs = np.stack([best_core_idxs_a, best_core_idxs_b], axis=1).astype(np.int32)
    print("core_idxs", core_idxs, "rmsd", best_rmsd)

    return core_idxs
