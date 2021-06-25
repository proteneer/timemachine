
from abc import ABC
import os
import pickle

import numpy as np
import jax.numpy as jnp
import functools
import tempfile
import mdtraj

from simtk import openmm
from simtk.openmm import app
from rdkit import Chem

from md import minimizer
from timemachine.lib import potentials
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine import constants
from timemachine.potentials import rmsd
from fe import free_energy_rabfe, topology, estimator_abfe, model_utils
from ff import Forcefield

from parallel.client import AbstractClient
from typing import Optional
from functools import partial
from scipy.optimize import linear_sum_assignment

from md.barostat.utils import get_group_indices, get_bond_list

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

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


def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

class AbsoluteModel(ABC):

    def __init__(
        self,
        client: AbstractClient or None,
        ff: Forcefield,
        host_system: openmm.System,
        host_coords: np.ndarray,
        host_box: np.ndarray,
        host_schedule: np.ndarray,
        host_topology: np.ndarray,
        equil_steps: int,
        prod_steps: int):

        self.host_system = host_system
        self.host_coords = host_coords
        self.host_box = host_box
        self.host_schedule = host_schedule
        self.host_topology = host_topology

        self.client = client
        self.ff = ff
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps

    def setup_topology(self, mol):
        raise NotImplementedError()

    def predict(self,
        ff_params,
        mol,
        prefix):
        """ Compute the absolute free of energy of decoupling mol_a.

        This function is differentiable w.r.t. ff_params.

        Parameters
        ----------

        ff_params: list of np.ndarray
            This should match the ordered params returned by the forcefield

        mol: Chem.Mol
            Molecule we want to decouple

        Returns
        -------
        float
            delta G in kJ/mol

        float
            BAR error in the delta G in kJ/mol

        """

        print(f"Minimizing the host structure to remove clashes.")
        minimized_host_coords = minimizer.minimize_host_4d(
            [mol],
            self.host_system,
            self.host_coords,
            self.ff,
            self.host_box
        )

        top = self.setup_topology(mol)

        afe = free_energy_rabfe.AbsoluteFreeEnergy(mol, self.ff)

        unbound_potentials, sys_params, masses = afe.prepare_host_edge(
            ff_params,
            self.host_system
        )


        ligand_coords = get_romol_conf(mol)
        combined_coords = np.concatenate([minimized_host_coords, ligand_coords])

        endpoint_correct = False

        seed = 0

        temperature = 300.0
        beta = 1/(constants.BOLTZ*temperature)

        bond_list = get_bond_list(unbound_potentials[0])
        masses = model_utils.apply_hmr(masses, bond_list)

        integrator = LangevinIntegrator(
            temperature,
            2.5e-3,
            1.0,
            masses,
            seed
        )

        group_indices = get_group_indices(bond_list)
        barostat_interval = 5
        barostat = MonteCarloBarostat(
            combined_coords.shape[0],
            1.0,
            temperature,
            group_indices,
            barostat_interval,
            seed
        )

        x0 = combined_coords
        v0 = np.zeros_like(combined_coords)

        model = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            self.client,
            self.host_box,
            x0,
            v0,
            integrator,
            barostat,
            self.host_schedule,
            self.equil_steps,
            self.prod_steps,
            beta,
            prefix
        )

        dG, dG_err, results = estimator_abfe.deltaG(model, sys_params)

        return dG, dG_err

        for idx, result in enumerate(results):
            # print(result.xs.shape)
            traj = mdtraj.Trajectory(result.xs, mdtraj.Topology.from_openmm(combined_topology))
            unit_cell = np.repeat(self.host_box[None, :], len(result.xs), axis=0)
            traj.unitcell_vectors = unit_cell
            traj.image_molecules()
            traj.save_xtc("complex_lambda_"+str(idx)+".xtc")

        np.savez("results.npz", results=results)

        assert 0

        return dG, results

class AbsoluteHydrationModel(AbsoluteModel):

    def setup_topology(self, mol):
        return topology.BaseTopology(mol, self.ff)


class RelativeModel(ABC):
    """
    Absolute free energy using a reference molecule to block the binding pocket.
    """

    def __init__(
        self,
        client: Optional[AbstractClient],
        ff: Forcefield,
        host_system: openmm.System,
        host_coords: np.ndarray,
        host_box: np.ndarray,
        host_schedule: np.ndarray,
        host_topology,
        equil_steps: int,
        prod_steps: int):

        self.host_system = host_system
        self.host_coords = host_coords
        self.host_box = host_box
        self.host_schedule = host_schedule
        self.host_topology = host_topology
        self.client = client
        self.ff = ff
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps

    def setup_topology(self, mol_a, mol_b):
        raise NotImplementedError()

    def _predict_a_to_b(
        self,
        ff_params,
        mol_a,
        mol_b,
        core_idxs,
        combined_coords,
        prefix):

        dual_topology = self.setup_topology(mol_a, mol_b)
        rfe = free_energy_rabfe.RelativeFreeEnergy(dual_topology)

        unbound_potentials, sys_params, masses = rfe.prepare_host_edge(
            ff_params,
            self.host_system
        )

        # setup restraints and align to the blocker
        num_host_atoms = len(self.host_coords)
        combined_topology = model_utils.generate_openmm_topology(
            [self.host_topology, mol_a, mol_b],
            self.host_coords,
            prefix+".pdb"
        )

        # generate initial structure
        coords = combined_coords

        traj = mdtraj.Trajectory([coords], mdtraj.Topology.from_openmm(combined_topology))
        traj.save_xtc("initial_coords_aligned.xtc")

        k_core = 75.0
        core_params = np.zeros_like(core_idxs).astype(np.float64)
        core_params[:, 0] = k_core

        B = len(core_idxs)

        restraint_potential = potentials.HarmonicBond(
            core_idxs,
        )

        unbound_potentials.append(restraint_potential)
        sys_params.append(core_params)

        endpoint_correct = True

        # tbd sample from boltzmann distribution later
        x0 = coords
        v0 = np.zeros_like(coords)

        seed = 0
        temperature = 300.0
        beta = 1/(constants.BOLTZ*temperature)

        bond_list = np.concatenate([unbound_potentials[0].get_idxs(), core_idxs])
        masses = model_utils.apply_hmr(masses, bond_list)

        integrator = LangevinIntegrator(
            temperature,
            2.5e-3,
            1.0,
            masses,
            seed
        )
        bond_list = list(map(tuple, bond_list))
        group_indices = get_group_indices(bond_list)
        barostat_interval = 5

        barostat = MonteCarloBarostat(
            coords.shape[0],
            1.0,
            temperature,
            group_indices,
            barostat_interval,
            seed
        )

        endpoint_correct = True
        model = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            self.client,
            self.host_box, # important, use equilibrated box.
            x0,
            v0,
            integrator,
            barostat,
            self.host_schedule,
            self.equil_steps,
            self.prod_steps,
            beta,
            prefix
        )

        dG, dG_err, results = estimator_abfe.deltaG(model, sys_params)

        # disable this for now since image_molecules() is unstable.
        # for idx, result in enumerate(results):
        #     traj = mdtraj.Trajectory(result.xs, mdtraj.Topology.from_openmm(combined_topology))
        #     traj.unitcell_vectors = result.boxes
        #     traj.image_molecules()
        #     traj.save_xtc(prefix+"_complex_lambda_"+str(idx)+".xtc")

        return dG, dG_err, results

    def predict(self, ff_params: list, mol_a: Chem.Mol, mol_b: Chem.Mol, prefix: str):
        """
        Compute the free of energy of converting mol_a into mol_b.

        This function is differentiable w.r.t. ff_params.

        Parameters
        ----------

        ff_params: list of np.ndarray
            This should match the ordered params returned by the forcefield

        mol_a: Chem.Mol
            Starting molecule

        mol_b: Chem.Mol
            Resulting molecule

        prefix: str
            Auxiliary string to prepend print-outs

        Returns
        -------
        float
            delta delta G in kJ/mol of morphing mol_a into mol_b

        float
            BAR error in the delta delta G in kJ/mol

        """

        host_system = self.host_system
        host_lambda_schedule = self.host_schedule

        minimized_host_coords = minimizer.minimize_host_4d(
            [mol_a, mol_b],
            self.host_system,
            self.host_coords,
            self.ff,
            self.host_box
        )

        num_host_atoms = self.host_coords.shape[0]

        # generate indices
        core_idxs = setup_relative_restraints(mol_a, mol_b)

        mol_a_coords = get_romol_conf(mol_a)
        mol_b_coords = get_romol_conf(mol_b)

        # pull out mol_b from combined state
        combined_core_idxs = np.copy(core_idxs)
        combined_core_idxs[:, 0] += num_host_atoms
        combined_core_idxs[:, 1] += num_host_atoms + mol_a.GetNumAtoms()
        combined_coords = np.concatenate([
            minimized_host_coords,
            mol_a_coords,
            mol_b_coords
        ])
        dG_0, dG_0_err, results_0 = self._predict_a_to_b(
            ff_params,
            mol_a,
            mol_b,
            combined_core_idxs,
            combined_coords,
            prefix+"_ref_to_mol")

        # pull out mol_a from combined state
        combined_core_idxs = np.copy(core_idxs)
        # swap
        combined_core_idxs[:, 0] = core_idxs[:, 1]
        combined_core_idxs[:, 1] = core_idxs[:, 0]
        combined_core_idxs[:, 0] += num_host_atoms
        combined_core_idxs[:, 1] += num_host_atoms + mol_b.GetNumAtoms()
        combined_coords = np.concatenate([
            minimized_host_coords,
            mol_b_coords,
            mol_a_coords
        ])
        dG_1, dG_1_err, results_1 = self._predict_a_to_b(
            ff_params,
            mol_b,
            mol_a,
            combined_core_idxs,
            combined_coords,
            prefix+"_mol_to_ref")

        # dG_0 is the free energy of moving X-B-A into X-B+A
        # dG_1 is the free energy of moving X-A-B into X-A+B
        # -dG_1 + dG_0 is the free energy of moving X-A+B -> X-B+A
        # i.e. the free energy of "unbinding" A

        dG_err = np.sqrt(dG_0_err**2 + dG_1_err**2)

        return -dG_0 + dG_1, dG_err


class RelativeHydrationModel(RelativeModel):

    def setup_topology(self, mol_a, mol_b):
        return topology.DualTopologyRHFE(mol_a, mol_b, self.ff)
