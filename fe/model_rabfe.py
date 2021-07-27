from abc import ABC

import functools
import numpy as np
import mdtraj

from simtk import openmm
from rdkit import Chem

from timemachine.lib import potentials
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine import constants
from fe import free_energy_rabfe, topology, estimator_abfe, model_utils
from ff import Forcefield

from parallel.client import AbstractClient
from typing import Optional

from md.barostat.utils import get_group_indices, get_bond_list

import pickle

class AbsoluteModel(ABC):

    def __init__(
        self,
        client: AbstractClient or None,
        ff: Forcefield,
        host_system: openmm.System,
        host_schedule: np.ndarray,
        host_topology: openmm.app.Topology,
        temperature: float,
        pressure: float,
        dt: float,
        equil_steps: int,
        prod_steps: int):

        self.host_system = host_system
        self.host_schedule = host_schedule
        self.host_topology = host_topology
        self.temperature = temperature
        self.pressure = pressure
        self.dt = dt

        self.client = client
        self.ff = ff
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps

    def setup_topology(self, mol):
        raise NotImplementedError()

    def predict(self,
        ff_params,
        mol,
        x0,
        box0,
        prefix,
        core_idxs=None):
        """ Compute the absolute free of energy of decoupling mol_a.

        This function is differentiable w.r.t. ff_params.

        Parameters
        ----------

        ff_params: list of np.ndarray
            This should match the ordered params returned by the forcefield

        mol: Chem.Mol
            Molecule we want to decouple

        x0: np.narray
            Initial coordinates of the combined system.

        box0: np.narray
            Initial box vectors of the combined system.

        prefix: str
            String to prepend to print out statements

        core_idxs: None or list of int
            List of core_idxs we may wish to turn off.

        Returns
        -------
        float
            delta G in kJ/mol

        float
            BAR error in the delta G in kJ/mol

        Note that the error estimate is likely to be biased for two reasons: we don't
            know the true decorrelation time, and by re-using intermediate windows
            to compute delta_Us, the BAR estimates themselves become correlated.

        """
        top = self.setup_topology(mol, core_idxs)

        afe = free_energy_rabfe.AbsoluteFreeEnergy(mol, top)

        unbound_potentials, sys_params, masses = afe.prepare_host_edge(
            ff_params,
            self.host_system
        )

        seed = 0

        beta = 1/(constants.BOLTZ*self.temperature)

        bond_list = get_bond_list(unbound_potentials[0])
        masses = model_utils.apply_hmr(masses, bond_list)
        friction = 1.0
        integrator = LangevinIntegrator(
            self.temperature,
            self.dt,
            friction,
            masses,
            seed
        )

        group_indices = get_group_indices(bond_list)
        barostat_interval = 5
        barostat = MonteCarloBarostat(
            x0.shape[0],
            self.pressure,
            self.temperature,
            group_indices,
            barostat_interval,
            seed
        )

        v0 = np.zeros_like(x0)

        endpoint_correct = False
        model = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            self.client,
            box0,
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

        # uncomment if we want to visualize
        combined_topology = model_utils.generate_imaged_topology(
            [self.host_topology, mol],
            x0,
            box0,
            "initial_"+prefix+".pdb"
        )

        for lambda_idx, res in enumerate(results):
            # used for debugging for now, try to reproduce mdtraj error
            outfile = open("pickle_"+prefix+"_lambda_idx_" + str(lambda_idx) + ".pkl", "wb")
            pickle.dump((res.xs, res.boxes, combined_topology), outfile)
            # pickle.dump((res.xs[:100], res.boxes[:100], combined_topology), outfile)
            traj = mdtraj.Trajectory(res.xs, mdtraj.Topology.from_openmm(combined_topology))
            traj.unitcell_vectors = res.boxes
            traj.save_xtc("initial_"+prefix+"_lambda_idx_" + str(lambda_idx) + ".xtc")
    
        return dG, dG_err

        # disabled since image molecules is broken.
        for idx, result in enumerate(results):
            traj = mdtraj.Trajectory(result.xs, mdtraj.Topology.from_openmm(combined_topology))
            unit_cell = np.repeat(self.host_box[None, :], len(result.xs), axis=0)
            traj.unitcell_vectors = unit_cell
            traj.image_molecules()
            traj.save_xtc("complex_lambda_"+str(idx)+".xtc")

        np.savez("results.npz", results=results)

        assert 0

        return dG, results



class RelativeModel(ABC):
    """
    Absolute free energy using a reference molecule to block the binding pocket.
    """

    def __init__(
        self,
        client: Optional[AbstractClient],
        ff: Forcefield,
        host_system: openmm.System,
        host_schedule: np.ndarray,
        host_topology: openmm.app.Topology,
        temperature: float,
        pressure: float,
        dt: float,
        equil_steps: int,
        prod_steps: int):

        self.host_system = host_system
        self.temperature = temperature
        self.pressure = pressure
        self.dt = dt
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
        combined_core_idxs,
        x0,
        box0,
        prefix):

        num_host_atoms = x0.shape[0] - mol_a.GetNumAtoms() - mol_b.GetNumAtoms()

        # (ytz): super ugly, undo combined_core_idxs to get back original idxs
        core_idxs = combined_core_idxs - num_host_atoms
        core_idxs[:, 1] -= mol_a.GetNumAtoms()

        dual_topology = self.setup_topology(mol_a, mol_b, core_idxs)
        rfe = free_energy_rabfe.RelativeFreeEnergy(dual_topology)

        unbound_potentials, sys_params, masses = rfe.prepare_host_edge(
            ff_params,
            self.host_system
        )

        k_core = 30.0
        core_params = np.zeros_like(combined_core_idxs).astype(np.float64)
        core_params[:, 0] = k_core

        B = len(combined_core_idxs)

        restraint_potential = potentials.HarmonicBond(
            combined_core_idxs,
        )

        unbound_potentials.append(restraint_potential)
        sys_params.append(core_params)

        # tbd sample from boltzmann distribution later
        v0 = np.zeros_like(x0)

        seed = 0
        beta = 1/(constants.BOLTZ*self.temperature)

        bond_list = np.concatenate([unbound_potentials[0].get_idxs(), core_idxs])
        masses = model_utils.apply_hmr(masses, bond_list)

        friction = 1.0
        integrator = LangevinIntegrator(
            self.temperature,
            self.dt,
            friction,
            masses,
            seed
        )
        bond_list = list(map(tuple, bond_list))
        group_indices = get_group_indices(bond_list)
        barostat_interval = 5

        barostat = MonteCarloBarostat(
            x0.shape[0],
            self.pressure,
            self.temperature,
            group_indices,
            barostat_interval,
            seed
        )

        endpoint_correct = True
        model = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            self.client,
            box0, # important, use equilibrated box.
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

        # uncomment if we want to visualize.
        combined_topology = model_utils.generate_imaged_topology(
            [self.host_topology, mol_a, mol_b],
            x0,
            box0,
            "initial_"+prefix+".pdb"
        )

        for lambda_idx, res in enumerate(results):
            outfile = open("pickle_"+prefix+"_lambda_idx_" + str(lambda_idx) + ".pkl", "wb")
            pickle.dump((res.xs, res.boxes, combined_topology), outfile)
            traj = mdtraj.Trajectory(res.xs, mdtraj.Topology.from_openmm(combined_topology))
            traj.unitcell_vectors = res.boxes
            traj.save_xtc("initial_"+prefix+"_lambda_idx_" + str(lambda_idx) + ".xtc")

        return dG, dG_err, results

    def predict(self,
        ff_params: list,
        mol_a: Chem.Mol,
        mol_b: Chem.Mol,
        core_idxs: np.array,
        x0: np.array,
        box0: np.array,
        prefix: str):
        """
        Compute the free of energy of converting mol_a into mol_b. The starting state
        has mol_a fully interacting with the environment, mol_b is non-interacting.
        The end state has mol_b fully interacting with the environment, and mol_a is
        non-interacting. The atom mapping defining the core need be neither
        bijective nor factorizable.

        This function is differentiable w.r.t. ff_params.

        Parameters
        ----------

        ff_params: list of np.ndarray
            This should match the ordered params returned by the forcefield

        mol_a: Chem.Mol
            Starting molecule

        mol_b: Chem.Mol
            Resulting molecule

        core_idxs: np.array (Nx2), dtype int32
            Atom mapping defining the core, mapping atoms from mol_a to atoms in mol_b.

        prefix: str
            Auxiliary string to prepend print-outs

        x0: np.ndarray
            Initial coordinates of the combined system.

        box0: np.ndarray
            Initial box vectors.

        Returns
        -------
        float
            delta delta G in kJ/mol of morphing mol_a into mol_b

        float
            BAR error in the delta delta G in kJ/mol

        Note that the error estimate is likely to be biased for two reasons: we don't
            know the true decorrelation time, and by re-using intermediate windows
            to compute delta_Us, the BAR estimates themselves become correlated.

        """
        num_host_atoms = x0.shape[0] - mol_a.GetNumAtoms() - mol_b.GetNumAtoms()
        host_coords = x0[:num_host_atoms]
        mol_a_coords = x0[num_host_atoms:num_host_atoms+mol_a.GetNumAtoms()]
        mol_b_coords = x0[num_host_atoms+mol_a.GetNumAtoms():]

        # pull out mol_b from combined state
        combined_core_idxs = np.copy(core_idxs)
        combined_core_idxs[:, 0] += num_host_atoms
        combined_core_idxs[:, 1] += num_host_atoms + mol_a.GetNumAtoms()

        # this is redundant, but thought it best to be explicit about ordering here..
        combined_coords = np.concatenate([
            host_coords,
            mol_a_coords,
            mol_b_coords
        ])
        dG_0, dG_0_err, results_0 = self._predict_a_to_b(
            ff_params,
            mol_a,
            mol_b,
            combined_core_idxs,
            combined_coords,
            box0,
            prefix+"_ref_to_mol")

        return dG_0, dG_0_err

        # pull out mol_a from combined state
        combined_core_idxs = np.copy(core_idxs)
        # swap the ligand coordinates in the reverse direction
        combined_core_idxs[:, 0] = core_idxs[:, 1]
        combined_core_idxs[:, 1] = core_idxs[:, 0]
        combined_core_idxs[:, 0] += num_host_atoms
        combined_core_idxs[:, 1] += num_host_atoms + mol_b.GetNumAtoms()
        combined_coords = np.concatenate([
            host_coords,
            mol_b_coords,
            mol_a_coords
        ])

        dG_1, dG_1_err, results_1 = self._predict_a_to_b(
            ff_params,
            mol_b,
            mol_a,
            combined_core_idxs,
            combined_coords,
            box0,
            prefix+"_mol_to_ref")

        # dG_0 is the free energy of moving X-A-B into X-A+B
        # dG_1 is the free energy of moving X-B-A into X-B+A
        # -dG_0 + dG_1 is the free energy of moving X-A+B -> X-B+A
        # i.e. the free energy of "unbinding" A

        dG_err = np.sqrt(dG_0_err**2 + dG_1_err**2)

        return -dG_0 + dG_1, dG_err

# subclasses specific for each model

class AbsoluteHydrationModel(AbsoluteModel):

    def setup_topology(self, mol, _):
        return topology.BaseTopologyRHFE(mol, self.ff)

class RelativeHydrationModel(RelativeModel):

    def setup_topology(self, mol_a, mol_b, _):
        return topology.DualTopologyRHFE(mol_a, mol_b, self.ff)

class AbsoluteConversionModel(AbsoluteModel):

    def setup_topology(self, mol, core_idxs):
        top = topology.BaseTopologyConversion(mol, self.ff)
        # top.parameterize_proper_torsion = functools.partial(
        #     top.parameterize_proper_torsion,
        #     core_idxs=core_idxs
        # )
        return top

class AbsoluteStandardHydrationModel(AbsoluteModel):

    def setup_topology(self, mol, core_idxs):
        top = topology.BaseTopologyStandardDecoupling(mol, self.ff)
        # top.parameterize_proper_torsion = functools.partial(
        #     top.parameterize_proper_torsion,
        #     core_idxs=core_idxs
        # )
        return top

class RelativeBindingModel(RelativeModel):

    def setup_topology(self, mol_a, mol_b, core_idxs):
        top = topology.DualTopologyStandardDecoupling(mol_a, mol_b, self.ff)
        # top.parameterize_proper_torsion = functools.partial(
        #     top.parameterize_proper_torsion,
        #     core_idxs_a=core_idxs[:, 0],
        #     core_idxs_b=core_idxs[:, 1]
        # )
        return top
