import os
from pickle import dump, load
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from rdkit import Chem
from simtk import openmm

from timemachine.fe import estimator, free_energy, topology
from timemachine.fe.model_utils import apply_hmr
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md import minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.parallel.client import AbstractClient, SerialClient


class RBFEModel:
    def __init__(
        self,
        client: Optional[AbstractClient],
        ff: Forcefield,
        complex_system: openmm.System,
        complex_coords: np.ndarray,
        complex_box: np.ndarray,
        complex_schedule: np.ndarray,
        solvent_system: openmm.System,
        solvent_coords: np.ndarray,
        solvent_box: np.ndarray,
        solvent_schedule: np.ndarray,
        equil_steps: int,
        prod_steps: int,
        barostat_interval: int = 25,
        pre_equilibrate: bool = False,
        hmr: bool = False,
    ):

        self.complex_system = complex_system
        self.complex_coords = complex_coords
        self.complex_box = complex_box
        self.complex_schedule = complex_schedule
        self.solvent_system = solvent_system
        self.solvent_coords = solvent_coords
        self.solvent_box = solvent_box
        self.solvent_schedule = solvent_schedule
        self.client = client
        if client is None:
            self.client = SerialClient()
        self.ff = ff
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        self.barostat_interval = barostat_interval
        self.pre_equilibrate = pre_equilibrate
        self.hmr = hmr
        self._equil_cache = {}

    def _edge_hash(self, stage: str, mol_a: Chem.Mol, mol_b: Chem.Mol, core: np.ndarray) -> str:
        a = Chem.MolToSmiles(mol_a)
        b = Chem.MolToSmiles(mol_b)
        # Perhaps bad idea to have ordering of a and b?
        return f"{stage}::{a}->{b}::{core.size}"

    def equilibrate_edges(
        self,
        edges: List[Tuple[Chem.Mol, Chem.Mol, np.ndarray]],
        lamb: float = 0.0,
        barostat_interval: int = 10,
        equilibration_steps: int = 100000,
        cache_path: str = "equilibration_cache.pkl",
    ):
        """
        edges: List of tuples with mol_a, mol_b, core
            Edges to equilibrate

        lamb: float
            Lambda value to equilibrate at. Uses Dual Topology to equilibrate

        barostat_interval: int
            Interval on which to run barostat during equilibration

        equilibration_steps: int
            Number of steps to equilibrate the edge for

        cache_path: string
            Path to look for existing cache or path to where to save cache. By default
            it will write out a pickle file in the local directory.

        Pre equilibrate edges and cache them for later use in predictions.

        Parallelized via the model client if possible
        """
        if not self.pre_equilibrate:
            return
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as ifs:
                self._equil_cache = load(ifs)
            print("Loaded Pre-equilibrated structures from cache")
            return
        futures = []
        ordered_params = self.ff.get_ordered_params()

        temperature = 300.0
        pressure = 1.0

        for stage, host_system, host_coords, host_box in [
            ("complex", self.complex_system, self.complex_coords, self.complex_box),
            ("solvent", self.solvent_system, self.solvent_coords, self.solvent_box),
        ]:
            # Run all complex legs first then solvent, as they will likely take longer than then solvent leg
            for mol_a, mol_b, core in edges:
                # Use DualTopology to ensure mols exist in the same space.
                topo = topology.DualTopologyMinimization(mol_a, mol_b, self.ff)
                rfe = free_energy.RelativeFreeEnergy(topo)
                min_coords = minimizer.minimize_host_4d([mol_a, mol_b], host_system, host_coords, self.ff, host_box)
                unbound_potentials, sys_params, masses, coords = rfe.prepare_host_edge(
                    ordered_params, host_system, min_coords
                )
                # num_host_coords = len(host_coords)
                # masses[num_host_coords:] *= 1000000 # Lets see if masses are the difference
                harmonic_bond_potential = unbound_potentials[0]
                bond_list = get_bond_list(harmonic_bond_potential)
                group_idxs = get_group_indices(bond_list)
                time_step = 1.5e-3
                if self.hmr:
                    masses = apply_hmr(masses, bond_list)
                    time_step = 2.5e-3
                integrator = LangevinIntegrator(temperature, time_step, 1.0, masses, 0)
                barostat = MonteCarloBarostat(coords.shape[0], pressure, temperature, group_idxs, barostat_interval, 0)
                pots = []
                for bp, params in zip(unbound_potentials, sys_params):
                    pots.append(bp.bind(np.asarray(params)))
                future = self.client.submit(
                    estimator.equilibrate, *[integrator, barostat, pots, coords, host_box, lamb, equilibration_steps]
                )
                futures.append((stage, (mol_a, mol_b, core), future))
        num_equil = len(futures)
        for i, (stage, edge, future) in enumerate(futures):
            edge_hash = self._edge_hash(stage, *edge)
            self._equil_cache[edge_hash] = future.result()
            if (i + 1) % 5 == 0:
                print(f"Pre-equilibrated {i+1} of {num_equil} edges")
        print(f"Pre-equilibrated {num_equil} edges")
        if cache_path:
            with open(cache_path, "wb") as ofs:
                dump(self._equil_cache, ofs)
            print(f"Saved equilibration_cache to {cache_path}")

    def predict(self, ff_params: list, mol_a: Chem.Mol, mol_b: Chem.Mol, core: np.ndarray):
        """
        Predict the ddG of morphing mol_a into mol_b. This function is differentiable w.r.t. ff_params.

        Parameters
        ----------

        ff_params: list of np.ndarray
            This should match the ordered params returned by the forcefield

        mol_a: Chem.Mol
            Starting molecule corresponding to lambda = 0

        mol_b: Chem.Mol
            Starting molecule corresponding to lambda = 1

        core: np.ndarray
            N x 2 list of ints corresponding to the atom mapping of the core.

        Returns
        -------
        float
            delta delta G in kJ/mol
        aux
            list of TI results
        """

        stage_dGs = []
        stage_results = []

        for stage, host_system, host_coords, host_box, lambda_schedule in [
            ("complex", self.complex_system, self.complex_coords, self.complex_box, self.complex_schedule),
            ("solvent", self.solvent_system, self.solvent_coords, self.solvent_box, self.solvent_schedule),
        ]:
            single_topology = topology.SingleTopology(mol_a, mol_b, core, self.ff)
            rfe = free_energy.RelativeFreeEnergy(single_topology)
            edge_hash = self._edge_hash(stage, mol_a, mol_b, core)
            if self.pre_equilibrate and edge_hash in self._equil_cache:
                cached_state = self._equil_cache[edge_hash]
                x0 = cached_state.coords
                host_box = cached_state.box
                num_host_coords = len(host_coords)
                unbound_potentials, sys_params, masses, _ = rfe.prepare_host_edge(ff_params, host_system, host_coords)
                mol_a_size = mol_a.GetNumAtoms()
                # Use Dual Topology to pre equilibrate, so have to get the mean of the two sets of mol,
                # normally done within prepare_host_edge, but the whole system has moved by this stage
                x0 = np.concatenate(
                    [
                        x0[:num_host_coords],
                        np.mean(
                            single_topology.interpolate_params(
                                x0[num_host_coords : num_host_coords + mol_a_size], x0[num_host_coords + mol_a_size :]
                            ),
                            axis=0,
                        ),
                    ]
                )
            else:
                if self.pre_equilibrate:
                    print("Edge not correctly pre-equilibrated, ensure equilibrate_edges was called")
                print(f"Minimizing the {stage} host structure to remove clashes.")
                # (ytz): this isn't strictly symmetric, and we should modify minimize later on remove
                # the hysteresis by jointly minimizing against a and b at the same time. We may also want
                # to remove the randomness completely from the minimization.
                min_host_coords = minimizer.minimize_host_4d(
                    [mol_a, mol_b], host_system, host_coords, self.ff, host_box
                )

                unbound_potentials, sys_params, masses, coords = rfe.prepare_host_edge(
                    ff_params, host_system, min_host_coords
                )

                x0 = coords
            v0 = np.zeros_like(x0)

            time_step = 1.5e-3

            harmonic_bond_potential = unbound_potentials[0]
            bond_list = get_bond_list(harmonic_bond_potential)
            if self.hmr:
                masses = apply_hmr(masses, bond_list)
                time_step = 2.5e-3
            group_idxs = get_group_indices(bond_list)

            seed = 0

            temperature = 300.0
            pressure = 1.0

            integrator = LangevinIntegrator(temperature, time_step, 1.0, masses, seed)

            barostat = MonteCarloBarostat(x0.shape[0], pressure, temperature, group_idxs, self.barostat_interval, seed)

            model = estimator.FreeEnergyModel(
                unbound_potentials,
                self.client,
                host_box,
                x0,
                v0,
                integrator,
                lambda_schedule,
                self.equil_steps,
                self.prod_steps,
                barostat,
            )

            dG, results = estimator.deltaG(model, sys_params)

            stage_dGs.append(dG)
            stage_results.append((stage, results))

        pred = stage_dGs[0] - stage_dGs[1]

        return pred, stage_results

    def loss(self, ff_params, mol_a, mol_b, core, label_ddG):
        """
        Computes the L1 loss relative to some label. See predict() for the type signature.

        This function is differentiable w.r.t. ff_params.

        Parameters
        ----------
        label_ddG: float
            Label ddG in kJ/mol of the alchemical transformation.

        Returns
        -------
        float
            loss

        TODO: make this configurable, using loss functions from in fe/loss.py

        """
        pred_ddG, results = self.predict(ff_params, mol_a, mol_b, core)
        loss = jnp.abs(pred_ddG - label_ddG)
        return loss, results
