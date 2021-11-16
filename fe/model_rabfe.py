from abc import ABC

import functools
import numpy as np
import dataclasses
from typing import Optional, Tuple, Any, List

from simtk import openmm
from rdkit import Chem

from timemachine.lib import potentials, LangevinIntegrator, MonteCarloBarostat
from timemachine import constants
from timemachine.potentials import rmsd

from fe.frames import endpoint_frames_only
from fe import free_energy_rabfe, topology, estimator_abfe, model_utils
from ff import Forcefield

from fe.free_energy_rabfe import (
    construct_absolute_lambda_schedule_complex,
    construct_absolute_lambda_schedule_solvent,
    construct_conversion_lambda_schedule,
    RABFEResult,
    get_romol_conf,
)

from parallel.client import AbstractClient, _MockFuture

from md import minimizer
from md.barostat.utils import get_group_indices, get_bond_list


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
        prod_steps: int,
        frame_filter: Optional[callable] = None,
    ):

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
        if frame_filter is None:
            frame_filter = endpoint_frames_only
        self.frame_filter = frame_filter

    def setup_topology(self, mol):
        raise NotImplementedError()

    def simulate_futures(
        self,
        ff_params,
        mol,
        x0,
        box0,
        prefix,
        core_idxs=None,
        seed=0,
    ) -> Tuple[List[Any], estimator_abfe.FreeEnergyModel, List[Any]]:
        top = self.setup_topology(mol)

        afe = free_energy_rabfe.AbsoluteFreeEnergy(mol, top)

        unbound_potentials, sys_params, masses = afe.prepare_host_edge(ff_params, self.host_system)

        beta = 1 / (constants.BOLTZ * self.temperature)

        bond_list = get_bond_list(unbound_potentials[0])
        masses = model_utils.apply_hmr(masses, bond_list)
        friction = 1.0
        integrator = LangevinIntegrator(self.temperature, self.dt, friction, masses, seed)

        group_indices = get_group_indices(bond_list)
        barostat_interval = 5
        barostat = MonteCarloBarostat(
            x0.shape[0], self.pressure, self.temperature, group_indices, barostat_interval, seed
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
            prefix,
        )
        bound_potentials = []
        for params, unbound_pot in zip(sys_params, model.unbound_potentials):
            bp = unbound_pot.bind(np.asarray(params))
            bound_potentials.append(bp)

        all_args = []
        for lamb_idx, lamb in enumerate(model.lambda_schedule):

            subsample_interval = 1000

            all_args.append(
                (
                    lamb,
                    model.box,
                    model.x0,
                    model.v0,
                    bound_potentials,
                    model.integrator,
                    model.barostat,
                    model.equil_steps,
                    model.prod_steps,
                    subsample_interval,
                    subsample_interval,
                    model.lambda_schedule,
                )
            )

        if endpoint_correct:

            assert isinstance(bound_potentials[-1], potentials.HarmonicBond)

            all_args.append(
                (
                    1.0,
                    model.box,
                    model.x0,
                    model.v0,
                    bound_potentials[:-1],  # strip out the restraints
                    model.integrator,
                    model.barostat,
                    model.equil_steps,
                    model.prod_steps,
                    subsample_interval,
                    subsample_interval,
                    [],  # no need to evaluate Us for the endpoint correction
                )
            )

        futures = []
        if self.client is None:
            for args in all_args:
                futures.append(_MockFuture(estimator_abfe.simulate(*args)))
        else:
            for args in all_args:
                futures.append(self.client.submit(estimator_abfe.simulate, *args))
        return sys_params, model, futures

    def predict_from_futures(self, sys_params, mol, model: estimator_abfe.FreeEnergyModel, futures: List[Any]):
        results = [fut.result() for fut in futures]
        dG, dG_err, results = estimator_abfe.deltaG_from_results(model, results, sys_params)

        # uncomment if we want to visualize
        combined_topology = model_utils.generate_imaged_topology(
            [self.host_topology, mol], model.x0, model.box, "initial_" + model.prefix + ".pdb"
        )

        for lambda_idx, res in self.frame_filter(results):
            np.savez(
                f"initial_{model.prefix}_lambda_idx_{lambda_idx}.npz",
                xs=res.xs,
                boxes=res.boxes,
                du_dps=res.du_dps,
                lambda_us=res.lambda_us,
            )

        return dG, dG_err

    def predict(self, ff_params, mol, x0, box0, prefix, core_idxs=None, seed=0):
        """Compute the absolute free of energy of decoupling mol_a.

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
        sys_params, model, futures = self.simulate_futures(
            ff_params, mol, x0, box0, prefix, core_idxs=core_idxs, seed=seed
        )

        dG, dG_err = self.predict_from_futures(sys_params, mol, model, futures)

        return dG, dG_err


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
        prod_steps: int,
        frame_filter: Optional[callable] = None,
    ):

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
        if frame_filter is None:
            frame_filter = endpoint_frames_only
        self.frame_filter = frame_filter

    def setup_topology(self, mol_a, mol_b):
        raise NotImplementedError()

    def _futures_a_to_b(self, ff_params, mol_a, mol_b, combined_core_idxs, x0, box0, prefix, seed):

        num_host_atoms = x0.shape[0] - mol_a.GetNumAtoms() - mol_b.GetNumAtoms()

        # (ytz): super ugly, undo combined_core_idxs to get back original idxs
        core_idxs = combined_core_idxs - num_host_atoms
        core_idxs[:, 1] -= mol_a.GetNumAtoms()

        dual_topology = self.setup_topology(mol_a, mol_b)
        rfe = free_energy_rabfe.RelativeFreeEnergy(dual_topology)

        unbound_potentials, sys_params, masses = rfe.prepare_host_edge(ff_params, self.host_system)

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

        beta = 1 / (constants.BOLTZ * self.temperature)

        bond_list = np.concatenate([unbound_potentials[0].get_idxs(), core_idxs])
        masses = model_utils.apply_hmr(masses, bond_list)

        friction = 1.0
        integrator = LangevinIntegrator(self.temperature, self.dt, friction, masses, seed)
        bond_list = list(map(tuple, bond_list))
        group_indices = get_group_indices(bond_list)
        barostat_interval = 5

        barostat = MonteCarloBarostat(
            x0.shape[0], self.pressure, self.temperature, group_indices, barostat_interval, seed
        )

        endpoint_correct = True
        model = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            self.client,
            box0,  # important, use equilibrated box.
            x0,
            v0,
            integrator,
            barostat,
            self.host_schedule,
            self.equil_steps,
            self.prod_steps,
            beta,
            prefix,
        )

        bound_potentials = []
        for params, unbound_pot in zip(sys_params, model.unbound_potentials):
            bp = unbound_pot.bind(np.asarray(params))
            bound_potentials.append(bp)

        all_args = []
        for lamb_idx, lamb in enumerate(model.lambda_schedule):

            subsample_interval = 1000

            all_args.append(
                (
                    lamb,
                    model.box,
                    model.x0,
                    model.v0,
                    bound_potentials,
                    model.integrator,
                    model.barostat,
                    model.equil_steps,
                    model.prod_steps,
                    subsample_interval,
                    subsample_interval,
                    model.lambda_schedule,
                )
            )

        if endpoint_correct:

            assert isinstance(bound_potentials[-1], potentials.HarmonicBond)

            all_args.append(
                (
                    1.0,
                    model.box,
                    model.x0,
                    model.v0,
                    bound_potentials[:-1],  # strip out the restraints
                    model.integrator,
                    model.barostat,
                    model.equil_steps,
                    model.prod_steps,
                    subsample_interval,
                    subsample_interval,
                    [],  # no need to evaluate Us for the endpoint correction
                )
            )

        futures = []
        if self.client is None:
            for args in all_args:
                futures.append(_MockFuture(estimator_abfe.simulate(*args)))
        else:
            for args in all_args:
                futures.append(self.client.submit(estimator_abfe.simulate, *args))

        return sys_params, model, futures

    def simulate_futures(
        self,
        ff_params,
        mol_a,
        mol_b,
        core,
        x0,
        box0,
        prefix,
        seed=0,
    ) -> Tuple[List[Any], List[estimator_abfe.FreeEnergyModel], List[List[Any]]]:

        num_host_atoms = x0.shape[0] - mol_a.GetNumAtoms() - mol_b.GetNumAtoms()
        host_coords = x0[:num_host_atoms]
        mol_a_coords = x0[num_host_atoms : num_host_atoms + mol_a.GetNumAtoms()]
        mol_b_coords = x0[num_host_atoms + mol_a.GetNumAtoms() :]

        # pull out mol_b from combined state
        combined_core_idxs = np.copy(core)
        combined_core_idxs[:, 0] += num_host_atoms
        combined_core_idxs[:, 1] += num_host_atoms + mol_a.GetNumAtoms()
        # this is redundant, but thought it best to be explicit about ordering here..
        combined_coords = np.concatenate([host_coords, mol_a_coords, mol_b_coords])

        all_sys = []
        models = []
        all_futures = []
        sys_params, model, futures = self._futures_a_to_b(
            ff_params,
            mol_a,
            mol_b,
            combined_core_idxs,
            combined_coords,
            box0,
            prefix + "_ref_to_mol",
            seed,
        )

        all_sys.append(sys_params)
        models.append(model)
        all_futures.append(futures)

        # pull out mol_a from combined state
        combined_core_idxs = np.copy(core)
        # swap the ligand coordinates in the reverse direction
        combined_core_idxs[:, 0] = core[:, 1]
        combined_core_idxs[:, 1] = core[:, 0]
        combined_core_idxs[:, 0] += num_host_atoms
        combined_core_idxs[:, 1] += num_host_atoms + mol_b.GetNumAtoms()
        combined_coords = np.concatenate([host_coords, mol_b_coords, mol_a_coords])
        sys_params, model, futures = self._futures_a_to_b(
            ff_params,
            mol_b,
            mol_a,
            combined_core_idxs,
            combined_coords,
            box0,
            prefix + "_mol_to_ref",
            seed,
        )

        all_sys.append(sys_params)
        models.append(model)
        all_futures.append(futures)

        return all_sys, models, all_futures

    def predict_from_futures(
        self, sys_params, mol_a, mol_b, models: List[estimator_abfe.FreeEnergyModel], futures: List[List[Any]]
    ):
        assert len(futures) == 2
        assert len(models) == 2
        assert len(sys_params) == 2
        err = 0
        fwd_dG = 0
        back_dG = 0
        for i, (params, model, sub_futures) in enumerate(zip(sys_params, models, futures)):
            results = [fut.result() for fut in sub_futures]
            dG, dG_err, results = estimator_abfe.deltaG_from_results(model, results, params)

            # Save out the pdb
            combined_topology = model_utils.generate_imaged_topology(
                [self.host_topology, mol_a, mol_b], model.x0, model.box, f"initial_{model.prefix}.pdb"
            )

            for lambda_idx, res in self.frame_filter(results):
                np.savez(
                    f"initial_{model.prefix}_lambda_idx_{lambda_idx}.npz",
                    xs=res.xs,
                    boxes=res.boxes,
                    du_dps=res.du_dps,
                    lambda_us=res.lambda_us,
                )
            # fwd_dG is the free energy of moving X-A-B into X-A+B
            # back_dG is the free energy of moving X-B-A into X-B+A
            # -fwd_dG + back_dG is the free energy of moving X-A+B -> X-B+A
            # i.e. the free energy of "unbinding" A
            if i == 0:
                fwd_dG = dG
            else:
                back_dG = dG
            err += dG_err ** 2
        err = np.sqrt(err)
        return -fwd_dG + back_dG, err

    def predict(
        self,
        ff_params: list,
        mol_a: Chem.Mol,
        mol_b: Chem.Mol,
        core_idxs: np.array,
        x0: np.array,
        box0: np.array,
        prefix: str,
        seed=0,
    ):
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

        sys_params, models, futures = self.simulate_futures(
            ff_params,
            mol_a,
            mol_b,
            core_idxs,
            x0,
            box0,
            prefix,
            seed=seed,
        )
        dG, dG_err = self.predict_from_futures(
            sys_params,
            mol_b,
            mol_a,
            models,
            futures,
        )

        return dG, dG_err


# subclasses specific for each model


class AbsoluteHydrationModel(AbsoluteModel):
    def setup_topology(self, mol):
        return topology.BaseTopologyRHFE(mol, self.ff)


class RelativeHydrationModel(RelativeModel):
    def setup_topology(self, mol_a, mol_b):
        return topology.DualTopologyRHFE(mol_a, mol_b, self.ff)


class AbsoluteConversionModel(AbsoluteModel):
    def setup_topology(self, mol):
        top = topology.BaseTopologyConversion(mol, self.ff)
        return top


class AbsoluteStandardHydrationModel(AbsoluteModel):
    def setup_topology(self, mol):
        top = topology.BaseTopologyStandardDecoupling(mol, self.ff)
        return top


class RelativeBindingModel(RelativeModel):
    def setup_topology(self, mol_a, mol_b):
        top = topology.DualTopologyStandardDecoupling(mol_a, mol_b, self.ff)
        return top


@dataclasses.dataclass
class RABFESimulationFutures:
    complex_decouple: None
    complex_conversion: None
    solvent_decouple: None
    solvent_conversion: None
    mol: None
    blocker: None
    seed: 0


class RABFEModel:
    def __init__(
        self,
        client: AbstractClient,
        forcefield,
        blocker: Chem.Mol,
        complex_system,
        complex_topology,
        solvent_system,
        solvent_topology,
        temperature: float,
        pressure: float,
        dt: float,
        complex_coords: np.ndarray,
        complex_box: np.ndarray,
        solvent_coords: np.ndarray,
        solvent_box: np.ndarray,
        frame_filter=None,
    ):
        self.client = client
        self.forcefield = forcefield
        self.blocker = blocker
        self.complex_system = complex_system
        self.complex_topology = complex_topology
        self.solvent_system = solvent_system
        self.solvent_topology = solvent_topology
        self.pressure = pressure
        self.dt = dt
        self.temperature = temperature

        self.complex_coords = complex_coords
        self.complex_box = complex_box

        self.solvent_coords = solvent_coords
        self.solvent_box = solvent_box

        if frame_filter is None:
            frame_filter = endpoint_frames_only
        self.frame_filter = frame_filter

        # Setup Complex Schedules
        self.complex_conversion_schedule = construct_conversion_lambda_schedule(64)
        self.complex_decouple_schedule = construct_absolute_lambda_schedule_complex(
            63
        )  # 1 less for endpoint correction

        # Setup Solvent Schedules
        self.solvent_conversion_schedule = construct_conversion_lambda_schedule(64)
        self.solvent_decouple_schedule = construct_absolute_lambda_schedule_solvent(64)

        # Setup Complex Steps
        self.complex_decouple_equil_steps = 200, 000
        self.complex_decouple_prod_steps = 800, 000
        self.complex_conversion_equil_steps = 200, 000
        self.complex_conversion_prod_steps = 800, 000

        self.solvent_decouple_equil_steps = 200, 000
        self.solvent_decouple_prod_steps = 800, 000
        self.solvent_conversion_equil_steps = 200, 000
        self.solvent_conversion_prod_steps = 800, 000

        # Will load these later to ensure user has configured all options
        self.solvent_conversion_model = None
        self.solvent_decouple_model = None

        self.complex_conversion_model = None
        self.complex_decouple_model = None

        self._models_loaded = False

    def load_models(self, force: bool = False):
        if not force and self._models_loaded:
            return

        self.complex_conversion_model = AbsoluteConversionModel(
            self.client,
            self.forcefield,
            self.complex_system,
            self.complex_conversion_schedule,
            self.complex_topology,
            self.temperature,
            self.pressure,
            self.dt,
            self.complex_conversion_equil_steps,
            self.complex_conversion_prod_steps,
            frame_filter=self.frame_filter,
        )

        self.complex_decouple_model = RelativeBindingModel(
            self.client,
            self.forcefield,
            self.complex_system,
            self.complex_decouple_schedule,
            self.complex_topology,
            self.temperature,
            self.pressure,
            self.dt,
            self.complex_decouple_equil_steps,
            self.complex_decouple_prod_steps,
            frame_filter=self.frame_filter,
        )

        self.solvent_conversion_model = AbsoluteConversionModel(
            self.client,
            self.forcefield,
            self.solvent_system,
            self.solvent_conversion_schedule,
            self.solvent_topology,
            self.temperature,
            self.pressure,
            self.dt,
            self.solvent_conversion_equil_steps,
            self.solvent_conversion_prod_steps,
            frame_filter=self.frame_filter,
        )

        self.solvent_decouple_model = AbsoluteStandardHydrationModel(
            self.client,
            self.forcefield,
            self.solvent_system,
            self.solvent_decouple_schedule,
            self.solvent_topology,
            self.temperature,
            self.pressure,
            self.dt,
            self.solvent_decouple_equil_steps,
            self.solvent_decouple_prod_steps,
            frame_filter=self.frame_filter,
        )

    def simulate_futures(
        self,
        ff_params: np.ndarray,
        mol: Chem.Mol,
        core_idxs: np.ndarray,
        suffix: str,
        seed: int = 0,
    ) -> RABFESimulationFutures:
        # Ensure that the relevant models are constructed
        self.load_models()

        if seed == 0:
            seed = np.random.randint(np.iinfo(np.int32).max)

        mol_coords = get_romol_conf(mol)  # original coords
        num_complex_atoms = self.complex_coords.shape[0] - self.blocker.GetNumAtoms()
        R, t = rmsd.get_optimal_rotation_and_translation(
            # Assumes complex coords have blocker already contained
            x1=self.complex_coords[num_complex_atoms:][core_idxs[:, 1]],  # reference core atoms
            x2=mol_coords[core_idxs[:, 0]],  # mol core atoms
        )

        aligned_mol_coords = rmsd.apply_rotation_and_translation(mol_coords, R, t)

        blocker_coords = self.complex_coords[num_complex_atoms:]
        complex_host_coords = self.complex_coords[:num_complex_atoms]
        complex_box0 = self.complex_box

        complex_decouple_x0 = minimizer.minimize_host_4d(
            [mol, self.blocker],
            self.complex_system,
            complex_host_coords,
            self.forcefield,
            complex_box0,
            [aligned_mol_coords, blocker_coords],
        )
        complex_decouple_x0 = np.concatenate([complex_decouple_x0, aligned_mol_coords, blocker_coords])

        # compute the free energy of conversion in complex
        complex_conversion_x0 = minimizer.minimize_host_4d(
            [mol],
            self.complex_system,
            complex_host_coords,
            self.forcefield,
            complex_box0,
            [aligned_mol_coords],
        )
        complex_conversion_x0 = np.concatenate([complex_conversion_x0, aligned_mol_coords])

        solvent_box0 = self.solvent_box
        min_solvent_coords = minimizer.minimize_host_4d(
            [mol], self.solvent_system, self.solvent_coords, self.forcefield, solvent_box0
        )
        solvent_x0 = np.concatenate([min_solvent_coords, mol_coords])

        futures = RABFESimulationFutures(
            complex_decouple=self.complex_decouple_model.simulate_futures(
                ff_params,
                mol,
                self.blocker,
                core_idxs,
                complex_decouple_x0,
                complex_box0,
                prefix="complex_decouple_" + suffix,
                seed=seed,
            ),
            complex_conversion=self.complex_conversion_model.simulate_futures(
                ff_params,
                mol,
                complex_conversion_x0,
                complex_box0,
                prefix="complex_conversion_" + suffix,
                seed=seed,
            ),
            solvent_decouple=self.solvent_decouple_model.simulate_futures(
                ff_params,
                mol,
                solvent_x0,
                solvent_box0,
                prefix="solvent_decouple_" + suffix,
                seed=seed,
            ),
            solvent_conversion=self.solvent_conversion_model.simulate_futures(
                ff_params,
                mol,
                solvent_x0,
                solvent_box0,
                prefix="solvent_conversion_" + suffix,
                seed=seed,
            ),
            mol=mol,
            blocker=self.blocker,
            seed=seed,
        )
        return futures

    def predict_from_futures(self, futures: RABFESimulationFutures) -> Tuple[float, float]:
        dG_complex_decouple, dG_complex_decouple_error = self.complex_decouple_model.predict_from_futures(
            futures.complex_decouple[0],
            futures.mol,
            futures.blocker,
            futures.complex_decouple[1],
            futures.complex_decouple[2],
        )
        dG_complex_conversion, dG_complex_conversion_error = self.complex_conversion_model.predict_from_futures(
            futures.complex_conversion[0],
            futures.mol,
            futures.complex_conversion[1],
            futures.complex_conversion[2],
        )
        dG_solvent_decouple, dG_solvent_decouple_error = self.solvent_decouple_model.predict_from_futures(
            futures.solvent_decouple[0],
            futures.mol,
            futures.solvent_decouple[1],
            futures.solvent_decouple[2],
        )
        dG_solvent_conversion, dG_solvent_conversion_error = self.solvent_conversion_model.predict_from_futures(
            futures.solvent_conversion[0],
            futures.mol,
            futures.solvent_conversion[1],
            futures.solvent_conversion[2],
        )

        rabfe_result = RABFEResult(
            mol_name=futures.mol.GetProp("_Name"),
            dG_complex_conversion=dG_complex_conversion,
            dG_complex_decouple=dG_complex_decouple,
            dG_solvent_conversion=dG_solvent_conversion,
            dG_solvent_decouple=dG_solvent_decouple,
        )
        rabfe_result.log()
        dG_err = np.sqrt(
            dG_complex_conversion_error ** 2
            + dG_complex_decouple_error ** 2
            + dG_solvent_conversion_error ** 2
            + dG_solvent_decouple_error ** 2
        )

        return rabfe_result.dG_bind, dG_err

    def predict(self, ff_params, mol, core_idxs, suffix) -> Tuple[float, float]:
        futures = self.simulate_futures(ff_params, mol, core_idxs, suffix)
        results = self.predict_from_futures(futures)
        return results

    def set_complex_conversion_schedule(self, schedule: List[int]):
        self.complex_conversion_schedule = schedule

    def set_complex_decouple_schedule(self, schedule: List[int]):
        self.complex_decouple_schedule = schedule

    def set_solvent_conversion_schedule(self, schedule: List[int]):
        self.solvent_conversion_schedule = schedule

    def set_solvent_decouple_schedule(self, schedule: List[int]):
        self.solvent_decouple_schedule = schedule

    def set_complex_decouple_equil_steps(self, steps: int):
        self.complex_decouple_equil_steps = steps

    def set_complex_decouple_prod_steps(self, steps: int):
        self.complex_decouple_prod_steps = steps

    def set_complex_conversion_equil_steps(self, steps: int):
        self.complex_conversion_equil_steps = steps

    def set_complex_conversion_prod_steps(self, steps: int):
        self.complex_conversion_prod_steps = steps

    def set_solvent_decouple_equil_steps(self, steps: int):
        self.solvent_decouple_equil_steps = steps

    def set_solvent_decouple_prod_steps(self, steps: int):
        self.solvent_decouple_prod_steps = steps

    def set_solvent_conversion_equil_steps(self, steps: int):
        self.solvent_conversion_equil_steps = steps

    def set_solvent_conversion_prod_steps(self, steps: int):
        self.solvent_conversion_prod_steps = steps
