import numpy as np
import jax.numpy as jnp

from simtk import openmm
from rdkit import Chem

from md import minimizer
from timemachine.lib import LangevinIntegrator
from fe import free_energy, topology, estimator
from ff import Forcefield

from parallel.client import AbstractClient

class RBFEModel():

    def __init__(
        self,
        client: AbstractClient or None,
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
        prod_steps: int):

        self.complex_system = complex_system
        self.complex_coords = complex_coords
        self.complex_box = complex_box
        self.complex_schedule = complex_schedule
        self.solvent_system = solvent_system
        self.solvent_coords = solvent_coords
        self.solvent_box = solvent_box
        self.solvent_schedule = solvent_schedule
        self.client = client
        self.ff = ff
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps

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

        """

        stage_dGs = []

        for stage, host_system, host_coords, host_box, lambda_schedule in [
            ("complex", self.complex_system, self.complex_coords, self.complex_box, self.complex_schedule),
            ("solvent", self.solvent_system, self.solvent_coords, self.solvent_box, self.solvent_schedule)]:

            print("Minimizing the host structure to remove clashes.")
            # (ytz): this isn't strictly symmetric, and we should modify minimize later on remove
            # the hysteresis by jointly minimizing against a and b at the same time. We may also want
            # to remove the randomness completely from the minimization.
            min_host_coords = minimizer.minimize_host_4d([mol_a, mol_b], host_system, host_coords, self.ff, host_box)

            single_topology = topology.SingleTopology(mol_a, mol_b, core, self.ff)
            rfe = free_energy.RelativeFreeEnergy(single_topology)

            unbound_potentials, sys_params, masses, coords = rfe.prepare_host_edge(ff_params, host_system, min_host_coords)

            x0 = coords
            v0 = np.zeros_like(coords)
            box = np.eye(3, dtype=np.float64)*100 # note: box unused

            seed = 0

            integrator = LangevinIntegrator(
                300.0,
                1.5e-3,
                1.0,
                masses,
                seed
            )

            model = estimator.FreeEnergyModel(
                unbound_potentials,
                self.client,
                host_box,
                x0,
                v0,
                integrator,
                lambda_schedule,
                self.equil_steps,
                self.prod_steps
            )

            dG = estimator.deltaG(model, sys_params)
            stage_dGs.append(dG)

        pred = stage_dGs[0] - stage_dGs[1]

        return pred

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

        """
        pred_ddG = self.predict(ff_params, mol_a, mol_b, core)
        loss = jnp.abs(pred_ddG - label_ddG)
        return loss
