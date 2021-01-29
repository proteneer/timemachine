import numpy as np

import jax.numpy as jnp


from md import minimizer

from timemachine.lib import LangevinIntegrator

from fe import free_energy_v2
from fe import estimator

class RBFEModel():

    def __init__(
        self,
        client,
        ff,
        complex_system,
        complex_coords,
        complex_box,
        complex_schedule,
        solvent_system,
        solvent_coords,
        solvent_box,
        solvent_schedule,
        equil_steps,
        prod_steps):

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

    def predict(self, ff_params, mol_a, mol_b, core):

        stage_dGs = []

        for stage, host_system, host_coords, host_box, lambda_schedule in [
            ("complex", self.complex_system, self.complex_coords, self.complex_box, self.complex_schedule),
            ("solvent", self.solvent_system, self.solvent_coords, self.solvent_box, self.solvent_schedule)]:

            print("Minimizing the host structure to remove clashes.")
            min_host_coords = minimizer.minimize_host_4d(mol_a, host_system, host_coords, self.ff, host_box)

            rfe = free_energy_v2.RelativeFreeEnergy(mol_a, mol_b, core, self.ff)

            unbound_potentials, sys_params, masses, coords = rfe.prepare_host_edge(ff_params, host_system, min_host_coords)

            x0 = coords
            v0 = np.zeros_like(coords)
            box = np.eye(3, dtype=np.float64)*100

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
        pred_ddG = self.predict(ff_params, mol_a, mol_b, core)
        loss = jnp.abs(pred_ddG - label_ddG)
        return loss