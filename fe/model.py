import copy
import numpy as np
import jax
import jax.numpy as jnp

from simtk import openmm
from rdkit import Chem

from md import minimizer
from timemachine.lib import LangevinIntegrator
from fe import free_energy, topology, estimator
from ff import Forcefield

from parallel.client import AbstractClient
from typing import Optional
from functools import partial


from timemachine.lib import custom_ops

def simulate(
    lamb,
    box,
    x0,
    v0,
    final_potentials,
    integrator,
    equil_steps,
    prod_steps,
    compute_grad,
    x_interval=1000,
    du_dl_interval=5):
    """
    Run a simulation and collect relevant statistics for this simulation.

    Parameters
    ----------
    lamb: float
        lambda parameter

    box: np.array
        3x3 numpy array of the box, dtype should be np.float64

    x0: np.array
        Nx3 numpy array of the coordinates

    v0: np.array
        Nx3 numpy array of the velocities

    final_potentials: list
        list of unbound potentials

    integrator: timemachine.Integrator
        integrator to be used for dynamics

    equil_steps: int
        number of equilibration steps

    prod_steps: int
        number of production steps

    compute_grad: bool
        whether or not we compute derivatives

    x_interval: int
        how often we store coordinates. if x_interval == 0 then
        no frames are returned.

    du_dl_interval: int
        how often we store du_dls. if du_dl_interval == 0 then
        no du_dls are returned

    Returns
    -------
    SimulationResult
        Results of the simulation.

    """
    all_impls = []

    # set up observables for du_dps here as well.
    du_dp_obs = []

    if compute_grad:
        for bp in final_potentials:
            impl = bp.bound_impl(np.float32)
            all_impls.append(impl)
            du_dp_obs.append(custom_ops.AvgPartialUPartialParam(impl, 5))

    if integrator.seed == 0:
        integrator = copy.deepcopy(integrator)
        integrator.seed = np.random.randint(np.iinfo(np.int32).max)

    intg_impl = integrator.impl()
    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg_impl,
        all_impls
    )

    # equilibration
    equil_schedule = np.ones(equil_steps)*lamb
    ctxt.multiple_steps(equil_schedule)

    for obs in du_dp_obs:
        ctxt.add_observable(obs)

    prod_schedule = np.ones(prod_steps)*lamb

    full_du_dls, xs = ctxt.multiple_steps(prod_schedule, du_dl_interval, x_interval)

    # keep the structure of grads the same as that of final_potentials so we can properly
    # form their vjps.
    if compute_grad:
        grads = []
        for obs in du_dp_obs:
            grads.append(obs.avg_du_dp())
    else:
        grads = None

    return xs, full_du_dls, grads


class RBFEModel():

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
        aux
            list of TI results
        """

        stage_dGs = []
        stage_results = []

        for stage, host_system, host_coords, host_box, lambda_schedule in [
            ("complex", self.complex_system, self.complex_coords, self.complex_box, self.complex_schedule),
            ("solvent", self.solvent_system, self.solvent_coords, self.solvent_box, self.solvent_schedule)]:

            print(f"Minimizing the {stage} host structure to remove clashes.")
            min_host_coords = minimizer.minimize_host_4d([mol_a, mol_b], host_system, host_coords, self.ff, host_box)

            single_topology = topology.SingleTopology(mol_a, mol_b, core, self.ff)
            rfe = free_energy.RelativeFreeEnergy(single_topology)

            unbound_potentials, sys_params, masses, coords = rfe.prepare_host_edge(ff_params, host_system, min_host_coords)

            x0 = coords
            v0 = np.zeros_like(coords)

            seed = 0

            integrator = LangevinIntegrator(
                300.0,
                1.5e-3,
                1.0,
                masses,
                seed
            )

            # dynamically detect if we need derivatives or not.
            assert np.all([isinstance(x, jax.interpreters.ad.JVPTracer) for x in sys_params])

            compute_grad = False

            for x in sys_params:
                if isinstance(x, jax.interpreters.ad.JVPTracer):
                    compute_grad = True
                    break

            bound_potentials = []
            for params, unbound_pot in zip(
                    # (ytz): stop backprop on sys_params
                    jax.lax.stop_gradient(sys_params),
                    unbound_potentials):
                bp = unbound_pot.bind(np.asarray(params))
                bound_potentials.append(bp)

            all_args = []
            for lamb in lambda_schedule:
                all_args.append((
                    lamb,
                    host_box,
                    x0,
                    v0,
                    bound_potentials,
                    integrator,
                    self.equil_steps,
                    self.prod_steps,
                    compute_grad
                ))

            if self.client is None:
                results = []
                for args in all_args:
                    results.append(simulate(*args))
            else:
                futures = []
                for args in all_args:
                    futures.append(self.client.submit(simulate, *args))
                results = [x.result() for x in futures]

            mean_du_dls = []
            # this will be populated with None if derivatives are not required
            dG_grads = []

            for _, du_dls, du_dps in results:
                # (ytz): figure out what to do with stddev(du_dl) later
                mean_du_dls.append(np.mean(du_dls))
                dG_grads.append(du_dps)

            dGs = np.trapz(mean_du_dls, lambda_schedule)

            # (ytz): use the thermodynamic derivative
            model = estimator.FreeEnergyModel(
                [dGs],
                [dG_grads]
            )

            dG = estimator.deltaG(model, sys_params)

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
