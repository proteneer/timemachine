from collections import namedtuple

import functools
import copy
import jax
import numpy as np

from timemachine.lib import potentials, custom_ops

from typing import Tuple, List, Any

import dataclasses
import jax.numpy as jnp

@dataclasses.dataclass
class SimulationResult:
   xs: np.array
   du_dls: np.array
   du_dps: np.array

def flatten(v):
    return tuple(), (v.xs, v.du_dls, v.du_dps)

def unflatten(aux_data, children):
    xs, du_dls, du_dps = aux_data
    return SimulationResult(xs, du_dls, du_dps)

jax.tree_util.register_pytree_node(SimulationResult, flatten, unflatten)

def simulate(lamb, box, x0, v0, final_potentials, integrator, equil_steps, prod_steps, get_trajectory=False,
    x_interval=1000, du_dl_interval=5):
    """
    Run a simulation and collect relevant statistics for this simulation.

    Parameters
    ----------

    """
    all_impls = []
    bonded_impls = []
    nonbonded_impls = []

    # set up observables for du_dps here as well.
    du_dp_obs = []

    for bp in final_potentials:
        impl = bp.bound_impl(np.float32)
        if isinstance(bp, potentials.Nonbonded):
            nonbonded_impls.append(impl)
        else:
            bonded_impls.append(impl)
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

    # run MD, optionally pausing every du_dl_freq steps to extract x snapshot
    if not get_trajectory:
        xs = None
        full_du_dls = ctxt.multiple_steps(prod_schedule, du_dl_interval)

    else:
        xs = [] # in nanometers
        full_du_dls = []

        # simulate in x_interval-sized chunks
        t = 0
        while t < len(prod_schedule):
            chunk_size = min(len(prod_schedule), (t + x_interval)) - t
            full_du_dls.append(ctxt.multiple_steps(prod_schedule[t: (t + chunk_size)], chunk_size))
            xs.append(ctxt.get_x_t())
            t += chunk_size

        # lists -> arrays
        xs = np.array(xs)
        full_du_dls = np.hstack(full_du_dls)


    # keep the structure of grads the same as that of final_potentials so we can properly
    # form their vjps.
    grads = []
    for obs in du_dp_obs:
        grads.append(obs.avg_du_dp())

    result = SimulationResult(xs=xs, du_dls=full_du_dls, du_dps=grads)
    return result


FreeEnergyModel = namedtuple(
    "FreeEnergyModel",
    ["unbound_potentials", "client", "box", "x0", "v0", "integrator", "lambda_schedule", "equil_steps", "prod_steps"]
)

gradient = List[Any] # TODO: make this more descriptive of dG_grad structure

def _deltaG(model, sys_params) -> Tuple[Tuple[float, List], np.array]:

    assert len(sys_params) == len(model.unbound_potentials)

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, model.unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    if model.client is None:
        results = []
        for lamb in model.lambda_schedule:
            results.append(simulate(lamb, model.box, model.x0, model.v0, bound_potentials, model.integrator, model.equil_steps, model.prod_steps))
    else:
        futures = []
        for lamb in model.lambda_schedule:
            args = (lamb, model.box, model.x0, model.v0, bound_potentials, model.integrator, model.equil_steps, model.prod_steps)
            futures.append(model.client.submit(simulate, *args))

        results = [x.result() for x in futures]

    mean_du_dls = []
    all_grads = []

    for result in results:
        # (ytz): figure out what to do with stddev(du_dl) later
        mean_du_dls.append(np.mean(result.du_dls))
        all_grads.append(result.du_dps)

    dG = np.trapz(mean_du_dls, model.lambda_schedule)
    dG_grad = []
    for rhs, lhs in zip(all_grads[-1], all_grads[0]):
        dG_grad.append(rhs - lhs)

    return (dG, results), dG_grad

@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def deltaG(model, sys_params) -> Tuple[float, List]:
    return _deltaG(model=model, sys_params=sys_params)[0]

def deltaG_fwd(model, sys_params) -> Tuple[Tuple[float, List], np.array]:
    """same signature as DeltaG, but returns the full tuple"""
    return _deltaG(model=model, sys_params=sys_params)

def deltaG_bwd(model, residual, grad) -> Tuple[np.array]:
    """Note: nondiff args must appear first here, even though one of them appears last in the original function's signature!
    """
    # residual are the partial dG / partial dparams for each term
    # grad[0] is the adjoint of dG w.r.t. loss: partial L/partial dG
    # grad[1] is the adjoint of dG w.r.t. simulation result, which we don't use
    return ([grad[0]*r for r in residual],)

deltaG.defvjp(deltaG_fwd, deltaG_bwd)