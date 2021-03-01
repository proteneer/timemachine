from collections import namedtuple

import functools
import copy
import jax
import numpy as np

from timemachine.lib import potentials, custom_ops

class SimulationResult:
    def __init__(self, xs=None, du_dls=None, du_dps=None):
        self.xs = xs
        self.du_dls = du_dls
        self.du_dps = du_dps

    def save(self, path):
        return np.savez(path, xs=self.xs, du_dls=self.du_dls, du_dps=self.du_dps)



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


class FreeEnergyModel:
    def __init__(self, unbound_potentials, client, box, x0, v0, integrator, lambda_schedule, equil_steps, prod_steps, callback=None):
        self.unbound_potentials = unbound_potentials
        self.client = client
        self.box = box
        self.x0 = x0
        self.v0 = v0
        self.integrator = integrator
        self.lambda_schedule = lambda_schedule
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        self.callback = callback


from typing import Tuple, List, Any
gradient = List[Any] # TODO: make this more descriptive of dG_grad structure

def _deltaG(model, sys_params, callback=None) -> Tuple[float, gradient]:

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

    if not (callback is None):
        # TODO: what should the signature of callback be?
        callback(results)

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

    return dG, dG_grad

@functools.partial(jax.custom_vjp, nondiff_argnums=(0,2))
def deltaG(model, sys_params, callback=None) -> float:
    return _deltaG(model=model, sys_params=sys_params, callback=callback)[0]

def deltaG_fwd(model, sys_params, callback=None) -> Tuple[float, gradient]:
    """same signature as DeltaG, but returns"""
    return _deltaG(model=model, sys_params=sys_params, callback=callback)

def deltaG_bwd(model, callback, residual, grad) -> Tuple[gradient]:
    """Note: nondiff args must appear first here, even though one of them appears last in the original function's signature!

    Ahh: https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#jax-custom-vjp-with-nondiff-argnums
    >A similar option exists for jax.custom_vjp, and similarly the convention is that the non-differentiable arguments
    >are passed as the first arguments to the rules, no matter where they appear in the original function’s signature.

    """
    return ([grad*r for r in residual], )

deltaG.defvjp(deltaG_fwd, deltaG_bwd)