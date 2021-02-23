from collections import namedtuple

import functools
import copy
import jax
import numpy as np

from timemachine.lib import potentials, custom_ops

def simulate(lamb, box, x0, v0, final_potentials, integrator, equil_steps, prod_steps):
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
    du_dl_freq = 5
    full_du_dls = ctxt.multiple_steps(prod_schedule, du_dl_freq)

    du_dl_mean, du_dl_std = np.mean(full_du_dls), np.std(full_du_dls)

    # keep the structure of grads the same as that of final_potentials so we can properly
    # form their vjps.
    grads = []
    for obs in du_dp_obs:
        grads.append(obs.avg_du_dp())

    return (du_dl_mean, du_dl_std), grads


FreeEnergyModel = namedtuple("FreeEnergyModel", [
    "unbound_potentials",
    "client",
    "box",
    "x0",
    "v0",
    "integrator",
    "lambda_schedule",
    "equil_steps",
    "prod_steps"])

def _deltaG(model, sys_params):

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

    du_dls = []
    all_grads = []

    for (du_dl_mean, du_dl_std), grads in results:
        # (ytz): figure out what to do with du_dl_std later
        du_dls.append(du_dl_mean)
        all_grads.append(grads)

    dG = np.trapz(du_dls, model.lambda_schedule)
    dG_grad = []
    for rhs, lhs in zip(all_grads[-1], all_grads[0]):
        dG_grad.append(rhs - lhs)

    return dG, dG_grad

@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def deltaG(model, sys_params):
    return _deltaG(model, sys_params)[0]

def deltaG_fwd(model, sys_params):
    return _deltaG(model, sys_params)

def deltaG_bwd(model, residual, grad):
    return ([grad*r for r in residual],)

deltaG.defvjp(deltaG_fwd, deltaG_bwd)