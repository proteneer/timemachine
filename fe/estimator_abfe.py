import pymbar
from fe import endpoint_correction
from collections import namedtuple

import dataclasses
import time
import functools
import copy
import jax
import numpy as np
from md import minimizer

from typing import Tuple, List, Any
import os

from fe import standard_state

from timemachine.lib import potentials, custom_ops

@dataclasses.dataclass
class SimulationResult:
   xs: np.array
   boxes: np.array
   du_dls: np.array
   du_dps: np.array
   left_dus: np.array
   right_dus: np.array

def flatten(v):
    return tuple(), (v.xs, v.boxes, v.du_dls, v.du_dps, v.left_dus, v.right_dus)

def unflatten(aux_data, children):
    xs, boxes, du_dls, du_dps, left_dus, right_dus = aux_data
    return SimulationResult(xs, boxes, du_dls, du_dps, left_dus, right_dus)

jax.tree_util.register_pytree_node(SimulationResult, flatten, unflatten)

def simulate(lamb, box, x0, v0, final_potentials, integrator, barostat, equil_steps, prod_steps,
    x_interval=50, du_dl_interval=200, lambda_left=None, lambda_right=None):
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

    barostat: timemachine.Barostat
        barostat to be used for dynamics

    equil_steps: int
        number of equilibration steps

    prod_steps: int
        number of production steps

    x_interval: int
        how often we store coordinates. if x_interval == 0 then
        no frames are returned.

    du_dl_interval: int
        how often we store du_dls. if du_dl_interval == 0 then
        no du_dls are returned

    lambda_left: float or None
        lhs lambda value

    lambda_right: float or None
        rhs lambda value

    Returns
    -------
    SimulationResult
        Results of the simulation.

    """

    all_impls = []

    # set up observables for du_dps here as well.
    du_dp_obs = []

    for bp in final_potentials:
        impl = bp.bound_impl(np.float32)
        all_impls.append(impl)
        du_dp_obs.append(custom_ops.AvgPartialUPartialParam(impl, 5))

    # fire minimize once again, needed for parameter interpolation
    x0 = minimizer.fire_minimize(x0, all_impls, box, np.ones(100, dtype=np.float64)*lamb)

    # sanity check that forces are well behaved
    for bp in all_impls:
        du_dx, du_dl, u = bp.execute(x0, box, lamb)
        norm_forces = np.linalg.norm(du_dx, axis=1)
        assert np.all(norm_forces < 25000)

    if integrator.seed == 0:
        # this deepcopy is needed if we're running if client == None
        integrator = copy.deepcopy(integrator)
        integrator.seed = np.random.randint(np.iinfo(np.int32).max)

    if barostat.seed == 0:
        barostat = copy.deepcopy(barostat)
        barostat.seed = np.random.randint(np.iinfo(np.int32).max)

    intg_impl = integrator.impl()
     # technically we need to only pass in the nonbonded impl
    barostat_impl = barostat.impl(all_impls)
    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg_impl,
        all_impls,
        barostat_impl
    )

    # equilibration
    equil_schedule = np.ones(equil_steps)*lamb
    ctxt.multiple_steps(equil_schedule)

    for obs in du_dp_obs:
        ctxt.add_observable(obs)

    prod_schedule = np.ones(prod_steps)*lamb

    full_du_dls, xs, boxes, = ctxt.multiple_steps(prod_schedule, du_dl_interval, x_interval)

    # keep the structure of grads the same as that of final_potentials so we can properly
    # form their vjps.
    grads = []
    for obs in du_dp_obs:
        grads.append(obs.avg_du_dp())

    # gather delta_Us
    unbound_impls = []
    unbound_params = []
    for bp in final_potentials:
        impl = bp.unbound_impl(np.float32)
        unbound_impls.append(impl)
        unbound_params.append(bp.params)

    if lambda_left is not None or lambda_right is not None:
        center_Us = []
        for x, b in zip(xs, boxes):
            u = 0
            for ub, p in zip(unbound_impls, unbound_params):
                _, _, _, nrg = ub.execute_selective(x, p, b, lamb, False, False, False, True)
                u += nrg
            center_Us.append(u)

    if lambda_left is not None:
        left_Us = []
        for x, b in zip(xs, boxes):
            u = 0
            for ub, p in zip(unbound_impls, unbound_params):
                _, _, _, nrg = ub.execute_selective(x, p, b, lambda_left, False, False, False, True)
                u += nrg
            left_Us.append(u)
        left_dus = []
        for lu, cu in zip(left_Us, center_Us):
            left_dus.append(lu - cu)
    else:
        left_dus = None

    if lambda_right is not None:
        right_Us = []
        for x, b in zip(xs, boxes):
            u = 0
            for ub, p in zip(unbound_impls, unbound_params):
                _, _, _, nrg = ub.execute_selective(x, p, b, lambda_right, False, False, False, True)
                u += nrg
            right_Us.append(u)
        right_dus = []
        for ru, cu in zip(right_Us, center_Us):
            right_dus.append(ru - cu)
    else:
        right_dus = None

    result = SimulationResult(
        xs=xs,
        boxes=boxes,
        du_dls=full_du_dls,
        du_dps=grads,
        left_dus=np.array(left_dus),
        right_dus=np.array(right_dus)
    )

    return result


FreeEnergyModel = namedtuple(
    "FreeEnergyModel",
    [
     "unbound_potentials",
     "endpoint_correct",
     "client",
     "box",
     "x0",
     "v0",
     "integrator",
     "barostat",
     "lambda_schedule",
     "equil_steps",
     "prod_steps",
     "beta",
     "prefix",
    ]
)

gradient = List[Any] # TODO: make this more descriptive of dG_grad structure

def _deltaG(model, sys_params) -> Tuple[Tuple[float, List], np.array]:

    assert len(sys_params) == len(model.unbound_potentials)

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, model.unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    all_args = []
    for lamb_idx, lamb in enumerate(model.lambda_schedule):

        if model.endpoint_correct and lamb_idx == len(model.lambda_schedule) - 1:
            x_interval = 200
        else:
            x_interval = 200

        if lamb_idx == 0:
            lambda_left = None
        else:
            lambda_left = model.lambda_schedule[lamb_idx-1]

        if lamb_idx == len(model.lambda_schedule) - 1:
            lambda_right = None
        else:
            lambda_right = model.lambda_schedule[lamb_idx+1]

        all_args.append((
            lamb,
            model.box,
            model.x0,
            model.v0,
            bound_potentials,
            model.integrator,
            model.barostat,
            model.equil_steps,
            model.prod_steps,
            x_interval,
            200,
            lambda_left,
            lambda_right
        ))

    if model.endpoint_correct:
        all_args.append((
            1.0,
            model.box,
            model.x0,
            model.v0,
            bound_potentials[:-1],
            model.integrator,
            model.barostat,
            model.equil_steps,
            model.prod_steps,
            200,
            200,
            None,
            None
        ))

    if model.client is None:
        results = []
        for args in all_args:
            results.append(simulate(*args))
    else:
        futures = []
        for args in all_args:
            futures.append(model.client.submit(simulate, *args))

        results = []
        for future in futures:
            results.append(future.result())

    mean_du_dls = []
    all_grads = []

    if model.endpoint_correct:
        ti_results = results[:-1]
    else:
        ti_results = results

    for lambda_idx, (lambda_window, result) in enumerate(zip(model.lambda_schedule, ti_results)):
        # (ytz): figure out what to do with stddev(du_dl) later
        print(f"{model.prefix} index {lambda_idx} lambda {lambda_window:.5f} <du/dl> {np.mean(result.du_dls):.5f} med(du/dl) {np.median(result.du_dls):.5f}  o(du/dl) {np.std(result.du_dls):.5f}")
        mean_du_dls.append(np.mean(result.du_dls))
        all_grads.append(result.du_dps)

    tibar_dG = 0

    bar_dG = 0
    for lambda_idx in range(len(model.lambda_schedule) - 1):
        # tibar
        lamb_start = model.lambda_schedule[lambda_idx]
        lamb_end = model.lambda_schedule[lambda_idx+1]
        delta_lamb = lamb_end - lamb_start
        fwd_work = ti_results[lambda_idx].du_dls*delta_lamb
        rev_work = -ti_results[lambda_idx+1].du_dls*delta_lamb
        tibar, ti_bar_err = pymbar.BAR(model.beta*fwd_work, model.beta*rev_work)
        tibar_overlap = endpoint_correction.overlap_from_cdf(fwd_work, rev_work)
        tibar_dG += tibar/model.beta

        # exact_bar
        fwd_work_exact = model.beta*ti_results[lambda_idx].right_dus
        rev_work_exact = model.beta*ti_results[lambda_idx+1].left_dus

        dG_exact, exact_bar_err = pymbar.BAR(fwd_work_exact, rev_work_exact)
        bar_dG += dG_exact/model.beta
        exact_bar_overlap = endpoint_correction.overlap_from_cdf(fwd_work_exact, -rev_work_exact)

        print("BAR: lamb_start", lamb_start, "ti_bar", tibar/model.beta, "exact_bar", dG_exact/model.beta, "tibar_overlap", tibar_overlap, "exact_bar_overlap", exact_bar_overlap, "ti_bar_err", ti_bar_err/model.beta, "exact_bar_err", exact_bar_err/model.beta)

    dG = np.trapz(mean_du_dls, model.lambda_schedule)
    dG_grad = []
    for rhs, lhs in zip(all_grads[-1], all_grads[0]):
        dG_grad.append(rhs - lhs)

    if model.endpoint_correct:
        core_restr = bound_potentials[-1]
        # (ytz): tbd, automatically find optimal k_translation/k_rotation such that
        # standard deviation and/or overlap is maximized
        k_translation = 200.0
        k_rotation = 100.0
        start = time.time()
        lhs_du, rhs_du, rotation_samples, translation_samples = endpoint_correction.estimate_delta_us(
            k_translation=k_translation,
            k_rotation=k_rotation,
            core_idxs=core_restr.get_idxs(),
            core_params=core_restr.params.reshape((-1,2)),
            beta=model.beta,
            lhs_xs=results[-2].xs,
            rhs_xs=results[-1].xs
        )
        dG_endpoint = pymbar.BAR(model.beta*lhs_du, model.beta*np.array(rhs_du))[0]/model.beta
        # compute standard state corrections for translation and rotation
        dG_ssc_translation, dG_ssc_rotation = standard_state.release_orientational_restraints(
            k_translation,
            k_rotation,
            model.beta
        )
        overlap = endpoint_correction.overlap_from_cdf(lhs_du, rhs_du)
        lhs_mean = np.mean(lhs_du)
        rhs_mean = np.mean(rhs_du)
        print(f"{model.prefix} dG_ti {dG:.3f} tibar_dG {tibar_dG:.3f} exact_bar {bar_dG:.3f} dG_endpoint {dG_endpoint:.3f} dG_ssc_translation {dG_ssc_translation:.3f} dG_ssc_rotation {dG_ssc_rotation:.3f} overlap {overlap:.3f} lhs_mean {lhs_mean:.3f} rhs_mean {rhs_mean:.3f} lhs_n {len(lhs_du)} rhs_n {len(rhs_du)} | time: {time.time()-start:.3f}s")
        dG += dG_endpoint + dG_ssc_translation + dG_ssc_rotation
    else:
        print(f"{model.prefix} dG_ti {dG:.3f} tibar_dG {tibar_dG:.3f} exact_bar {bar_dG:.3f} ")

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