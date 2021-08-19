import pymbar
from fe import endpoint_correction
from collections import namedtuple
import pickle

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
from fe.utils import sanitize_energies, extract_delta_Us_from_U_knk

from timemachine.lib import potentials, custom_ops

@dataclasses.dataclass
class SimulationResult:
   xs: np.array
   boxes: np.array
   du_dps: np.array
   lambda_us: np.array

def flatten(v):
    return tuple(), (v.xs, v.boxes, v.du_dps, v.lambda_us)

def unflatten(aux_data, children):
    xs, boxes, du_dps, lambda_us = aux_data
    return SimulationResult(xs, boxes, du_dps, lambda_us)

jax.tree_util.register_pytree_node(SimulationResult, flatten, unflatten)

def simulate(lamb, box, x0, v0, final_potentials, integrator, barostat, equil_steps, prod_steps,
    x_interval, u_interval, lambda_windows):
    """
    Run a simulation and collect relevant statistics for this simulation.

    Parameters
    ----------
    lamb: float
        lambda value used for the equilibrium simulation

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
        barostat to be used for equilibration

    equil_steps: int
        number of equilibration steps

    prod_steps: int
        number of production steps

    x_interval: int
        how often we store coordinates. If x_interval == 0 then
        no frames are returned.

    u_interval: int
        how often we store energies. If u_interval == 0 then
        no energies are returned

    lambda_windows: list of float
        lambda windows we evaluate energies at.

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
        du_dp_obs.append(custom_ops.AvgPartialUPartialParam(impl, 25))

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

    # (ytz): intentionally hard-coded, I'd rather the end-user *not*
    # muck with this unless they have a good reason to.
    barostat_impl.set_interval(25)

    for obs in du_dp_obs:
        ctxt.add_observable(obs)

    full_us, xs, boxes = ctxt.multiple_steps_U(
        lamb,
        prod_steps,
        np.array(lambda_windows),
        u_interval,
        x_interval
    )

    # keep the structure of grads the same as that of final_potentials so we can properly
    # form their vjps.
    grads = []
    for obs in du_dp_obs:
        grads.append(obs.avg_du_dp())

    result = SimulationResult(
        xs=xs,
        boxes=boxes,
        du_dps=grads,
        lambda_us=full_us,
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

        subsample_interval = 1000

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
            subsample_interval,
            subsample_interval, 
            model.lambda_schedule
        ))

    if model.endpoint_correct:

        assert isinstance(bound_potentials[-1], potentials.HarmonicBond)

        all_args.append((
            1.0,
            model.box,
            model.x0,
            model.v0,
            bound_potentials[:-1], # strip out the restraints
            model.integrator,
            model.barostat,
            model.equil_steps,
            model.prod_steps,
            subsample_interval,
            subsample_interval, 
            [] # no need to evaluate Us for the endpoint correction
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

    if model.endpoint_correct:
        sim_results = results[:-1]
    else:
        sim_results = results

    U_knk = []
    N_k = []
    for lambda_idx, (lambda_window, result) in enumerate(zip(model.lambda_schedule, sim_results)):
        U_knk.append(result.lambda_us)
        N_k.append(len(result.lambda_us)) # number of frames

    U_knk = np.array(U_knk)

    bar_dG = 0
    bar_dG_err = 0

    delta_Us = extract_delta_Us_from_U_knk(U_knk)

    for lambda_idx in range(len(model.lambda_schedule) - 1):

        fwd_delta_u = model.beta*delta_Us[lambda_idx][0]
        rev_delta_u = model.beta*delta_Us[lambda_idx][1]

        dG_exact, exact_bar_err = pymbar.BAR(fwd_delta_u, rev_delta_u)
        bar_dG += dG_exact/model.beta
        exact_bar_overlap = endpoint_correction.overlap_from_cdf(fwd_delta_u, rev_delta_u)

        # probably off by a factor of two since we re-use samples.
        bar_dG_err += (exact_bar_err/model.beta)**2

        lamb_start = model.lambda_schedule[lambda_idx]
        lamb_end = model.lambda_schedule[lambda_idx+1]

        print(f"{model.prefix}_BAR: lambda {lamb_start:.3f} -> {lamb_end:.3f} dG: {dG_exact/model.beta:.3f} dG_err: {exact_bar_err/model.beta:.3f} overlap: {exact_bar_overlap:.3f}")

    # for MBAR we need to sanitize the energies
    clean_U_knks = [] # [K, F, K]
    for lambda_idx, full_us in enumerate(U_knk):
        clean_U_knks.append(sanitize_energies(full_us, lambda_idx))

    print(model.prefix, " MBAR: amin", np.amin(clean_U_knks), "median", np.median(clean_U_knks), "max", np.amax(clean_U_knks))

    K = len(model.lambda_schedule)
    clean_U_knks = np.array(clean_U_knks) # [K, F, K]
    U_kn = np.reshape(clean_U_knks, (-1, K)).transpose() # [K, F*K]
    u_kn = U_kn*model.beta

    np.save(model.prefix+"_U_kn.npy", U_kn)

    mbar = pymbar.MBAR(u_kn, N_k)
    differences, error_estimates = mbar.getFreeEnergyDifferences()
    f_k, error_k = differences[0], error_estimates[0]
    mbar_dG = f_k[-1]/model.beta
    mbar_dG_err = error_k[-1]/model.beta

    bar_dG_err = np.sqrt(bar_dG_err)

    dG = bar_dG # use the exact answer
    dG_grad = []

    # (ytz): results[-1].du_dps contain system parameter derivatives for the
    # independent, gas phase simulation. They're usually ordered as:
    # [Bonds, Angles, Torsions, Nonbonded]
    #
    # results[0].du_dps contain system parameter derivatives for the core
    # restrained state. If we're doing the endpoint correction during
    # decoupling stages, the derivatives are ordered as:

    # [Bonds, Angles, Torsions, Nonbonded, RestraintBonds]
    # Otherwise, in stages like conversion where the endpoint correction
    # is turned off, the derivatives are ordered as :
    # [Bonds, Angles, Torsions, Nonbonded]

    # Note that this zip will always loop over only the
    # [Bonds, Angles, Torsions, Nonbonded] terms, since it only
    # enumerates over the smaller of the two lists.
    for rhs, lhs in zip(results[-1].du_dps, results[0].du_dps):
        dG_grad.append(rhs - lhs)

    if model.endpoint_correct:
        assert len(results[0].du_dps) - len(results[-1].du_dps) == 1
        # (ytz): Fill in missing derivatives since zip() from above loops
        # over the shorter array.
        lhs = results[0].du_dps[-1]
        rhs = 0 # zero as the energies do not depend the core restraints.
        dG_grad.append(rhs - lhs)

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
        dG_endpoint, endpoint_err = pymbar.BAR(model.beta*lhs_du, model.beta*np.array(rhs_du))
        dG_endpoint = dG_endpoint/model.beta
        endpoint_err = endpoint_err/model.beta
        # compute standard state corrections for translation and rotation
        dG_ssc_translation, dG_ssc_rotation = standard_state.release_orientational_restraints(
            k_translation,
            k_rotation,
            model.beta
        )
        overlap = endpoint_correction.overlap_from_cdf(lhs_du, rhs_du)
        lhs_mean = np.mean(lhs_du)
        rhs_mean = np.mean(rhs_du)
        print(f"{model.prefix} bar (A) {bar_dG:.3f} bar_err {bar_dG_err:.3f} mbar (A) {mbar_dG:.3f} mbar_err {mbar_dG_err:.3f} dG_endpoint (E) {dG_endpoint:.3f} dG_endpoint_err {endpoint_err:.3f} dG_ssc_translation {dG_ssc_translation:.3f} dG_ssc_rotation {dG_ssc_rotation:.3f} overlap {overlap:.3f} lhs_mean {lhs_mean:.3f} rhs_mean {rhs_mean:.3f} lhs_n {len(lhs_du)} rhs_n {len(rhs_du)} | time: {time.time()-start:.3f}s")
        dG += dG_endpoint + dG_ssc_translation + dG_ssc_rotation
        bar_dG_err = np.sqrt(bar_dG_err**2 + endpoint_err**2)
    else:
        print(f"{model.prefix} bar (A) {bar_dG:.3f} bar_err {bar_dG_err:.3f} mbar (A) {mbar_dG:.3f} mbar_err {mbar_dG_err:.3f} ")

    return (dG, bar_dG_err, results), dG_grad

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
    # grad[1] is the adjoint of dG_err w.r.t. loss: which we don't use
    # grad[2] is the adjoint of simulation results w.r.t. loss: which we don't use
    return ([grad[0]*r for r in residual],)

deltaG.defvjp(deltaG_fwd, deltaG_bwd)
