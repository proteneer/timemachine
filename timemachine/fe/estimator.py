import copy
import dataclasses
import time
from typing import List, Optional, Tuple

import numpy as np
import pymbar
from numpy.typing import NDArray

from timemachine.fe import endpoint_correction, standard_state
from timemachine.fe.utils import extract_delta_Us_from_U_knk, sanitize_energies
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops, potentials
from timemachine.md import minimizer
from timemachine.md.states import CoordsVelBox
from timemachine.parallel.client import AbstractClient, SerialClient


@dataclasses.dataclass
class SimulationResult:
    xs: NDArray
    boxes: NDArray
    lambda_us: NDArray


@dataclasses.dataclass
class FreeEnergyModel:
    unbound_potentials: List
    endpoint_correct: bool
    client: Optional[AbstractClient]
    box: NDArray
    x0: NDArray
    v0: NDArray
    integrator: LangevinIntegrator
    barostat: MonteCarloBarostat
    lambda_schedule: List[float]
    equil_steps: int
    prod_steps: int
    beta: float
    prefix: str


def equilibrate(
    integrator: LangevinIntegrator,
    barostat: LangevinIntegrator,
    potentials: List,
    coords: NDArray,
    box: NDArray,
    lamb: float,
    equil_steps: int,
) -> Tuple:
    all_impls = []
    v0 = np.zeros_like(coords)

    for bp in potentials:
        impl = bp.bound_impl(np.float32)
        all_impls.append(impl)

    if integrator.seed == 0:
        integrator = copy.deepcopy(integrator)
        integrator.seed = np.random.randint(np.iinfo(np.int32).max)

    if barostat.seed == 0:
        barostat = copy.deepcopy(barostat)
        barostat.seed = np.random.randint(np.iinfo(np.int32).max)

    intg_impl = integrator.impl()
    baro_impl = barostat.impl(all_impls)
    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(
        coords,
        v0,
        box,
        intg_impl,
        all_impls,
        barostat=baro_impl,
    )

    # equilibration
    equil_schedule = np.ones(equil_steps) * lamb
    ctxt.multiple_steps(equil_schedule)
    return CoordsVelBox(coords=ctxt.get_x_t(), velocities=ctxt.get_v_t(), box=ctxt.get_box())


def run_model_simulations(model: FreeEnergyModel, sys_params: NDArray, subsample_interval: int = 1000) -> List:
    """
    Runs simulations as defined by the FreeEnergyModel using the client.

    Parameters
    ----------

    model: FreeEnergyModel
        Defines the free energy calculations to be run

    sys_params: np.array
        Defines the system's potential parameters

    subsample_interval: int
        Interval to collect energies and frames

    Returns
    -------

    A list of Futures that return SimulationResults by calling `result()`


    Note:
        If the FreeEnergyModel has `endpoint_correct` set to True then the final potential in `unbound_potentials` must be
        a HarmonicBond potential. It will also run an additional simulation to collect vacuum samples to perform endpoint correction.
    """
    assert len(sys_params) == len(model.unbound_potentials)

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, model.unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    all_args = []
    for lamb_idx, lamb in enumerate(model.lambda_schedule):

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

    if model.endpoint_correct:

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

    results = []
    client = model.client
    if client is None:
        client = SerialClient()
        client.verify()
    futures = []
    for args in all_args:
        futures.append(client.submit(simulate, *args))

    for future in futures:
        results.append(future.result())
    return results


def simulate(
    lamb: float,
    box: NDArray,
    x0: NDArray,
    v0: NDArray,
    final_potentials: List,
    integrator: LangevinIntegrator,
    barostat: MonteCarloBarostat,
    equil_steps: int,
    prod_steps: int,
    x_interval: int,
    u_interval: int,
    lambda_windows: List[float],
):
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

    for bp in final_potentials:
        impl = bp.bound_impl(np.float32)
        all_impls.append(impl)

    # fire minimize once again, needed for parameter interpolation
    x0 = minimizer.fire_minimize(x0, all_impls, box, np.ones(100, dtype=np.float64) * lamb)

    # sanity check that forces are well behaved
    for bp in all_impls:
        du_dx, du_dl, u = bp.execute(x0, box, lamb)
        norm_forces = np.linalg.norm(du_dx, axis=1)
        assert np.all(norm_forces < 25000), "Forces much greater than expected after minimization"
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
    ctxt = custom_ops.Context(x0, v0, box, intg_impl, all_impls, barostat_impl)

    # equilibration
    equil_schedule = np.ones(equil_steps) * lamb
    ctxt.multiple_steps(equil_schedule)

    # (ytz): intentionally hard-coded, I'd rather the end-user *not*
    # muck with this unless they have a good reason to.
    barostat_impl.set_interval(25)

    full_us, xs, boxes = ctxt.multiple_steps_U(lamb, prod_steps, np.array(lambda_windows), u_interval, x_interval)

    result = SimulationResult(
        xs=xs.astype("float32"),
        boxes=boxes.astype("float32"),
        lambda_us=full_us,
    )

    return result


def deltaG_from_results(
    model: FreeEnergyModel, results: List, sys_params: NDArray
) -> Tuple[float, float, List[SimulationResult]]:
    """
    Computes the deltaG from a set of results

    Parameters
    ----------

    model: FreeEnergyModel
        Defines the free energy calculations to be run

    results: List of futures that return SimulationResults

    sys_params: np.array
        Defines the system's potential parameters

    Returns
    -------

    A tuple containing the BAR delta G, the BAR delta G Error and a list of SimulationResults


    Note:
        If the FreeEnergyModel has `endpoint_correct` set to True then the final potential in `unbound_potentials` must be
        a HarmonicBond potential. Assumes the last result is a vacuum simulation that is used to compute the endpoint
        correction.
    """
    assert len(sys_params) == len(model.unbound_potentials)

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, model.unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    if model.endpoint_correct:
        sim_results = results[:-1]
    else:
        sim_results = results

    U_knk = []
    N_k = []
    for result in sim_results:
        U_knk.append(result.lambda_us)
        N_k.append(len(result.lambda_us))  # number of frames

    U_knk = np.array(U_knk)

    bar_dG = 0
    bar_dG_err = 0

    delta_Us = extract_delta_Us_from_U_knk(U_knk)

    for lambda_idx in range(len(model.lambda_schedule) - 1):

        fwd_delta_u = model.beta * delta_Us[lambda_idx][0]
        rev_delta_u = model.beta * delta_Us[lambda_idx][1]

        dG_exact, exact_bar_err = pymbar.BAR(fwd_delta_u, rev_delta_u)
        bar_dG += dG_exact / model.beta
        exact_bar_overlap = endpoint_correction.overlap_from_cdf(fwd_delta_u, rev_delta_u)

        # probably off by a factor of two since we re-use samples.
        bar_dG_err += (exact_bar_err / model.beta) ** 2

        lamb_start = model.lambda_schedule[lambda_idx]
        lamb_end = model.lambda_schedule[lambda_idx + 1]

        print(
            f"{model.prefix}_BAR: lambda {lamb_start:.3f} -> {lamb_end:.3f} dG: {dG_exact/model.beta:.3f} dG_err: {exact_bar_err/model.beta:.3f} overlap: {exact_bar_overlap:.3f}"
        )

    # for MBAR we need to sanitize the energies
    clean_U_knks = []  # [K, F, K]
    for lambda_idx, full_us in enumerate(U_knk):
        clean_U_knks.append(sanitize_energies(full_us, lambda_idx))

    print(
        model.prefix,
        " MBAR: amin",
        np.amin(clean_U_knks),
        "median",
        np.median(clean_U_knks),
        "max",
        np.amax(clean_U_knks),
    )

    K = len(model.lambda_schedule)
    clean_U_knks = np.array(clean_U_knks)  # [K, F, K]
    U_kn = np.reshape(clean_U_knks, (-1, K)).transpose()  # [K, F*K]
    u_kn = U_kn * model.beta

    np.save(model.prefix + "_U_kn.npy", U_kn)

    mbar = pymbar.MBAR(u_kn, N_k)
    differences, error_estimates = mbar.getFreeEnergyDifferences()
    f_k, error_k = differences[0], error_estimates[0]
    mbar_dG = f_k[-1] / model.beta
    mbar_dG_err = error_k[-1] / model.beta

    bar_dG_err = np.sqrt(bar_dG_err)

    dG = bar_dG  # use the exact answer

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
            core_params=core_restr.params.reshape((-1, 2)),
            beta=model.beta,
            lhs_xs=results[-2].xs,
            rhs_xs=results[-1].xs,
            seed=2021,
        )
        dG_endpoint, endpoint_err = pymbar.BAR(model.beta * lhs_du, model.beta * np.array(rhs_du))
        dG_endpoint = dG_endpoint / model.beta
        endpoint_err = endpoint_err / model.beta
        # compute standard state corrections for translation and rotation
        dG_ssc_translation, dG_ssc_rotation = standard_state.release_orientational_restraints(
            k_translation, k_rotation, model.beta
        )
        overlap = endpoint_correction.overlap_from_cdf(lhs_du, rhs_du)
        lhs_mean = np.mean(lhs_du)
        rhs_mean = np.mean(rhs_du)
        print(
            f"{model.prefix} bar (A) {bar_dG:.3f} bar_err {bar_dG_err:.3f} mbar (A) {mbar_dG:.3f} mbar_err {mbar_dG_err:.3f} dG_endpoint (E) {dG_endpoint:.3f} dG_endpoint_err {endpoint_err:.3f} dG_ssc_translation {dG_ssc_translation:.3f} dG_ssc_rotation {dG_ssc_rotation:.3f} overlap {overlap:.3f} lhs_mean {lhs_mean:.3f} rhs_mean {rhs_mean:.3f} lhs_n {len(lhs_du)} rhs_n {len(rhs_du)} | time: {time.time()-start:.3f}s"
        )
        dG += dG_endpoint + dG_ssc_translation + dG_ssc_rotation
        bar_dG_err = np.sqrt(bar_dG_err ** 2 + endpoint_err ** 2)
    else:
        print(
            f"{model.prefix} bar (A) {bar_dG:.3f} bar_err {bar_dG_err:.3f} mbar (A) {mbar_dG:.3f} mbar_err {mbar_dG_err:.3f} "
        )

    return dG, bar_dG_err, results


def deltaG(
    model: FreeEnergyModel, sys_params: NDArray, subsample_interval: int = 1000
) -> Tuple[float, float, List[SimulationResult]]:
    """
    Computes the delta G of a FreeEnergyModel

    Parameters
    ----------

    model: FreeEnergyModel
        Defines the free energy calculations

    sys_params: np.array
        Defines the system's potential parameters

    subsample_interval: int
        Interval to collect energies and frames

    Returns
    -------

    A tuple containing the BAR delta G, the BAR delta G Error and a list of SimulationResults


    Note:
        If the FreeEnergyModel has `endpoint_correct` set to True then the final potential in `unbound_potentials` must be
        a HarmonicBond potential. It will also run an additional simulation to collect vacuum samples to perform endpoint correction.
    """
    results = run_model_simulations(model, sys_params, subsample_interval=subsample_interval)
    return deltaG_from_results(model=model, results=results, sys_params=sys_params)
