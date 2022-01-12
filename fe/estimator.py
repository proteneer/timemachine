from collections import namedtuple

import copy
import numpy as np

from timemachine.lib import potentials, custom_ops

from typing import Tuple, List

import dataclasses

from parallel.client import SerialClient
from md.states import CoordsVelBox


@dataclasses.dataclass
class SimulationResult:
    xs: np.array
    du_dls: np.array


def equilibrate(integrator, barostat, potentials, coords, box, lamb, equil_steps) -> Tuple:
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


def simulate(
    lamb,
    box,
    x0,
    v0,
    final_potentials,
    integrator,
    equil_steps,
    prod_steps,
    barostat,
    x_interval=1000,
    du_dl_interval=5,
) -> SimulationResult:
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

    x_interval: int
        how often we store coordinates. if x_interval == 0 then
        no frames are returned.

    du_dl_interval: int
        how often we store du_dls. if du_dl_interval == 0 then
        no du_dls are returned

    barostat: timemachine.lib.MonteCarloBarostat
        Monte carlo barostat to use when simulating.

    Returns
    -------
    SimulationResult
        Results of the simulation.

    """
    all_impls = []
    bonded_impls = []
    nonbonded_impls = []

    for bp in final_potentials:
        impl = bp.bound_impl(np.float32)
        if isinstance(bp, potentials.Nonbonded):
            nonbonded_impls.append(impl)
        else:
            bonded_impls.append(impl)
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
        x0,
        v0,
        box,
        intg_impl,
        all_impls,
        barostat=baro_impl,
    )
    base_interval = baro_impl.get_interval()
    # Use an interval of 5 for equilibration
    baro_impl.set_interval(5)

    # equilibration
    equil_schedule = np.ones(equil_steps) * lamb
    ctxt.multiple_steps(equil_schedule)

    baro_impl.set_interval(base_interval)

    prod_schedule = np.ones(prod_steps) * lamb

    full_du_dls, xs, _ = ctxt.multiple_steps(prod_schedule, du_dl_interval, x_interval)

    result = SimulationResult(xs=xs.astype("float32"), du_dls=full_du_dls)
    return result


FreeEnergyModel = namedtuple(
    "FreeEnergyModel",
    [
        "unbound_potentials",
        "client",
        "box",
        "x0",
        "v0",
        "integrator",
        "lambda_schedule",
        "equil_steps",
        "prod_steps",
        "barostat",
    ],
)


def deltaG(model, sys_params) -> Tuple[float, List]:

    assert len(sys_params) == len(model.unbound_potentials)

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, model.unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    client = model.client
    if client is None:
        client = SerialClient()
        client.verify()

    futures = []
    for lamb in model.lambda_schedule:
        args = (
            lamb,
            model.box,
            model.x0,
            model.v0,
            bound_potentials,
            model.integrator,
            model.equil_steps,
            model.prod_steps,
            model.barostat,
        )
        kwargs = {}  # Unused for now
        futures.append(client.submit(simulate, *args, **kwargs))

    results = [x.result() for x in futures]

    mean_du_dls = []

    for result in results:
        # (ytz): figure out what to do with stddev(du_dl) later
        mean_du_dls.append(np.mean(result.du_dls))

    dG = np.trapz(mean_du_dls, model.lambda_schedule)

    return dG, results
