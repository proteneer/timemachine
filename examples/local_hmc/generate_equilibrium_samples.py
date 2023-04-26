# adapted from proteneer/timemachine/tests/test_benchmark.py

import time

import numpy as np

from timemachine import constants
from timemachine.fe.model_utils import apply_hmr
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.testsystems.dhfr import get_dhfr_system, setup_dhfr

dhfr_system = get_dhfr_system()
host_fns, host_masses, host_conf, box = setup_dhfr()

SECONDS_PER_DAY = 24 * 60 * 60


def simulate(x0, v0, box0, masses, bound_potentials, thinning=1_000, n_samples=1_000, verbose=True):
    seed = 1234

    temperature = constants.DEFAULT_TEMP
    pressure = constants.DEFAULT_PRESSURE

    harmonic_bond_potential = bound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential.potential)

    hmr = True
    if hmr:
        dt = 2.5e-3
        masses = apply_hmr(masses, bond_list)
    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(masses), seed).impl()

    bps = []

    for potential in bound_potentials:
        bps.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    baro_impl = None
    barostat_interval = 15
    if barostat_interval > 0:
        group_idxs = get_group_indices(bond_list)
        baro = MonteCarloBarostat(
            x0.shape[0],
            pressure,
            temperature,
            group_idxs,
            barostat_interval,
            seed,
        )
        baro_impl = baro.impl(bps)

    ctxt = custom_ops.Context(
        x0,
        v0,
        box0,
        intg,
        bps,
        barostat=baro_impl,
    )

    batch_times = []

    # run once before timer starts
    ctxt.multiple_steps(thinning)

    start = time.time()

    xs = []
    boxes = []

    for _ in range(n_samples):

        # time the current batch
        batch_start = time.time()
        _, _ = ctxt.multiple_steps(thinning, store_x_interval=0)
        batch_end = time.time()

        xs.append(ctxt.get_x_t())
        boxes.append(ctxt.get_box())

        delta = batch_end - batch_start

        batch_times.append(delta)

        steps_per_second = thinning / np.mean(batch_times)
        steps_per_day = steps_per_second * SECONDS_PER_DAY

        ps_per_day = dt * steps_per_day
        ns_per_day = ps_per_day * 1e-3

        if verbose:
            print(f"steps per second: {steps_per_second:.3f}")
            print(f"ns per day: {ns_per_day:.3f}")

    assert np.all(np.abs(ctxt.get_x_t()) < 1000)

    print(
        f"N={x0.shape[0]} speed: {ns_per_day:.2f}ns/day dt: {dt * 1e3}fs (ran {thinning * n_samples} steps in {(time.time() - start):.2f}s)"
    )

    return np.array(xs), np.array(boxes)


if __name__ == "__main__":
    host_fns, host_masses, host_conf, box = setup_dhfr()
    xs, boxes = simulate(host_conf, np.zeros_like(host_conf), box, host_masses, host_fns)
    np.savez("dhfr_langevin_samples.npz", xs=xs, boxes=boxes)
