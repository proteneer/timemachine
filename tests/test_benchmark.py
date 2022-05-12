"""Run vanilla "apo" MD on DHFR and HIF2A test systems,
and running an intermediate lambda window "rbfe" MD for a
relative binding free energy edge from the HIF2A test system"""

import time

import numpy as np
import pytest

from timemachine import constants
from timemachine.fe.model_utils import apply_hmr
from timemachine.fe.utils import to_md_units
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.lib.potentials import (
    Nonbonded,
    NonbondedInteractionGroup,
    NonbondedInteractionGroupInterpolated,
    NonbondedInterpolated,
)
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.testsystems.dhfr import setup_dhfr
from timemachine.testsystems.relative import setup_hif2a_ligand_pair


@pytest.fixture(scope="module")
def hi2fa_test_frames():
    return generate_hif2a_frames(100, 1, seed=2022)


def generate_hif2a_frames(n_frames: int, frame_interval: int, seed=None, barostat_interval: int = 5, hmr: bool = True):
    rfe = setup_hif2a_ligand_pair("smirnoff_1_1_0_ccc.py")
    mol_a, mol_b = rfe.mol_a, rfe.mol_b

    # build the protein system.
    host_system, host_coords, _, _, host_box, _ = builders.build_protein_system("tests/data/hif2a_nowater_min.pdb")

    unbound_potentials, sys_params, masses = rfe.prepare_host_edge(rfe.ff.get_ordered_params(), host_system)
    min_host_coords = minimizer.minimize_host_4d([mol_a, mol_b], host_system, host_coords, rfe.ff, host_box)
    x0 = rfe.prepare_combined_coords(min_host_coords)
    ligand_idxs = np.arange(len(host_coords), len(x0), dtype=np.int32)

    temperature = 300
    pressure = 1.0
    lamb = 0.0

    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    if hmr:
        dt = 2.5e-3
        masses = apply_hmr(masses, bond_list)
    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)
    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(masses), seed).impl()

    bps = []

    for potential, params in zip(unbound_potentials, sys_params):
        bps.append(potential.bind(params).bound_impl(precision=np.float32))  # get the bound implementation

    baro_impl = None
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
        np.zeros_like(x0),
        host_box,
        intg,
        bps,
        barostat=baro_impl,
    )
    steps = n_frames * frame_interval
    _, coords, boxes = ctxt.multiple_steps(np.ones(steps) * lamb, 0, frame_interval)
    assert coords.shape[0] == n_frames, f"Got {coords.shape[0]} frames, expected {n_frames}"
    return unbound_potentials, sys_params, coords, boxes, ligand_idxs


def benchmark_potential(
    label,
    potential,
    precision,
    params,
    coords,
    boxes,
    lambdas,
    verbose=True,
    num_batches=5,
    compute_du_dx=True,
    compute_du_dp=True,
    compute_du_dl=True,
    compute_u=True,
):
    if precision == np.float32:
        label = label + "_f32"
    else:
        label = label + "_f64"
    unbound = potential.unbound_impl(precision=precision)
    start = time.time()
    batch_times = []
    frames = coords.shape[0]
    param_batches = params.shape[0]
    num_lambs = len(lambdas)
    runs_per_batch = frames * param_batches * num_lambs
    for _ in range(num_batches):
        batch_start = time.time()
        _, _, _, _ = unbound.execute_selective_batch(
            coords,
            params,
            boxes,
            lambdas,
            compute_du_dx,
            compute_du_dp,
            compute_du_dl,
            compute_u,
        )
        batch_end = time.time()
        delta = batch_end - batch_start

        batch_times.append(delta)
        runs_per_second = runs_per_batch / np.mean(batch_times)

        if verbose:
            print(f"executions per second: {runs_per_second:.3f}")
    print(
        f"{label}: N={coords.shape[1]} Frames={frames} Params={param_batches} Lambdas={num_lambs} speed: {runs_per_second:.2f} executions/seconds (ran {runs_per_batch * num_batches} potentials in {(time.time() - start):.2f}s)"
    )


def benchmark(
    label,
    masses,
    lamb,
    x0,
    v0,
    box,
    bound_potentials,
    hmr=False,
    verbose=True,
    num_batches=100,
    steps_per_batch=1000,
    compute_du_dl_interval=0,
    barostat_interval=0,
):
    """
    TODO: configuration blob containing num_batches, steps_per_batch, and any other options
    """

    seed = 1234
    dt = 1.5e-3
    temperature = 300
    pressure = 1.0
    seconds_per_day = 86400

    harmonic_bond_potential = bound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    if hmr:
        dt = 2.5e-3
        masses = apply_hmr(masses, bond_list)
    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(masses), seed).impl()

    bps = []

    for potential in bound_potentials:
        bps.append(potential.bound_impl(precision=np.float32))  # get the bound implementation

    baro_impl = None
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
        box,
        intg,
        bps,
        barostat=baro_impl,
    )

    batch_times = []

    lambda_schedule = np.ones(steps_per_batch) * lamb

    # run once before timer starts
    ctxt.multiple_steps(lambda_schedule, compute_du_dl_interval)

    start = time.time()

    for batch in range(num_batches):

        # time the current batch
        batch_start = time.time()
        du_dls, _, _ = ctxt.multiple_steps(lambda_schedule, compute_du_dl_interval)
        batch_end = time.time()

        delta = batch_end - batch_start

        batch_times.append(delta)

        steps_per_second = steps_per_batch / np.mean(batch_times)
        steps_per_day = steps_per_second * seconds_per_day

        ps_per_day = dt * steps_per_day
        ns_per_day = ps_per_day * 1e-3

        if verbose:
            print(f"steps per second: {steps_per_second:.3f}")
            print(f"ns per day: {ns_per_day:.3f}")

    assert np.all(np.abs(ctxt.get_x_t()) < 1000)

    print(
        f"{label}: N={x0.shape[0]} speed: {ns_per_day:.2f}ns/day dt: {dt*1e3}fs (ran {steps_per_batch * num_batches} steps in {(time.time() - start):.2f}s)"
    )


def benchmark_dhfr(verbose=False, num_batches=100, steps_per_batch=1000):

    host_fns, host_masses, host_coords, box = setup_dhfr()

    host_conf = []
    for x, y, z in host_coords:
        host_conf.append([to_md_units(x), to_md_units(y), to_md_units(z)])
    host_conf = np.array(host_conf)

    x0 = host_conf
    v0 = np.zeros_like(host_conf)

    benchmark(
        "dhfr-apo",
        host_masses,
        0.0,
        x0,
        v0,
        box,
        host_fns,
        verbose=verbose,
        num_batches=num_batches,
        steps_per_batch=steps_per_batch,
    )
    benchmark(
        "dhfr-apo-barostat-interval-25",
        host_masses,
        0.0,
        x0,
        v0,
        box,
        host_fns,
        verbose=verbose,
        num_batches=num_batches,
        steps_per_batch=steps_per_batch,
        barostat_interval=25,
    )
    benchmark(
        "dhfr-apo-hmr-barostat-interval-25",
        host_masses,
        0.0,
        x0,
        v0,
        box,
        host_fns,
        verbose=verbose,
        hmr=True,
        num_batches=num_batches,
        steps_per_batch=steps_per_batch,
        barostat_interval=25,
    )


def benchmark_hif2a(verbose=False, num_batches=100, steps_per_batch=1000):

    rfe = setup_hif2a_ligand_pair("smirnoff_1_1_0_sc.py")
    mol_a, mol_b = rfe.mol_a, rfe.mol_b

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
        "tests/data/hif2a_nowater_min.pdb"
    )

    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

    for stage, host_system, host_coords, host_box in [
        ("hif2a", complex_system, complex_coords, complex_box),
        ("solvent", solvent_system, solvent_coords, solvent_box),
    ]:

        host_fns, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.0)

        # resolve host clashes
        min_host_coords = minimizer.minimize_host_4d([mol_a, mol_b], host_system, host_coords, rfe.ff, host_box)

        x0 = min_host_coords
        v0 = np.zeros_like(x0)

        # lamb = 0.0
        benchmark(
            stage + "-apo",
            host_masses,
            0.0,
            x0,
            v0,
            host_box,
            host_fns,
            verbose=verbose,
            num_batches=num_batches,
            steps_per_batch=steps_per_batch,
        )
        benchmark(
            stage + "-apo-barostat-interval-25",
            host_masses,
            0.0,
            x0,
            v0,
            host_box,
            host_fns,
            verbose=verbose,
            num_batches=num_batches,
            steps_per_batch=steps_per_batch,
            barostat_interval=25,
        )

        # RBFE
        unbound_potentials, sys_params, masses = rfe.prepare_host_edge(rfe.ff.get_ordered_params(), host_system)
        combined_coords = rfe.prepare_combined_coords(x0)
        bound_potentials = [x.bind(y) for (x, y) in zip(unbound_potentials, sys_params)]

        x0 = combined_coords
        v0 = np.zeros_like(x0)

        # lamb = 0.5
        benchmark(
            stage + "-rbfe-with-du-dp",
            masses,
            0.5,
            x0,
            v0,
            host_box,
            bound_potentials,
            verbose=verbose,
            num_batches=num_batches,
            steps_per_batch=steps_per_batch,
        )

        for du_dl_interval in [0, 1, 5]:
            benchmark(
                stage + "-rbfe-du-dl-interval-" + str(du_dl_interval),
                masses,
                0.5,
                x0,
                v0,
                host_box,
                bound_potentials,
                verbose=verbose,
                num_batches=num_batches,
                steps_per_batch=steps_per_batch,
                compute_du_dl_interval=du_dl_interval,
            )


def test_dhfr():
    benchmark_dhfr(verbose=True, num_batches=2, steps_per_batch=100)


def test_hif2a():
    benchmark_hif2a(verbose=True, num_batches=2, steps_per_batch=100)


def test_nonbonded_interaction_group_potential(hi2fa_test_frames):
    pots, sys_params, frames, boxes, ligand_idxs = hi2fa_test_frames
    lambdas = np.array([0.0, 1.0])
    nonbonded_interp = pots[3]
    assert isinstance(nonbonded_interp, NonbondedInterpolated)
    num_param_batches = 5
    beta = 1 / (constants.BOLTZ * 300)
    cutoff = 1.2

    precisions = [np.float32, np.float64]
    nonbonded_params = np.stack([sys_params[3]] * num_param_batches)

    potential = NonbondedInteractionGroup(
        ligand_idxs,
        nonbonded_interp.get_lambda_plane_idxs(),
        nonbonded_interp.get_lambda_offset_idxs(),
        beta,
        cutoff,
    )
    class_name = potential.__class__.__name__
    for precision in precisions:
        benchmark_potential(
            class_name,
            potential,
            precision,
            nonbonded_params[:, : nonbonded_params.shape[1] // 2],
            frames,
            boxes,
            lambdas,
            verbose=False,
        )

    potential = NonbondedInteractionGroupInterpolated(
        ligand_idxs,
        nonbonded_interp.get_lambda_plane_idxs(),
        nonbonded_interp.get_lambda_offset_idxs(),
        beta,
        cutoff,
    )
    class_name = potential.__class__.__name__
    for precision in precisions:
        benchmark_potential(
            class_name,
            potential,
            precision,
            nonbonded_params,
            frames,
            boxes,
            lambdas,
            verbose=False,
        )


def test_nonbonded_potential(hi2fa_test_frames):
    pots, sys_params, frames, boxes, _ = hi2fa_test_frames

    nonbonded_interp = pots[3]
    assert isinstance(nonbonded_interp, NonbondedInterpolated)
    lambdas = np.array([0.0, 1.0])

    num_param_batches = 5

    nonbonded_params = np.stack([sys_params[3]] * num_param_batches)

    precisions = [np.float32, np.float64]

    potential = Nonbonded(
        nonbonded_interp.get_exclusion_idxs(),
        nonbonded_interp.get_scale_factors(),
        nonbonded_interp.get_lambda_plane_idxs(),
        nonbonded_interp.get_lambda_offset_idxs(),
        nonbonded_interp.get_beta(),
        nonbonded_interp.get_cutoff(),
    )

    class_name = potential.__class__.__name__
    for precision in precisions:
        benchmark_potential(
            class_name,
            potential,
            precision,
            nonbonded_params[:, : nonbonded_params.shape[1] // 2],
            frames,
            boxes,
            lambdas,
            verbose=False,
        )

    class_name = nonbonded_interp.__class__.__name__

    for precision in precisions:
        benchmark_potential(
            class_name,
            nonbonded_interp,
            precision,
            nonbonded_params,
            frames,
            boxes,
            lambdas,
            verbose=False,
        )


def test_bonded_potentials(hi2fa_test_frames):
    pots, sys_params, frames, boxes, _ = hi2fa_test_frames

    lambdas = np.array([0.0, 1.0])
    # Drop the nonbonded potential
    for potential, params in zip(pots[:-1], sys_params):
        class_name = potential.__class__.__name__
        params = np.expand_dims(params, axis=0)
        for precision in [np.float32, np.float64]:
            benchmark_potential(
                class_name,
                potential,
                precision,
                params,
                frames,
                boxes,
                lambdas,
                verbose=False,
            )


if __name__ == "__main__":

    benchmark_dhfr(verbose=False, num_batches=100)
    benchmark_hif2a(verbose=False, num_batches=100)

    hif2a_frames = generate_hif2a_frames(1000, 5, seed=2022)
    test_nonbonded_interaction_group_potential(hif2a_frames)
    test_nonbonded_potential(hif2a_frames)
    test_bonded_potentials(hif2a_frames)
