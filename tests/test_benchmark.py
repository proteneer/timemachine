"""Run vanilla "apo" MD on DHFR and HIF2A test systems,
and running an intermediate lambda window "rbfe" MD for a
relative binding free energy edge from the HIF2A test system"""

import time
from argparse import ArgumentParser
from dataclasses import dataclass
from importlib import resources
from typing import List, Optional

import numpy as np
import pytest
from common import prepare_single_topology_initial_state
from numpy.typing import NDArray

from timemachine import constants
from timemachine.fe.free_energy import HostConfig, InitialState, MDParams, WaterSamplingParams, get_context
from timemachine.fe.model_utils import apply_hmr
from timemachine.fe.single_topology import SingleTopology
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.potentials import (
    BoundPotential,
    HarmonicBond,
    Nonbonded,
    NonbondedInteractionGroup,
    Potential,
    SummedPotential,
)
from timemachine.testsystems.dhfr import setup_dhfr
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology

SECONDS_PER_DAY = 24 * 60 * 60


@dataclass
class BenchmarkConfig:
    num_batches: int
    steps_per_batch: int
    verbose: bool


@pytest.fixture(scope="module")
def hi2fa_test_frames():
    return generate_hif2a_frames(100, 10, seed=2022, barostat_interval=20)


def generate_hif2a_frames(n_frames: int, frame_interval: int, seed=None, barostat_interval: int = 5, hmr: bool = True):
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    # build the protein system.
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        host_system, host_coords, host_box, _, num_water_atoms = builders.build_protein_system(
            str(path_to_pdb), forcefield.protein_ff, forcefield.water_ff
        )
    host_config = HostConfig(host_system, host_coords, host_box, num_water_atoms)
    initial_state = prepare_single_topology_initial_state(st, host_config)

    ligand_idxs = np.arange(len(host_coords), len(initial_state.x0), dtype=np.int32)

    temperature = constants.DEFAULT_TEMP
    pressure = constants.DEFAULT_PRESSURE

    harmonic_bond_potential = initial_state.potentials[0].potential
    assert isinstance(harmonic_bond_potential, HarmonicBond)
    bond_list = get_bond_list(harmonic_bond_potential)
    masses = initial_state.integrator.masses
    if hmr:
        dt = 2.5e-3
        masses = apply_hmr(masses, bond_list)
    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)
    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(masses), seed).impl()

    bps = []

    for potential in initial_state.potentials:
        bps.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    movers = []

    if barostat_interval > 0:
        group_idxs = get_group_indices(bond_list, len(masses))
        baro = MonteCarloBarostat(
            initial_state.x0.shape[0],
            pressure,
            temperature,
            group_idxs,
            barostat_interval,
            seed,
        )
        baro_impl = baro.impl(bps)
        movers.append(baro_impl)

    ctxt = custom_ops.Context(
        initial_state.x0,
        initial_state.v0,
        host_box,
        intg,
        bps,
        movers=movers,
    )
    steps = n_frames * frame_interval
    coords, boxes = ctxt.multiple_steps(steps, frame_interval)
    assert coords.shape[0] == n_frames, f"Got {coords.shape[0]} frames, expected {n_frames}"
    return initial_state.potentials, coords, boxes, ligand_idxs


def benchmark_potential(
    config: BenchmarkConfig,
    label: str,
    potential: Potential,
    precision,
    params: NDArray,
    coords: NDArray,
    boxes: NDArray,
    compute_du_dx: bool = True,
    compute_du_dp: bool = True,
    compute_u: bool = True,
):
    if precision == np.float32:
        label = label + "_f32"
    else:
        label = label + "_f64"
    unbound = potential.to_gpu(precision=precision).unbound_impl
    start = time.time()
    batch_times = []
    frames = coords.shape[0]
    param_batches = params.shape[0]
    runs_per_batch = frames * param_batches
    for _ in range(config.num_batches):
        batch_start = time.time()
        _, _, _ = unbound.execute_batch(
            coords,
            params,
            boxes,
            compute_du_dx,
            compute_du_dp,
            compute_u,
        )
        batch_end = time.time()
        delta = batch_end - batch_start

        batch_times.append(delta)
        runs_per_second = runs_per_batch / np.mean(batch_times)

        if config.verbose:
            print(f"executions per second: {runs_per_second:.3f}")
    print(
        f"{label}: N={coords.shape[1]} Frames={frames} Params={param_batches} speed: {runs_per_second:.2f} executions/seconds (ran {runs_per_batch * config.num_batches} potentials in {(time.time() - start):.2f}s)",
        f"du_dp={compute_du_dp}, du_dx={compute_du_dx}, u={compute_u}",
    )


def benchmark(
    config: BenchmarkConfig,
    label: str,
    masses: NDArray,
    x0: NDArray,
    v0: NDArray,
    box: NDArray,
    bound_potentials: List[BoundPotential],
    hmr: bool = True,
    barostat_interval: int = 0,
):
    if barostat_interval > 0:
        label += f"-barostat-interval-{barostat_interval}"
    seed = 1234
    dt = 1.5e-3
    temperature = constants.DEFAULT_TEMP
    pressure = constants.DEFAULT_PRESSURE

    harmonic_bond_potential = next(p for p in bound_potentials if isinstance(p.potential, HarmonicBond))
    bond_list = get_bond_list(harmonic_bond_potential.potential)
    if hmr:
        dt = 2.5e-3
        masses = apply_hmr(masses, bond_list)
    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(masses), seed).impl()

    bps = []

    for potential in bound_potentials:
        bps.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    movers = []
    if barostat_interval > 0:
        group_idxs = get_group_indices(bond_list, len(masses))
        baro = MonteCarloBarostat(
            x0.shape[0],
            pressure,
            temperature,
            group_idxs,
            barostat_interval,
            seed,
        )
        movers.append(baro.impl(bps))

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        bps,
        movers=movers,
    )

    batch_times = []

    steps_per_batch = config.steps_per_batch
    num_batches = config.num_batches

    # run once before timer starts
    ctxt.multiple_steps(steps_per_batch)

    start = time.time()

    for batch in range(num_batches):
        # time the current batch
        batch_start = time.time()
        _, _ = ctxt.multiple_steps(steps_per_batch)
        batch_end = time.time()

        delta = batch_end - batch_start

        batch_times.append(delta)

        steps_per_second = steps_per_batch / np.mean(batch_times)
        steps_per_day = steps_per_second * SECONDS_PER_DAY

        ps_per_day = dt * steps_per_day
        ns_per_day = ps_per_day * 1e-3

        if config.verbose:
            print(f"steps per second: {steps_per_second:.3f}")
            print(f"ns per day: {ns_per_day:.3f}")

    assert np.all(np.abs(ctxt.get_x_t()) < 1000)

    print(
        f"{label}: N={x0.shape[0]} speed: {ns_per_day:.2f}ns/day dt: {dt*1e3}fs (ran {steps_per_batch * num_batches} steps in {(time.time() - start):.2f}s)"
    )


def benchmark_rbfe_water_sampling(
    config: BenchmarkConfig,
    label: str,
    state: InitialState,
    water_sampling_interval: int = 400,
    radius: float = 1.0,
    batch_size: int = 250,
):
    if state.barostat is not None:
        label += f"-barostat-interval-{state.barostat.interval}"
    label += f"-water-sampling-interval-{water_sampling_interval}"
    seed = 1234

    if config.steps_per_batch < water_sampling_interval:
        print("Warning::Not running water sampling every batch, interval is too large")

    ctxt = get_context(
        state,
        MDParams(
            n_frames=1,
            n_eq_steps=0,
            steps_per_frame=1,
            seed=seed,
            water_sampling_params=WaterSamplingParams(
                interval=water_sampling_interval, radius=radius, batch_size=batch_size
            ),
        ),
    )
    assert len(ctxt.get_movers()) == 2, "Expected barostat and water sampler"

    batch_times = []

    steps_per_batch = config.steps_per_batch
    num_batches = config.num_batches

    dt = state.integrator.dt

    # run once before timer starts
    ctxt.multiple_steps(steps_per_batch)

    start = time.time()

    for batch in range(num_batches):
        # time the current batch
        batch_start = time.time()
        _, _ = ctxt.multiple_steps(steps_per_batch)
        batch_end = time.time()

        delta = batch_end - batch_start

        batch_times.append(delta)

        steps_per_second = steps_per_batch / np.mean(batch_times)
        steps_per_day = steps_per_second * SECONDS_PER_DAY

        ps_per_day = dt * steps_per_day
        ns_per_day = ps_per_day * 1e-3

        if config.verbose:
            print(f"steps per second: {steps_per_second:.3f}")
            print(f"ns per day: {ns_per_day:.3f}")

    assert np.all(np.abs(ctxt.get_x_t()) < 1000)

    print(
        f"{label}: N={state.x0.shape[0]} speed: {ns_per_day:.2f}ns/day dt: {dt*1e3}fs (ran {steps_per_batch * num_batches} steps in {(time.time() - start):.2f}s)"
    )


def benchmark_local(
    config: BenchmarkConfig,
    label: str,
    masses: NDArray,
    x0: NDArray,
    v0: NDArray,
    box: NDArray,
    bound_potentials: List[BoundPotential],
    ligand_idxs: NDArray,
    hmr: bool = True,
):
    seed = 1234
    dt = 1.5e-3
    temperature = constants.DEFAULT_TEMP
    friction = 1.0

    rng = np.random.default_rng(seed)

    harmonic_bond_potential = next(p for p in bound_potentials if isinstance(p.potential, HarmonicBond))
    bond_list = get_bond_list(harmonic_bond_potential.potential)
    if hmr:
        dt = 2.5e-3
        masses = apply_hmr(masses, bond_list)
    intg = LangevinIntegrator(temperature, dt, friction, np.array(masses), seed).impl()

    bps = []

    for potential in bound_potentials:
        bps.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        bps,
    )

    batch_times = []

    steps_per_batch = config.steps_per_batch
    num_batches = config.num_batches

    ligand_idxs = ligand_idxs.astype(np.int32)

    local_seed = rng.integers(np.iinfo(np.int32).max)
    # run once before timer starts
    ctxt.multiple_steps_local(steps_per_batch, ligand_idxs, seed=local_seed)

    start = time.time()

    for batch in range(num_batches):
        local_seed = rng.integers(np.iinfo(np.int32).max)
        # time the current batch
        batch_start = time.time()
        _, _ = ctxt.multiple_steps_local(steps_per_batch, ligand_idxs, seed=local_seed)
        batch_end = time.time()

        delta = batch_end - batch_start

        batch_times.append(delta)

        steps_per_second = steps_per_batch / np.mean(batch_times)
        steps_per_day = steps_per_second * SECONDS_PER_DAY

        ps_per_day = dt * steps_per_day
        ns_per_day = ps_per_day * 1e-3

        if config.verbose:
            print(f"steps per second: {steps_per_second:.3f}")
            print(f"ns per day: {ns_per_day:.3f}")

    assert np.all(np.abs(ctxt.get_x_t()) < 1000)

    print(
        f"{label}: N={x0.shape[0]} speed: {ns_per_day:.2f}ns/day dt: {dt*1e3}fs (ran {steps_per_batch * num_batches} steps in {(time.time() - start):.2f}s)"
    )


def run_single_topology_benchmarks(
    config: BenchmarkConfig,
    stage: str,
    st: SingleTopology,
    host_config: Optional[HostConfig],
):
    initial_state = prepare_single_topology_initial_state(st, host_config)
    barostat_interval = 0
    if host_config is not None:
        host_fns, host_masses = openmm_deserializer.deserialize_system(host_config.omm_system, cutoff=1.2)

        # RBFE
        x0 = initial_state.x0[: len(host_config.conf)]
        v0 = np.zeros_like(x0)

        for barostat_interval in [0, 25]:
            benchmark(
                config,
                f"{stage}-apo",
                np.array(host_masses),
                x0,
                v0,
                host_config.box,
                host_fns,
                barostat_interval=barostat_interval,
            )

        barostat_interval = initial_state.barostat.interval

    benchmark(
        config,
        f"{stage}-rbfe",
        initial_state.integrator.masses,
        initial_state.x0,
        initial_state.v0,
        initial_state.box0,
        initial_state.potentials,
        barostat_interval=barostat_interval,
    )

    if host_config is not None:
        benchmark_local(
            config,
            f"{stage}-rbfe-local",
            initial_state.integrator.masses,
            initial_state.x0,
            initial_state.v0,
            initial_state.box0,
            initial_state.potentials,
            initial_state.ligand_idxs,
        )

        # Only in the case where the ligand is in complex do we want to look at water sampling
        if host_config.num_water_atoms < host_config.conf.shape[0]:
            benchmark_rbfe_water_sampling(
                config,
                f"{stage}-rbfe",
                initial_state,
                water_sampling_interval=400,
            )


def benchmark_dhfr(config: BenchmarkConfig):
    host_fns, host_masses, host_conf, box = setup_dhfr()

    x0 = host_conf
    v0 = np.zeros_like(host_conf)

    for barostat_interval in [0, 25]:
        benchmark(
            config,
            "dhfr-apo",
            host_masses,
            x0,
            v0,
            box,
            host_fns,
            barostat_interval=barostat_interval,
        )


def benchmark_hif2a(config: BenchmarkConfig):
    # we use simple charge "sc" to be able to run on machines that don't have openeye licenses.
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    # build the protein system.
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        host_system, host_coords, host_box, _, host_num_waters = builders.build_protein_system(
            str(path_to_pdb), forcefield.protein_ff, forcefield.water_ff
        )

    # resolve host clashes
    host_config = HostConfig(host_system, host_coords, host_box, host_num_waters)

    run_single_topology_benchmarks(config, "hif2a", st, host_config)


def benchmark_solvent(config: BenchmarkConfig):
    # we use simple charge "sc" to be able to run on machines that don't have openeye licenses.
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    host_system, host_coords, host_box, _ = builders.build_water_system(4.0, forcefield.water_ff)

    num_water_atoms = host_coords.shape[0]

    # resolve host clashes
    host_config = HostConfig(host_system, host_coords, host_box, num_water_atoms)
    run_single_topology_benchmarks(config, "solvent", st, host_config)


def benchmark_vacuum(config: BenchmarkConfig):
    # we use simple charge "sc" to be able to run on machines that don't have openeye licenses.
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    run_single_topology_benchmarks(config, "vacuum", st, None)


def test_dhfr():
    benchmark_dhfr(BenchmarkConfig(verbose=True, num_batches=2, steps_per_batch=100))


def test_hif2a():
    benchmark_hif2a(BenchmarkConfig(verbose=True, num_batches=2, steps_per_batch=100))


def test_solvent():
    benchmark_solvent(BenchmarkConfig(verbose=True, num_batches=2, steps_per_batch=100))


def test_vacuum():
    benchmark_vacuum(BenchmarkConfig(verbose=True, num_batches=2, steps_per_batch=100))


def get_nonbonded_pot_params(bps):
    for bp in bps:
        if isinstance(bp.potential, Nonbonded):
            return bp.potential, bp.params
    else:
        raise AssertionError("Nonbonded potential not found")


def test_nonbonded_interaction_group_potential(hi2fa_test_frames):
    bps, frames, boxes, ligand_idxs = hi2fa_test_frames
    nonbonded_potential, nonbonded_params = get_nonbonded_pot_params(bps)

    config = BenchmarkConfig(num_batches=2, steps_per_batch=0, verbose=False)

    num_param_batches = 5
    beta = 1 / (constants.BOLTZ * constants.DEFAULT_TEMP)
    cutoff = 1.2

    precisions = [np.float32, np.float64]
    nonbonded_params = np.stack([nonbonded_params] * num_param_batches)

    potential = NonbondedInteractionGroup(
        nonbonded_potential.num_atoms,
        ligand_idxs,
        beta,
        cutoff,
    )
    class_name = potential.__class__.__name__
    for precision in precisions:
        benchmark_potential(
            config,
            class_name,
            potential,
            precision,
            nonbonded_params,
            frames,
            boxes,
        )


def test_hif2a_potentials(hi2fa_test_frames):
    bps, frames, boxes, _ = hi2fa_test_frames

    config = BenchmarkConfig(num_batches=2, steps_per_batch=0, verbose=False)

    num_param_batches = 5

    for bp in bps:
        potential = bp.potential
        class_name = potential.__class__.__name__
        if isinstance(potential, SummedPotential):
            class_name += "(" + ", ".join([pot.__class__.__name__ for pot in potential.potentials]) + ")"
        params = np.stack([bp.params] * num_param_batches)
        for precision in [np.float32, np.float64]:
            benchmark_potential(
                config,
                class_name,
                bp.potential,
                precision,
                params,
                frames,
                boxes,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_batches", default=100, type=int)
    parser.add_argument("--steps_per_batch", default=1000, type=int)
    parser.add_argument("--skip_dhfr", action="store_true")
    parser.add_argument("--skip_hif2a", action="store_true")
    parser.add_argument("--skip_solvent", action="store_true")
    parser.add_argument("--skip_vacuum", action="store_true")
    parser.add_argument("--skip_potentials", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = BenchmarkConfig(verbose=args.verbose, num_batches=args.num_batches, steps_per_batch=args.steps_per_batch)

    if not args.skip_dhfr:
        benchmark_dhfr(config)
    if not args.skip_hif2a:
        benchmark_hif2a(config)
    if not args.skip_solvent:
        benchmark_solvent(config)
    if not args.skip_vacuum:
        benchmark_vacuum(config)

    if not args.skip_potentials:
        hif2a_frames = generate_hif2a_frames(1000, 20, seed=2022, barostat_interval=20)
        test_nonbonded_interaction_group_potential(hif2a_frames)
        test_hif2a_potentials(hif2a_frames)
