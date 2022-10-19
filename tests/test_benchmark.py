"""Run vanilla "apo" MD on DHFR and HIF2A test systems,
and running an intermediate lambda window "rbfe" MD for a
relative binding free energy edge from the HIF2A test system"""

import json
import time
from importlib import resources
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import timemachine
from timemachine import constants
from timemachine.fe import rbfe
from timemachine.fe.model_utils import apply_hmr
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.utils import to_md_units
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.lib.potentials import (
    HarmonicAngle,
    HarmonicBond,
    Nonbonded,
    NonbondedInteractionGroup,
    PeriodicTorsion,
)
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.testsystems.dhfr import setup_dhfr
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


@pytest.fixture(scope="module")
def hi2fa_test_frames():
    return generate_hif2a_frames(100, 1, seed=2022)


def generate_hif2a_frames(n_frames: int, frame_interval: int, seed=None, barostat_interval: int = 5, hmr: bool = True):

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    # build the protein system.
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        host_system, host_coords, _, _, host_box, _ = builders.build_protein_system(
            str(path_to_pdb), forcefield.protein_ff, forcefield.water_ff
        )

    initial_state = prepare_hif2a_initial_state(st, host_system, host_coords, host_box)

    ligand_idxs = np.arange(len(host_coords), len(initial_state.x0), dtype=np.int32)

    temperature = 300
    pressure = 1.0
    lamb = 0.0

    harmonic_bond_potential = initial_state.potentials[0]
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
        bps.append(potential.bound_impl(precision=np.float32))  # get the bound implementation

    baro_impl = None
    if barostat_interval > 0:
        group_idxs = get_group_indices(bond_list)
        baro = MonteCarloBarostat(
            initial_state.x0.shape[0],
            pressure,
            temperature,
            group_idxs,
            barostat_interval,
            seed,
        )
        baro_impl = baro.impl(bps)

    ctxt = custom_ops.Context(
        initial_state.x0,
        initial_state.v0,
        host_box,
        intg,
        bps,
        barostat=baro_impl,
    )
    steps = n_frames * frame_interval
    _, coords, boxes = ctxt.multiple_steps(np.ones(steps) * lamb, 0, frame_interval)
    assert coords.shape[0] == n_frames, f"Got {coords.shape[0]} frames, expected {n_frames}"
    return initial_state.potentials, coords, boxes, ligand_idxs


class BenchmarkPotentialResult(NamedTuple):
    label: str
    num_atoms: int
    frames: int
    param_batches: int
    num_lambdas: int
    runs_per_second: float
    runs_per_batch: int
    num_batches: int
    duration: float

    def summarize(self) -> str:
        return " ".join(
            [
                self.label,
                f"N={self.num_atoms}",
                f"Frames={self.frames}",
                f"Params={self.param_batches}",
                f"Lambdas={self.num_lambdas}",
                f"speed: {self.runs_per_second:.2f} executions/seconds",
                f"(ran {self.runs_per_batch * self.num_batches} potentials in {self.duration:.2f}s)",
            ]
        )


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
) -> BenchmarkPotentialResult:
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
    runs_per_second = None

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

    assert runs_per_second is not None
    result = BenchmarkPotentialResult(
        label,
        coords.shape[1],
        frames,
        param_batches,
        num_lambs,
        runs_per_second,
        runs_per_batch,
        num_batches,
        time.time() - start,
    )

    print(result.summarize())

    return result


class BenchmarkResult(NamedTuple):
    label: str
    num_atoms: int
    ns_per_day: float
    dt: float
    steps_per_batch: int
    num_batches: int
    duration: float

    def summarize(self) -> str:
        return " ".join(
            [
                self.label,
                f"N={self.num_atoms}",
                f"speed: {self.ns_per_day:.2f}ns/day",
                f"dt: {self.dt*1e3}fs",
                f"(ran {self.steps_per_batch * self.num_batches} steps in {self.duration:.2f}s)",
            ]
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
) -> BenchmarkResult:
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
    ns_per_day = None

    for batch in range(num_batches):

        # time the current batch
        batch_start = time.time()
        _, _, _ = ctxt.multiple_steps(lambda_schedule, compute_du_dl_interval)
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
    assert ns_per_day is not None

    result = BenchmarkResult(
        label,
        x0.shape[0],
        float(ns_per_day),
        dt,
        steps_per_batch,
        num_batches,
        time.time() - start,
    )

    print(result.summarize())

    return result


def benchmark_dhfr(verbose=False, num_batches=100, steps_per_batch=1000) -> List[BenchmarkResult]:

    host_fns, host_masses, host_coords, box = setup_dhfr()

    host_conf = []
    for x, y, z in host_coords:
        host_conf.append([to_md_units(x), to_md_units(y), to_md_units(z)])
    host_conf = np.array(host_conf)

    x0 = host_conf
    v0 = np.zeros_like(host_conf)

    return [
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
        ),
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
        ),
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
        ),
    ]


def prepare_hif2a_initial_state(st, host_system, host_coords, host_box):
    st = rbfe.SingleTopology(st.mol_a, st.mol_b, st.core, st.ff)
    host_config = rbfe.HostConfig(host_system, host_coords, host_box)
    temperature = 300.0
    lamb = 0.1
    initial_state = rbfe.setup_initial_states(st, host_config, temperature, [lamb], seed=2022)[0]
    bound_impls = [p.bound_impl(np.float32) for p in initial_state.potentials]
    val_and_grad_fn = minimizer.get_val_and_grad_fn(bound_impls, initial_state.box0, initial_state.lamb)
    assert np.all(np.isfinite(initial_state.x0)), "Initial coordinates contain nan or inf"
    ligand_coords = initial_state.x0[initial_state.ligand_idxs]
    d_ij = cdist(ligand_coords, initial_state.x0)
    # if any atom is within any of the ligand's atom's ixn radius, flag it for minimization
    cutoff = 0.5  # in nanometers
    free_idxs = np.where(np.any(d_ij < cutoff, axis=0))[0].tolist()
    x0_min = minimizer.local_minimize(initial_state.x0, val_and_grad_fn, free_idxs)
    initial_state.x0 = x0_min
    return initial_state


def benchmark_hif2a(verbose=False, num_batches=100, steps_per_batch=1000) -> List[BenchmarkResult]:

    # we use simple charge "sc" to be able to run on machines that don't have openeye licenses.
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    # build the protein system.
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
            str(path_to_pdb), forcefield.protein_ff, forcefield.water_ff
        )

    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0, forcefield.water_ff)

    results = []

    for stage, host_system, host_coords, host_box in [
        ("hif2a", complex_system, complex_coords, complex_box),
        ("solvent", solvent_system, solvent_coords, solvent_box),
    ]:

        host_fns, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.0)

        # resolve host clashes
        min_host_coords = minimizer.minimize_host_4d([st.mol_a, st.mol_b], host_system, host_coords, st.ff, host_box)

        x0 = min_host_coords
        v0 = np.zeros_like(x0)

        initial_state = prepare_hif2a_initial_state(st, host_system, host_coords, host_box)

        results.extend(
            [
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
                ),
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
                ),
                # RBFE
                benchmark(
                    stage + "-rbfe",
                    initial_state.integrator.masses,
                    initial_state.lamb,
                    initial_state.x0,
                    initial_state.v0,
                    host_box,
                    initial_state.potentials,
                    verbose=verbose,
                    num_batches=num_batches,
                    steps_per_batch=steps_per_batch,
                ),
            ]
        )

    return results


def save_results(prefix, group, results, dirname="benchmark_results"):
    version_info = timemachine._version.get_versions()
    rev = version_info["full-revisionid"][:7] + ("-dirty" if version_info["dirty"] else "")
    dir = Path(dirname)
    dir.mkdir(exist_ok=True)
    stem = "__".join([prefix, group, rev])
    path = (dir / stem).with_suffix(".json")
    with path.open("w") as f:
        json.dump([result._asdict() for result in results], f)


def test_dhfr():
    results = benchmark_dhfr(verbose=True, num_batches=2, steps_per_batch=100)
    save_results("benchmark_results", "dhfr", results)


def test_hif2a():
    results = benchmark_hif2a(verbose=True, num_batches=2, steps_per_batch=100)
    save_results("benchmark_results", "hif2a", results)


def test_nonbonded_interaction_group_potential(hi2fa_test_frames):
    pots, frames, boxes, ligand_idxs = hi2fa_test_frames
    lambdas = np.array([0.0, 1.0])
    nonbonded_potential = next(p for p in pots if isinstance(p, Nonbonded))
    assert nonbonded_potential is not None

    num_param_batches = 5
    beta = 1 / (constants.BOLTZ * 300)
    cutoff = 1.2

    precisions = [np.float32, np.float64]
    nonbonded_params = np.stack([nonbonded_potential.params] * num_param_batches)

    potential = NonbondedInteractionGroup(
        ligand_idxs,
        nonbonded_potential.get_lambda_plane_idxs(),
        nonbonded_potential.get_lambda_offset_idxs(),
        beta,
        cutoff,
    )
    class_name = potential.__class__.__name__
    results = []
    for precision in precisions:
        results.append(
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
        )

    save_results("benchmark_potential_results", "nonbonded_interaction_group", results)


def test_nonbonded_potential(hi2fa_test_frames):
    pots, frames, boxes, _ = hi2fa_test_frames

    nonbonded_pot = next(p for p in pots if isinstance(p, Nonbonded))
    assert nonbonded_pot is not None
    lambdas = np.array([0.0, 1.0])

    num_param_batches = 5

    nonbonded_params = np.stack([nonbonded_pot.params] * num_param_batches)

    precisions = [np.float32, np.float64]

    potential = Nonbonded(
        nonbonded_pot.get_exclusion_idxs(),
        nonbonded_pot.get_scale_factors(),
        nonbonded_pot.get_lambda_plane_idxs(),
        nonbonded_pot.get_lambda_offset_idxs(),
        nonbonded_pot.get_beta(),
        nonbonded_pot.get_cutoff(),
    )

    class_name = potential.__class__.__name__
    results = []
    for precision in precisions:
        results.append(
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
        )

    save_results("benchmark_potential_results", "nonbonded", results)


def test_bonded_potentials(hi2fa_test_frames):
    pots, frames, boxes, _ = hi2fa_test_frames

    lambdas = np.array([0.0, 1.0])
    results = []
    for potential in pots:
        if type(potential) not in {HarmonicAngle, HarmonicBond, PeriodicTorsion}:
            continue
        class_name = type(potential).__name__
        params = np.expand_dims(potential.params, axis=0)
        for precision in [np.float32, np.float64]:
            results.append(
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
            )

    save_results("benchmark_potential_results", "bonded", results)


if __name__ == "__main__":

    save_results("benchmark_results", "dhfr", benchmark_dhfr(verbose=False, num_batches=100))
    save_results("benchmark_results", "hif2a", benchmark_hif2a(verbose=False, num_batches=100))

    hif2a_frames = generate_hif2a_frames(1000, 5, seed=2022)
    test_nonbonded_interaction_group_potential(hif2a_frames)
    test_nonbonded_potential(hif2a_frames)
    test_bonded_potentials(hif2a_frames)
