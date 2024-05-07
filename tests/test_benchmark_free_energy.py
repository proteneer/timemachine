import time
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import replace
from functools import partial
from importlib import resources
from itertools import product
from typing import Callable, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pytest

from timemachine.constants import DEFAULT_TEMP
from timemachine.fe.free_energy import (
    HostConfig,
    MDParams,
    WaterSamplingParams,
    run_sims_bisection,
    run_sims_hrex,
    run_sims_sequential,
)
from timemachine.fe.rbfe import setup_initial_states, setup_optimized_host, setup_optimized_initial_state
from timemachine.fe.single_topology import SingleTopology
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology

A = TypeVar("A")


def run_with_timing(f: Callable[[], A]) -> Tuple[A, float]:
    start_time = time.perf_counter_ns()
    result = f()
    elapsed_ns = time.perf_counter_ns() - start_time
    return result, elapsed_ns


hif2a_single_topology_leg_params = {
    f"{host_name}-{n_windows}": (host_name, n_windows)
    for host_name, n_windows in product([None, "complex", "solvent"], [5, 10, 20])
}


@pytest.fixture(
    scope="module", params=hif2a_single_topology_leg_params.values(), ids=tuple(hif2a_single_topology_leg_params.keys())
)
def hif2a_single_topology_leg(request):
    host_name, n_windows = request.param
    return setup_hif2a_single_topology_leg(host_name, n_windows, (0.0, 0.2))


def setup_hif2a_single_topology_leg(host_name: str, n_windows: int, lambda_endpoints: Tuple[float, float]):
    forcefield = Forcefield.load_default()
    host_config: Optional[HostConfig] = None
    assert len(lambda_endpoints) == 2
    assert lambda_endpoints[0] < lambda_endpoints[1]

    if host_name == "complex":
        with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
            host_sys, host_conf, box, _, num_water_atoms = builders.build_protein_system(
                str(protein_path), forcefield.protein_ff, forcefield.water_ff
            )
            box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
        host_config = HostConfig(host_sys, host_conf, box, num_water_atoms)
    elif host_name == "solvent":
        host_sys, host_conf, box, _ = builders.build_water_system(4.0, forcefield.water_ff)
        box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
        host_config = HostConfig(host_sys, host_conf, box, host_conf.shape[0])

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    single_topology = SingleTopology(mol_a, mol_b, core, forcefield)
    host = setup_optimized_host(single_topology, host_config) if host_config else None

    lambdas = np.linspace(lambda_endpoints[0], lambda_endpoints[1], n_windows)

    initial_states = setup_initial_states(single_topology, host, DEFAULT_TEMP, lambdas, seed=2023, min_cutoff=0.7)

    n_frames = 500 // n_windows

    return single_topology, host, host_name, n_frames, n_windows, initial_states


def run_benchmark_hif2a_single_topology(hif2a_single_topology_leg, mode, enable_water_sampling) -> float:
    single_topology, host, host_name, n_frames, n_windows, initial_states = hif2a_single_topology_leg
    if host_name != "complex" and enable_water_sampling:
        pytest.skip("Water sampling disabled outside of complex")

    md_params = MDParams(n_frames=n_frames, n_eq_steps=1, steps_per_frame=400, seed=2023)
    if enable_water_sampling:
        md_params = replace(md_params, water_sampling_params=WaterSamplingParams())
    temperature = DEFAULT_TEMP

    run: Optional[Callable] = None
    if mode == "hrex":
        run = partial(
            run_sims_hrex,
            initial_states,
            md_params,
            n_frames_per_iter=1,
            print_diagnostics_interval=None,
        )
    elif mode == "sequential":
        run = partial(run_sims_sequential, initial_states, md_params, temperature=temperature)
    elif mode == "bisection":
        # Function to be used by run_sims_bisection
        make_optimized_initial_state = partial(
            setup_optimized_initial_state,
            single_topology,
            host=host,
            optimized_initial_states=initial_states,
            temperature=temperature,
            seed=md_params.seed,
        )
        # Bisection is a bit different since it has to generate new windows, but still good to benchmark
        # as it is done upfront before HREX in practice
        assert len(initial_states) >= 2
        initial_lambdas = [initial_states[0].lamb, initial_states[-1].lamb]
        run = partial(
            run_sims_bisection,
            initial_lambdas,
            make_optimized_initial_state,
            md_params,
            n_bisections=n_windows - len(initial_lambdas),
            temperature=temperature,
            min_overlap=None,
            verbose=True,
        )
    else:
        assert False

    _, elapsed_ns = run_with_timing(run)

    print("water sampling:", enable_water_sampling)
    print("mode:", mode)
    print("host:", host_name)
    print("n_windows:", n_windows)
    print("n_frames:", n_frames)

    dt_ns = initial_states[0].integrator.dt / 1e3
    ns = md_params.n_frames * md_params.steps_per_frame * dt_ns
    elapsed_days = elapsed_ns / 1e9 / 60 / 60 / 24

    ns_per_day = n_windows * ns / elapsed_days

    print(f"total ns/day: {ns_per_day:.1f}")
    return ns_per_day


@pytest.mark.nightly(reason="Slow")
@pytest.mark.parametrize("enable_water_sampling", [False, True])
@pytest.mark.parametrize("mode", ["sequential", "bisection", "hrex"])
def test_benchmark_hif2a_single_topology(hif2a_single_topology_leg, mode, enable_water_sampling):
    run_benchmark_hif2a_single_topology(hif2a_single_topology_leg, mode, enable_water_sampling)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--leg", default="vacuum", choices=["vacuum", "solvent", "complex"])
    parser.add_argument("--n_frames", default=100, type=int)
    parser.add_argument("--modes", nargs="*", default=["sequential", "bisection", "hrex"])
    parser.add_argument("--n_windows", nargs="*", type=int, default=[2, 4, 8, 16, 32, 48])
    parser.add_argument("--water_sampling", action="store_true", default=False)
    args = parser.parse_args()

    if args.leg != "complex" and args.water_sampling:
        print("Water sampling will not run outside of complex")

    timings = defaultdict(list)
    for windows in args.n_windows:
        single_topology, host, host_name, _, _, initial_states = setup_hif2a_single_topology_leg(
            args.leg, windows, (0.0, 1.0)
        )
        for mode in args.modes:
            ns_per_day = run_benchmark_hif2a_single_topology(
                (single_topology, host, host_name, args.n_frames, windows, initial_states), mode, args.water_sampling
            )
            timings[mode].append(ns_per_day)
    fig, axes = plt.subplots(ncols=len(args.modes), sharey=True)
    if len(args.modes) == 1:
        axes = [axes]
    fig.suptitle(f"Leg: {args.leg}, Frames {args.n_frames}, Water Sampling {args.water_sampling}")
    for i, (ax, mode) in enumerate(zip(axes, args.modes)):
        ax.set_title(mode)
        ax.plot(args.n_windows, timings[mode], marker="x")
        ax.set_xlabel("windows")
        if i == 0:
            ax.set_ylabel("ns/day")
    fig.tight_layout()
    plt.savefig(f"{args.leg}_{args.n_frames}_benchmarks.png", dpi=150)
