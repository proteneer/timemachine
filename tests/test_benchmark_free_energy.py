import time
from dataclasses import replace
from functools import partial
from importlib import resources
from itertools import product
from typing import Callable, Optional, Tuple, TypeVar

import numpy as np
import pytest

from timemachine.constants import DEFAULT_TEMP
from timemachine.fe.free_energy import HostConfig, MDParams, WaterSamplingParams, run_sims_hrex, run_sims_sequential
from timemachine.fe.rbfe import setup_initial_states, setup_optimized_host
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
    forcefield = Forcefield.load_default()
    host_config: Optional[HostConfig] = None

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

    lambdas = np.linspace(0.0, 0.2, n_windows)

    initial_states = setup_initial_states(single_topology, host, DEFAULT_TEMP, lambdas, seed=2023, min_cutoff=0.7)

    return host_name, n_windows, initial_states


@pytest.mark.nightly(reason="Slow")
@pytest.mark.parametrize(
    "enable_water_sampling,intermediate_window_water_sampling", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("enable_hrex", [False, True])
def test_benchmark_hif2a_single_topology(
    hif2a_single_topology_leg, enable_hrex, enable_water_sampling, intermediate_window_water_sampling
):
    host_name, n_windows, initial_states = hif2a_single_topology_leg
    if host_name != "complex" and enable_water_sampling:
        pytest.skip("Water sampling disabled outside of complex")

    n_frames = 500 // n_windows
    md_params = MDParams(n_frames=n_frames, n_eq_steps=1, steps_per_frame=400, seed=2023)
    if enable_water_sampling:
        md_params = replace(
            md_params,
            water_sampling_params=WaterSamplingParams(
                n_initial_iterations=1, intermediate_sampling=intermediate_window_water_sampling
            ),
        )
    temperature = DEFAULT_TEMP

    if enable_hrex:
        run = partial(
            run_sims_hrex,
            initial_states,
            md_params,
            n_frames_per_iter=1,
            print_diagnostics_interval=None,
        )
    else:
        run = partial(run_sims_sequential, initial_states, md_params, temperature=temperature)

    _, elapsed_ns = run_with_timing(run)

    print("water sampling:", enable_water_sampling, ", intermediate windows:", intermediate_window_water_sampling)
    print("hrex:", enable_hrex)
    print("host:", host_name)
    print("n_windows:", n_windows)
    print("n_frames:", n_frames)

    dt_ns = initial_states[0].integrator.dt / 1e3
    ns = md_params.n_frames * md_params.steps_per_frame * dt_ns
    elapsed_days = elapsed_ns / 1e9 / 60 / 60 / 24

    print(f"total ns/day: {n_windows * ns / elapsed_days:.1f}")
