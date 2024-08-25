import json
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
    sample,
)
from timemachine.fe.lambda_schedule import bisection_lambda_schedule
from timemachine.fe.rbfe import (
    DEFAULT_HREX_PARAMS,
    setup_initial_states,
    setup_optimized_host,
    setup_optimized_initial_state,
)
from timemachine.fe.single_topology import SingleTopology
from timemachine.ff import Forcefield
from timemachine.lib import VelocityVerletIntegrator
from timemachine.md import builders
from timemachine.potentials import (
    ChiralAtomRestraint,
    HarmonicAngle,
    HarmonicAngleStable,
    HarmonicBond,
    NonbondedPairList,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
)
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

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    if host_name == "complex":
        with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
            host_sys, host_conf, box, _, num_water_atoms = builders.build_protein_system(
                str(protein_path), forcefield.protein_ff, forcefield.water_ff, mols=[mol_a, mol_b]
            )
            box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
        host_config = HostConfig(host_sys, host_conf, box, num_water_atoms)
    elif host_name == "solvent":
        host_sys, host_conf, box, _ = builders.build_water_system(4.0, forcefield.water_ff, mols=[mol_a, mol_b])
        box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
        host_config = HostConfig(host_sys, host_conf, box, host_conf.shape[0])

    single_topology = SingleTopology(mol_a, mol_b, core, forcefield)
    host = setup_optimized_host(single_topology, host_config) if host_config else None

    lambda_grid = np.linspace(*lambda_endpoints, n_windows)

    initial_states = setup_initial_states(
        single_topology, host, DEFAULT_TEMP, lambda_grid, seed=2023, min_cutoff=0.7 if host_name == "complex" else None
    )

    n_frames = 500 // n_windows

    assert len(initial_states) == n_windows

    return single_topology, host, host_name, n_frames, n_windows, initial_states


def combine_vacuum_initial_states(states):
    from collections import defaultdict

    atoms_per_state = [state.x0.shape[0] for state in states]
    combined_coords = np.concatenate([state.x0 for state in states])
    combined_masses = np.concatenate([state.integrator.masses for state in states])
    combined_velos = np.concatenate([state.v0 for state in states])
    # Box doesn't really matter
    box = max([x.box0 for x in states], key=lambda x: np.linalg.det(x))
    potentials_by_type = defaultdict(list)
    for state in states:
        for pot in state.potentials:
            potentials_by_type[type(pot.potential)].append(pot)
    valid = [
        HarmonicBond,
        HarmonicAngle,
        HarmonicAngleStable,
        PeriodicTorsion,
        NonbondedPairList,
        NonbondedPairListPrecomputed,
        ChiralAtomRestraint,
    ]
    for x in potentials_by_type.keys():
        if x not in valid:
            raise ValueError(f"{x} {valid}")
    mega_bps = []
    for pot_type, bps in potentials_by_type.items():
        assert len(bps) == len(states)
        if pot_type in (PeriodicTorsion, HarmonicBond, HarmonicAngleStable, HarmonicAngle, ChiralAtomRestraint):
            new_idxs = []
            new_params = []
            offset = 0
            for i, bp in enumerate(bps):
                pot = bp.potential
                # Offset the indices by the number of atoms that have come before
                idxs = pot.idxs + offset
                new_params.append(bp.params)
                new_idxs.append(idxs)
                offset += atoms_per_state[i]
            mega_bps.append(pot_type(idxs=np.concatenate(new_idxs)).bind(np.concatenate(new_params)))
        elif pot_type == NonbondedPairListPrecomputed:
            new_idxs = []
            new_params = []
            cutoff = bps[0].potential.cutoff
            beta = bps[0].potential.beta
            offset = 0
            for i, bp in enumerate(bps):
                pot = bp.potential
                assert pot.cutoff == cutoff
                assert pot.beta == beta
                # Offset the indices by the number of atoms that have come before
                idxs = pot.idxs + offset
                new_params.append(bp.params)
                new_idxs.append(idxs)
                offset += atoms_per_state[i]
            mega_bps.append(
                pot_type(idxs=np.concatenate(new_idxs), cutoff=cutoff, beta=beta).bind(np.concatenate(new_params))
            )
        else:
            assert 0
    assert len(mega_bps) == len(states[0].potentials)
    new_integrator = replace(states[0].integrator, masses=combined_masses)
    return replace(
        states[0], integrator=new_integrator, x0=combined_coords, v0=combined_velos, box0=box, potentials=mega_bps
    )


def run_benchmark_hif2a_single_topology(hif2a_single_topology_leg, mode, enable_water_sampling) -> float:
    single_topology, host, host_name, n_frames, n_windows, initial_states = hif2a_single_topology_leg
    assert n_windows >= 2
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
            replace(md_params, hrex_params=DEFAULT_HREX_PARAMS.hrex_params),
            print_diagnostics_interval=None,
        )
    elif mode == "sequential":
        run = partial(run_sims_sequential, initial_states, md_params, temperature=temperature)
    elif mode == "bisection":
        # Bisection is expected to use a slightly different initial schedule to reduce
        # amount of minimization. The additional logic it to emulate estimate_relative_free_energy_bisection
        lambda_grid = bisection_lambda_schedule(
            n_windows, lambda_interval=(initial_states[0].lamb, initial_states[-1].lamb)
        )
        # Function to be used by run_sims_bisection
        make_optimized_initial_state = partial(
            setup_optimized_initial_state,
            single_topology,
            host=host,
            optimized_initial_states=initial_states,
            temperature=temperature,
            seed=md_params.seed,
        )
        bisection_initial_states = [make_optimized_initial_state(lamb) for lamb in lambda_grid]

        # Redefine function with new optimized initial states
        make_optimized_initial_state = partial(
            make_optimized_initial_state,
            optimized_initial_states=bisection_initial_states,
        )

        # Bisection is a bit different since it has to generate new windows, but still good to benchmark
        # as it is done upfront before HREX in practice
        run = partial(
            run_sims_bisection,
            [initial_states[0].lamb, initial_states[-1].lamb],
            make_optimized_initial_state,
            md_params,
            n_bisections=n_windows - 2,
            temperature=temperature,
            min_overlap=None,
            verbose=True,
        )
    elif mode == "combined":
        mega_initial_state = combine_vacuum_initial_states(initial_states)
        run = partial(sample, mega_initial_state, md_params, max_buffer_frames=100)
    else:
        assert False

    var, elapsed_ns = run_with_timing(run)  # type: ignore
    if mode == "combined":
        print(len(var.frames), var.frames[0].shape, mega_initial_state.x0.shape)
        last_frame = var.frames[-1]
        box = var.boxes[-1]
        for pot in mega_initial_state.potentials:
            from timemachine.md.minimizer import check_force_norm

            du_dx, _ = pot.to_gpu(np.float32).bound_impl.execute(last_frame, box, compute_u=False)
            check_force_norm(-du_dx)

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


def test_combined_vacuum_matches_sequential(hif2a_single_topology_leg):
    single_topology, host, host_name, n_frames, n_windows, initial_states = hif2a_single_topology_leg
    assert n_windows >= 2
    if host_name is not None:
        pytest.skip("Only supports vacuum")

    md_params = MDParams(n_frames=n_frames, n_eq_steps=1, steps_per_frame=400, seed=2023)

    new_impl = VelocityVerletIntegrator(initial_states[0].integrator.dt, initial_states[0].integrator.masses)
    for state in initial_states:
        state.integrator = new_impl
    mega_initial_state = combine_vacuum_initial_states(initial_states)

    _, ref_trajs = run_sims_sequential(initial_states, md_params, temperature=DEFAULT_TEMP)
    combined_trajs = sample(mega_initial_state, md_params, max_buffer_frames=n_frames)
    state_coords = initial_states[0].x0
    combined_xs = np.array(combined_trajs.frames).reshape(n_frames, n_windows, *state_coords.shape)
    ref_xs_concat = np.hstack([np.array(traj.frames) for traj in ref_trajs]).reshape(
        n_frames, n_windows, *state_coords.shape
    )
    np.testing.assert_array_equal(ref_xs_concat, combined_xs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--leg", default="vacuum", choices=["vacuum", "solvent", "complex"])
    parser.add_argument("--n_frames", default=100, type=int)
    parser.add_argument("--modes", nargs="*", default=["sequential", "bisection", "hrex", "combined"])
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
    with open(f"{args.leg}_{args.n_frames}_benchmarks.json", "w") as ofs:
        json.dump(
            {
                "n_windows": args.n_windows,
                "n_frames": args.n_frames,
                "water_sampling": args.water_sampling,
                "leg": args.leg,
                "timings": timings,
            },
            ofs,
        )
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
