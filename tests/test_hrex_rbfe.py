from dataclasses import replace
from importlib import resources
from typing import Optional
from unittest.mock import patch
from warnings import catch_warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from psutil import Process
from scipy import stats

from timemachine.fe.free_energy import (
    HostConfig,
    HREXParams,
    HREXSimulationResult,
    MDParams,
    RESTParams,
    WaterSamplingParams,
    sample_with_context_iter,
)
from timemachine.fe.plots import (
    plot_hrex_replica_state_distribution_heatmap,
    plot_hrex_swap_acceptance_rates_convergence,
    plot_hrex_transition_matrix,
)
from timemachine.fe.rbfe import estimate_relative_free_energy_bisection_hrex
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology

DEBUG = False


def get_hif2a_single_topology_leg(host_name: str | None):
    forcefield = Forcefield.load_default()
    host_config: Optional[HostConfig] = None

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    if host_name == "complex":
        with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
            host_sys, host_conf, box, host_top, num_water_atoms = builders.build_protein_system(
                str(protein_path), forcefield.protein_ff, forcefield.water_ff, mols=[mol_a, mol_b]
            )
            box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
        host_config = HostConfig(host_sys, host_conf, box, num_water_atoms, host_top)
    elif host_name == "solvent":
        solvent_sys, solvent_conf, box, solvent_top = builders.build_water_system(
            4.0, forcefield.water_ff, mols=[mol_a, mol_b]
        )
        box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
        host_config = HostConfig(solvent_sys, solvent_conf, box, solvent_conf.shape[0], solvent_top)

    return mol_a, mol_b, core, forcefield, host_config


@pytest.fixture(
    scope="module",
    params=[
        None,
        pytest.param("solvent", marks=pytest.mark.nightly(reason="slow")),
        pytest.param("complex", marks=pytest.mark.nightly(reason="slow")),
    ],
)
def hif2a_single_topology_leg(request):
    host_name = request.param
    return host_name, get_hif2a_single_topology_leg(request.param)


@pytest.mark.parametrize("seed", [2024])
def test_hrex_rbfe_hif2a_water_sampling_warning(hif2a_single_topology_leg, seed):
    host_name, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg
    if host_name != "complex":
        pytest.skip("Only relevant in complex")
    md_params = MDParams(
        n_frames=2,
        n_eq_steps=100,
        steps_per_frame=10,
        seed=seed,
        hrex_params=HREXParams(n_frames_bisection=100),
        water_sampling_params=WaterSamplingParams(interval=400, n_proposals=1000) if host_name == "complex" else None,
    )
    # Warning will only be triggered if total steps per window is less than the water sampling interval
    assert md_params.n_frames * md_params.steps_per_frame < md_params.water_sampling_params.interval
    n_windows = 2

    with catch_warnings(record=True) as captured_warnings:
        estimate_relative_free_energy_bisection_hrex(
            mol_a,
            mol_b,
            core,
            forcefield,
            host_config,
            md_params,
            lambda_interval=(0.0, 0.15),
            n_windows=n_windows,
            min_cutoff=0.7,
        )
    # We have hundreds of warnings thrown by MBAR in this code, so got to sift through
    assert len(captured_warnings) >= 1

    assert any("Not running any water sampling" in str(warn.message) for warn in captured_warnings)


@pytest.mark.parametrize("max_bisection_windows, target_overlap", [(5, None), (5, 0.667)])
@pytest.mark.parametrize("enable_rest", [False, True])
@pytest.mark.parametrize("seed", [2024])
def test_hrex_rbfe_hif2a(hif2a_single_topology_leg, seed, max_bisection_windows, target_overlap, enable_rest):
    host_name, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg
    md_params = MDParams(
        n_frames=200,
        n_eq_steps=10_000,
        steps_per_frame=400,
        seed=seed,
        hrex_params=HREXParams(
            n_frames_bisection=100,
            optimize_target_overlap=target_overlap,
            rest_params=(
                RESTParams(max_temperature_scale=3.0, temperature_scale_interpolation="exponential")
                if enable_rest
                else None
            ),
        ),
        water_sampling_params=WaterSamplingParams(interval=400, n_proposals=1000) if host_name == "complex" else None,
    )

    rss_traj = []

    def sample_and_record_rss(*args, **kwargs):
        result = sample_with_context_iter(*args, **kwargs)
        rss_traj.append(Process().memory_info().rss)
        return result

    with patch("timemachine.fe.free_energy.sample_with_context_iter", sample_and_record_rss):
        result = estimate_relative_free_energy_bisection_hrex(
            mol_a,
            mol_b,
            core,
            forcefield,
            host_config,
            md_params,
            prefix=host_name if host_name is not None else "vacuum",
            lambda_interval=(0.0, 0.15),
            n_windows=max_bisection_windows,
            min_cutoff=0.7 if host_name == "complex" else None,
        )

    final_windows = len(result.final_result.initial_states)
    # All of the lambda values of the initial states should be different
    assert len(set([s.lamb for s in result.final_result.initial_states])) == final_windows

    if md_params.hrex_params.optimize_target_overlap is not None:
        assert final_windows <= max_bisection_windows
    else:
        # min_overlap is None here, will reach the max number of windows
        assert final_windows == max_bisection_windows

    assert len(rss_traj) > final_windows * md_params.n_frames
    # Check that memory usage is not increasing
    rss_traj = rss_traj[10:]  # discard initial transients
    assert len(rss_traj)
    rss_diff_count = np.sum(np.diff(rss_traj) != 0)
    rss_increase_count = np.sum(np.diff(rss_traj) > 0)
    assert stats.binom.pmf(rss_increase_count, n=rss_diff_count, p=0.5) >= 0.001

    if DEBUG:
        plot_hrex_rbfe_hif2a(result)

    assert result.hrex_diagnostics.cumulative_swap_acceptance_rates.shape[1] == final_windows - 1

    # Swap acceptance rates for all neighboring pairs should be >~ 20%
    final_swap_acceptance_rates = result.hrex_diagnostics.cumulative_swap_acceptance_rates[-1]
    assert np.all(final_swap_acceptance_rates > 0.2)

    # Expect some replicas to visit every state
    final_replica_state_counts = result.hrex_diagnostics.cumulative_replica_state_counts[-1]
    assert np.any(np.all(final_replica_state_counts > 0, axis=0))

    assert isinstance(result.hrex_diagnostics.relaxation_time, float)
    assert result.hrex_diagnostics.normalized_kl_divergence >= 0.0

    assert len(result.hrex_diagnostics.replica_idx_by_state_by_iter) == md_params.n_frames
    assert all(
        len(replica_idx_by_state) == final_windows
        for replica_idx_by_state in result.hrex_diagnostics.replica_idx_by_state_by_iter
    )

    # Initial permutation should be the identity
    np.testing.assert_array_equal(result.hrex_diagnostics.replica_idx_by_state_by_iter[0], np.arange(final_windows))

    # Check that we can extract replica trajectories
    n_atoms = result.final_result.initial_states[0].x0.shape[0]
    rng = np.random.default_rng(seed)
    n_atoms_subset = rng.choice(n_atoms) + 1  # in [1, n_atoms]
    atom_idxs = rng.choice(n_atoms, n_atoms_subset, replace=False)
    trajs_by_replica = result.extract_trajectories_by_replica(atom_idxs)
    assert trajs_by_replica.shape == (final_windows, md_params.n_frames, n_atoms_subset, 3)

    # Check that the frame-to-frame rmsd is lower for replica trajectories versus state trajectories
    def time_lagged_rmsd(traj):
        sds = jnp.sum(jnp.diff(traj, axis=0) ** 2, axis=(1, 2))
        return jnp.sqrt(jnp.mean(sds))

    # (states, frames)
    trajs_by_state = np.array(
        [[np.array(frame)[atom_idxs] for frame in state_traj.frames] for state_traj in result.trajectories]
    )

    replica_traj_rmsds = jax.vmap(time_lagged_rmsd)(trajs_by_replica)
    state_traj_rmsds = jax.vmap(time_lagged_rmsd)(trajs_by_state)

    # should have rmsd(replica trajectory) < rmsd(state trajectory) for all pairs (replica, state)
    assert np.max(replica_traj_rmsds) < np.min(state_traj_rmsds)

    # Check that we can extract ligand trajectories by replica
    ligand_trajs_by_replica = result.extract_ligand_trajectories_by_replica()
    n_ligand_atoms = len(result.final_result.initial_states[0].ligand_idxs)
    assert ligand_trajs_by_replica.shape == (final_windows, md_params.n_frames, n_ligand_atoms, 3)

    # Check plots were generated
    assert result.hrex_plots
    assert result.hrex_plots.transition_matrix_png
    assert result.hrex_plots.swap_acceptance_rates_convergence_png
    assert result.hrex_plots.replica_state_distribution_heatmap_png


def plot_hrex_rbfe_hif2a(result: HREXSimulationResult):
    plot_hrex_swap_acceptance_rates_convergence(result.hrex_diagnostics.cumulative_swap_acceptance_rates)
    plot_hrex_transition_matrix(result.hrex_diagnostics.transition_matrix)
    plot_hrex_replica_state_distribution_heatmap(
        result.hrex_diagnostics.cumulative_replica_state_counts,
        [state.lamb for state in result.final_result.initial_states],
    )
    plt.show()


@pytest.mark.parametrize("seed", [2023])
@pytest.mark.parametrize("local_md", [True, False])
def test_hrex_rbfe_reproducibility(hif2a_single_topology_leg, local_md: bool, seed):
    host_name, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg

    if local_md and host_name is None:
        pytest.skip("No local MD for vacuum")

    md_params = MDParams(
        n_frames=10,
        n_eq_steps=10,
        steps_per_frame=400,
        local_steps=200 if local_md else 0,
        seed=seed,
        hrex_params=HREXParams(n_frames_bisection=1),
    )

    run = lambda seed: estimate_relative_free_energy_bisection_hrex(
        mol_a,
        mol_b,
        core,
        forcefield,
        host_config,
        replace(md_params, seed=seed),
        lambda_interval=(0.0, 0.1),
        n_windows=3,
    )

    res1 = run(seed)
    res2 = run(seed)
    np.testing.assert_equal(res1.frames, res2.frames)
    np.testing.assert_equal(res1.boxes, res2.boxes)

    res3 = run(seed + 1)

    assert not np.all(res1.frames == res3.frames)

    if host_config:
        # for vacuum leg, boxes are trivially identical
        assert not np.all(np.array(res1.boxes) == np.array(res3.boxes))
