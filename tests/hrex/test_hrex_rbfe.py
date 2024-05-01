from dataclasses import replace
from importlib import resources
from typing import Optional
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from psutil import Process
from scipy import stats

from timemachine.fe.free_energy import (
    HostConfig,
    HREXParams,
    MDParams,
    SimulationResult,
    WaterSamplingParams,
    sample_with_context,
)
from timemachine.fe.plots import (
    plot_hrex_replica_state_distribution_convergence,
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
    forcefield = Forcefield.load_default()

    return mol_a, mol_b, core, forcefield, host_config


@pytest.fixture(
    scope="module", params=[None, "solvent", pytest.param("complex", marks=pytest.mark.nightly(reason="slow"))]
)
def hif2a_single_topology_leg(request):
    host_name = request.param
    return host_name, get_hif2a_single_topology_leg(request.param)


@pytest.mark.nightly(reason="Slow")
def test_hrex_rbfe_hif2a(hif2a_single_topology_leg):
    host_name, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg
    md_params = MDParams(
        n_frames=200,
        n_eq_steps=10_000,
        steps_per_frame=400,
        seed=2024,
        hrex_params=HREXParams(n_frames_bisection=100, n_frames_per_iter=1),
        water_sampling_params=WaterSamplingParams(interval=400, n_proposals=1000) if host_name == "complex" else None,
    )
    n_windows = 5

    rss_traj = []

    def sample_and_record_rss(*args, **kwargs):
        traj = sample_with_context(*args, **kwargs)
        rss_traj.append(Process().memory_info().rss)
        return traj

    with patch("timemachine.fe.free_energy.sample_with_context", sample_and_record_rss):
        result = estimate_relative_free_energy_bisection_hrex(
            mol_a,
            mol_b,
            core,
            forcefield,
            host_config,
            md_params,
            lambda_interval=(0.0, 0.15),
            n_windows=n_windows,
        )

    # Check that memory usage is not increasing
    rss_traj = rss_traj[10:]  # discard initial transients
    rss_diff_count = np.sum(np.diff(rss_traj) != 0)
    rss_increase_count = np.sum(np.diff(rss_traj) > 0)
    assert stats.binom.pmf(rss_increase_count, n=rss_diff_count, p=0.5) >= 0.001

    if DEBUG:
        plot_hrex_rbfe_hif2a(result)

    assert result.hrex_diagnostics

    assert result.hrex_diagnostics.cumulative_swap_acceptance_rates.shape[1] == n_windows - 1

    # Swap acceptance rates for all neighboring pairs should be >~ 20%
    final_swap_acceptance_rates = result.hrex_diagnostics.cumulative_swap_acceptance_rates[-1]
    assert np.all(final_swap_acceptance_rates > 0.2)

    # Expect some replicas to visit every state
    final_replica_state_counts = result.hrex_diagnostics.cumulative_replica_state_counts[-1]
    assert np.any(np.all(final_replica_state_counts > 0, axis=0))

    # Check plots were generated
    assert result.hrex_plots
    assert result.hrex_plots.transition_matrix_png
    assert result.hrex_plots.swap_acceptance_rates_convergence_png
    assert result.hrex_plots.replica_state_distribution_convergence_png
    assert result.hrex_plots.replica_state_distribution_heatmap_png


def plot_hrex_rbfe_hif2a(result: SimulationResult):
    assert result.hrex_diagnostics
    plot_hrex_swap_acceptance_rates_convergence(result.hrex_diagnostics.cumulative_swap_acceptance_rates)
    plot_hrex_transition_matrix(result.hrex_diagnostics.transition_matrix)
    plot_hrex_replica_state_distribution_convergence(result.hrex_diagnostics.cumulative_replica_state_counts)
    plot_hrex_replica_state_distribution_heatmap(result.hrex_diagnostics.cumulative_replica_state_counts)
    plt.show()


def test_hrex_rbfe_reproducibility(hif2a_single_topology_leg):
    _, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg

    md_params = MDParams(
        n_frames=10,
        n_eq_steps=10,
        steps_per_frame=400,
        seed=2023,
        hrex_params=HREXParams(n_frames_bisection=1, n_frames_per_iter=1),
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

    res1 = run(2023)
    res2 = run(2023)
    res3 = run(2024)

    np.testing.assert_equal(res1.frames, res2.frames)
    np.testing.assert_equal(res1.boxes, res2.boxes)
    assert not np.all(res1.frames == res3.frames)

    if host_config:
        # for vacuum leg, boxes are trivially identical
        assert not np.all(np.array(res1.boxes) == np.array(res3.boxes))
