from importlib import resources
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest

from timemachine.fe.free_energy import HostConfig, MDParams, SimulationResult
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


@pytest.mark.nightly(reason="Slow")
@pytest.mark.parametrize("host", [None, "complex", "solvent"])
def test_hrex_rbfe_hif2a(host: Optional[str]):
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    md_params = MDParams(n_frames=200, n_eq_steps=10_000, steps_per_frame=400, seed=2023)

    host_config: Optional[HostConfig] = None

    if host == "complex":
        with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
            host_sys, host_conf, box, _, num_water_atoms = builders.build_protein_system(
                str(protein_path), forcefield.protein_ff, forcefield.water_ff
            )
            box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
        host_config = HostConfig(host_sys, host_conf, box, num_water_atoms)
    elif host == "solvent":
        host_sys, host_conf, box, _ = builders.build_water_system(4.0, forcefield.water_ff)
        box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
        host_config = HostConfig(host_sys, host_conf, box, host_conf.shape[0])

    result = estimate_relative_free_energy_bisection_hrex(
        mol_a,
        mol_b,
        core,
        forcefield,
        host_config,
        md_params,
        lambda_interval=(0.0, 0.15),
        n_windows=5,
        n_frames_bisection=100,
        n_frames_per_iter=5,
    )

    if DEBUG:
        plot_hrex_rbfe_hif2a(result)

    assert result.hrex_diagnostics

    # Swap acceptance rates for all neighboring pairs should be >~ 20%
    final_swap_acceptance_rates = result.hrex_diagnostics.cumulative_swap_acceptance_rates[-1]
    assert np.all(final_swap_acceptance_rates > 0.2)

    # All replicas should have visited each state at least once
    final_replica_state_counts = result.hrex_diagnostics.cumulative_replica_state_counts[-1]
    assert np.all(final_replica_state_counts > 0)


def plot_hrex_rbfe_hif2a(result: SimulationResult):
    assert result.hrex_diagnostics
    plot_hrex_swap_acceptance_rates_convergence(result.hrex_diagnostics.cumulative_swap_acceptance_rates)
    plot_hrex_transition_matrix(result.hrex_diagnostics.transition_matrix)
    plot_hrex_replica_state_distribution_convergence(result.hrex_diagnostics.cumulative_replica_state_counts)
    plot_hrex_replica_state_distribution_heatmap(result.hrex_diagnostics.cumulative_replica_state_counts)
    plt.show()
