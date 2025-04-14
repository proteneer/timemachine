import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from timemachine.fe import single_topology
from timemachine.fe.plots import (
    plot_core_interpolation_schedule,
    plot_dummy_a_interpolation_schedule,
    plot_dummy_b_interpolation_schedule,
    plot_forward_and_reverse_ddg,
    plot_forward_and_reverse_dg,
    plot_water_proposals_by_state,
    plot_work,
)
from timemachine.ff import Forcefield
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology

# Plotting code should not depend on CUDA
pytestmark = [pytest.mark.nocuda]


def test_forward_and_reverse_ddg_plot():
    rng = np.random.default_rng(2023)
    ukln_shape = (47, 2, 2, 2000)
    dummy_solv_ukln = rng.random(size=ukln_shape)
    dummy_complex_ukln = rng.random(size=ukln_shape)

    assert len(plot_forward_and_reverse_ddg(dummy_solv_ukln, dummy_complex_ukln)) > 0


def test_forward_and_reverse_ddg_plot_validation():
    rng = np.random.default_rng(2023)
    ukln_shape = (47, 2, 2, 2000)
    dummy_solv_ukln = rng.random(size=ukln_shape)
    dummy_complex_ukln = rng.random(size=ukln_shape)

    with pytest.raises(AssertionError, match="fewer samples than frames_per_step"):
        plot_forward_and_reverse_ddg(dummy_solv_ukln, dummy_complex_ukln, frames_per_step=ukln_shape[-1] + 1)
    # Verify that with different size arrays it fails
    with pytest.raises(AssertionError):
        plot_forward_and_reverse_ddg(dummy_solv_ukln, dummy_complex_ukln[0])


@pytest.mark.parametrize("ukln_shape", [(47, 2, 2, 2000), (5, 2, 2, 10)])
def test_forward_and_reverse_dg_plot(ukln_shape):
    rng = np.random.default_rng(2023)
    dummy_ukln = rng.random(size=ukln_shape) * 1000

    frames_per_step = min(ukln_shape[-1], 100)
    assert len(plot_forward_and_reverse_dg(dummy_ukln, frames_per_step=frames_per_step)) > 0


def test_forward_and_reverse_dg_plot_validation():
    rng = np.random.default_rng(2023)
    ukln_shape = (47, 2, 2, 2000)
    dummy_ukln = rng.random(size=ukln_shape)

    with pytest.raises(AssertionError, match="fewer samples than frames_per_step"):
        plot_forward_and_reverse_dg(dummy_ukln, frames_per_step=ukln_shape[-1] + 1)


def test_plots_correctly_closed():
    """Ensure that we are correctly closing figures and don't trigger a warning indicating figures that are still open"""
    # Get the number of figures that can be open before triggering a warning
    max_figures = plt.rcParams["figure.max_open_warning"]

    rng = np.random.default_rng(2024)
    ukln_shape = (10, 2, 2, 1000)
    dummy_ukln = rng.random(size=ukln_shape)

    with warnings.catch_warnings(record=True) as captured_warnings:
        for _ in range(max_figures + 1):
            assert len(plot_forward_and_reverse_dg(dummy_ukln, frames_per_step=ukln_shape[-1])) > 0
    assert all(
        "are retained until explicitly closed and may consume too much memory" not in str(warn.message)
        for warn in captured_warnings
    )


def test_plot_work_with_infs():
    rng = np.random.default_rng(2024)

    finite_values = rng.normal(loc=1000, scale=300, size=2000)

    # With finite values all should be well
    _, ax = plt.subplots()
    plot_work(finite_values, -finite_values, ax)
    plt.clf()

    nonfinite_values = np.concatenate([[np.inf, -np.inf, np.nan], finite_values])

    _, ax = plt.subplots()
    plot_work(nonfinite_values, -nonfinite_values, ax)
    plt.clf()


def test_plot_interpolation_schedule():
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    st = single_topology.SingleTopology(mol_a, mol_b, core, ff)
    plot_core_interpolation_schedule(st)
    plot_dummy_a_interpolation_schedule(st)
    plot_dummy_b_interpolation_schedule(st)


def test_plot_water_sampling_proposals():
    lambdas = np.linspace(0, 1.0, 5)
    n_iters = 10
    proposals_per_iter = 10
    rng = np.random.default_rng(2024)
    proposals_by_state_by_iter = []
    for _ in range(n_iters):
        acceptances = rng.integers(0, proposals_per_iter, size=len(lambdas))
        proposals_by_state_by_iter.append([[x, proposals_per_iter] for x in acceptances])

    cummulative_proposals_by_state = np.array(proposals_by_state_by_iter, dtype=np.int32).sum(axis=0)
    plot_water_proposals_by_state(lambdas, cummulative_proposals_by_state, prefix="example prefix")
    plt.clf()
