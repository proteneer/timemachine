import io
import warnings

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from timemachine.constants import BOLTZ, DEFAULT_TEMP, KCAL_TO_KJ
from timemachine.fe.bar import compute_fwd_and_reverse_df_over_time


def plot_work(w_forward, w_reverse, axes):
    """histograms of +forward and -reverse works"""
    axes.hist(+w_forward, alpha=0.5, label="fwd", density=True, bins=20)
    axes.hist(-w_reverse, alpha=0.5, label="-rev", density=True, bins=20)
    axes.set_xlabel("work (kT)")
    axes.legend()


def plot_BAR(df, df_err, fwd_delta_u, rev_delta_u, title, axes):
    """
    Generate a subplot showing overlap for a particular pair of delta_us.

    Parameters
    ----------
    df: float
        reduced free energy

    df_err: float
        reduced free energy error

    fwd_delta_u: array
        reduced works

    rev_delta_u: array
        reduced reverse works

    title: str
        title to use

    axes: matplotlib axis
        obj used to draw the figures

    """
    axes.set_title(f"{title}, dg: {df:.2f} +- {df_err:.2f} kTs")
    plot_work(fwd_delta_u, rev_delta_u, axes)


def plot_dG_errs(ax, components, lambdas, dG_errs):
    # one line per energy component
    for component, ys in zip(components, dG_errs):
        ax.plot(lambdas[:-1], ys, marker=".", label=component)

    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$\lambda_i$")
    ax.set_ylabel(r"$\Delta G$ error ($\lambda_i$, $\lambda_{i+1}$) / (kJ / mol)")
    ax.legend()


def make_dG_errs_figure(components, lambdas, dG_err_by_lambda, dG_err_by_component_by_lambda):
    _, (ax_top, ax_btm) = plt.subplots(2, 1, figsize=(7, 9))
    plot_dG_errs(ax_top, ["Overall"], lambdas, [dG_err_by_lambda])
    plot_dG_errs(ax_btm, components, lambdas, dG_err_by_component_by_lambda.T)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer.read()


def plot_overlap_summary(ax, components, lambdas, overlaps):
    # one line per energy component
    for component, ys in zip(components, overlaps):
        percentages = 100 * np.asarray(ys)
        ax.plot(lambdas[:-1], percentages, marker=".", label=component)

    # min and max within axis limits
    ax.set_ylim(-5, 105)
    ax.hlines([0, 100], min(lambdas), max(lambdas), color="grey", linestyles="--")

    # express in %
    ticks = [0, 25, 50, 75, 100]
    labels = [f"{t}%" for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    # labels and legends
    ax.set_xlabel(r"$\lambda_i$")
    ax.set_ylabel(r"pair BAR overlap ($\lambda_i$, $\lambda_{i+1}$)")
    ax.legend()


def make_overlap_summary_figure(components, lambdas, overlap_by_lambda, overlap_by_component_by_lambda):
    _, (ax_top, ax_btm) = plt.subplots(2, 1, figsize=(7, 9))
    plot_overlap_summary(ax_top, ["Overall"], lambdas, [overlap_by_lambda])
    plot_overlap_summary(ax_btm, components, lambdas, overlap_by_component_by_lambda.T)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer.read()


def make_overlap_detail_figure(
    components,
    dGs,
    dG_errs,
    u_kln_by_component_by_lambda,
    temperature,
    prefix,
):
    """Make (n_lambdas - 1) x (n_components + 1) overlap plots, and return related diagnostics

    Parameters
    ----------
    components : n_components list of strings
        component names
    dGs: (n_lambdas - 1) floats
    dG_errs: (n_lambdas - 1) floats
    u_kln_by_component_by_lambda: [L,P,2,2,T] array
    temperature: float
        kelvin
    prefix: string
        beginning of plot titles

    Returns
    -------
    overlap_detail_png: bytes


    Notes
    -----
    * May be a useful source of hints, but can be misleading in simple cases
        https://github.com/proteneer/timemachine/issues/923
    """
    num_energy_components = len(components)
    assert num_energy_components == u_kln_by_component_by_lambda[0].shape[0]

    num_rows = len(u_kln_by_component_by_lambda)  # L - 1 adjacent pairs
    num_cols = num_energy_components + 1  # one per component + one for overall energy

    figure, all_axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 3))
    if num_rows == 1:
        all_axes = [all_axes]

    # free energy analysis + plots (not relying on energy decomposition)
    kBT = BOLTZ * temperature
    beta = 1 / kBT

    for lamb_idx, u_kln_by_component in enumerate(u_kln_by_component_by_lambda):
        u_kln = u_kln_by_component.sum(0)

        w_fwd = u_kln[1, 0] - u_kln[0, 0]
        w_rev = u_kln[0, 1] - u_kln[1, 1]

        df = beta * dGs[lamb_idx]
        df_err = beta * dG_errs[lamb_idx]

        # add to plot
        plot_axis = all_axes[lamb_idx - 1][num_energy_components]
        plot_title = f"{prefix}_{lamb_idx - 1}_to_{lamb_idx}"
        plot_BAR(df, df_err, w_fwd, w_rev, plot_title, plot_axis)

    # [n_lambdas x num_energy_components] plots (relying on energy decomposition)
    for lamb_idx, u_kln_by_component in enumerate(u_kln_by_component_by_lambda):
        w_fwd_by_component = u_kln_by_component[:, 1, 0] - u_kln_by_component[:, 0, 0]
        w_rev_by_component = u_kln_by_component[:, 0, 1] - u_kln_by_component[:, 1, 1]

        # loop over bond, angle, torsion, nonbonded terms etc.
        for u_idx in range(num_energy_components):
            plot_axis = all_axes[lamb_idx - 1][u_idx]

            plot_work(w_fwd_by_component[u_idx], w_rev_by_component[u_idx], plot_axis)
            plot_axis.set_title(components[u_idx])

    # detail plot as png
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    overlap_detail_png = buffer.read()

    return overlap_detail_png


def plot_forward_and_reverse_ddg(
    solvent_ukln_by_lambda: NDArray,
    complex_ukln_by_lambda: NDArray,
    temperature: float = DEFAULT_TEMP,
    frames_per_step: int = 100,
) -> bytes:
    """Forward and reverse ddG plot given a solvent and complex ukln.
    In the case of good convergence, the forward and reverse ddGs should be similar and the ddG should
    not be drifting with all samples.

    Refer to figure 5 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4420631/ for more details.

    Parameters
    ----------
    solvent_ukln_by_lambda: [n_lambdas, 2, 2, N] array
        Solvent unitless ukln broken up by lambdas
    complex_ukln_by_lambda: [n_lambdas, 2, 2, N] array
        Complex unitless ukln broken up by lambdas
    temperature: float
        Temperature that samples were collected at.
    frames_per_step: int
        Number of frames to include in a sample when computing u_kln over time

    Returns
    -------
    ddg_convergence_plot_bytes: bytes
    """

    solvent_fwd, solvent_fwd_err, solvent_rev, solvent_rev_err = compute_fwd_and_reverse_df_over_time(
        solvent_ukln_by_lambda, frames_per_step=frames_per_step
    )
    complex_fwd, complex_fwd_err, complex_rev, complex_rev_err = compute_fwd_and_reverse_df_over_time(
        complex_ukln_by_lambda, frames_per_step=frames_per_step
    )

    kBT = BOLTZ * temperature

    fwd = (complex_fwd - solvent_fwd) * kBT / KCAL_TO_KJ
    rev = (complex_rev - solvent_rev) * kBT / KCAL_TO_KJ

    fwd_err = np.linalg.norm([complex_fwd_err, solvent_fwd_err], axis=0) * kBT / KCAL_TO_KJ
    rev_err = np.linalg.norm([complex_rev_err, solvent_rev_err], axis=0) * kBT / KCAL_TO_KJ

    return plot_fwd_reverse_predictions(fwd, fwd_err, rev, rev_err)


def plot_forward_and_reverse_dg(
    ukln_by_lambda: NDArray,
    temperature: float = DEFAULT_TEMP,
    frames_per_step: int = 100,
) -> bytes:
    """Forward and reverse dG plot given a ukln.
    In the case of good convergence, the forward and reverse dGs should be similar and the dG should
    not be drifting with all samples.

    Refer to figure 5 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4420631/ for more details.

    Parameters
    ----------
    ukln_by_lambda: [n_lambdas, 2, 2, N] array
        Unitless ukln broken up by lambdas
    temperature: float
        Temperature that samples were collected at.
    frames_per_step: int
        Number of frames to include in a sample when computing u_kln over time

    Returns
    -------
    dg_convergence_plot_bytes: bytes
    """

    fwd, fwd_err, rev, rev_err = compute_fwd_and_reverse_df_over_time(ukln_by_lambda, frames_per_step=frames_per_step)

    kBT = BOLTZ * temperature

    return plot_fwd_reverse_predictions(
        fwd * kBT / KCAL_TO_KJ,
        fwd_err * kBT / KCAL_TO_KJ,
        rev * kBT / KCAL_TO_KJ,
        rev_err * kBT / KCAL_TO_KJ,
        energy_type="∆G",
    )


def plot_fwd_reverse_predictions(
    fwd: NDArray, fwd_err: NDArray, rev: NDArray, rev_err: NDArray, energy_type: str = "∆∆G"
):
    """Forward and reverse plot given forward and reverse estimates of energies.
    In the case of good convergence, the forward and reverse predictions should be similar and the energies should
    not be drifting with all samples.

    Refer to figure 5 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4420631/ for more details.

    Parameters
    ----------
    fwd: [N] array
        Energies computed in forward direction, in units of kcal/mol
    fwd_err: [N] array
        Energies std errors computed in forward direction, in units of kcal/mol
    rev: [N] array
        Energies computed in reverse direction, in units of kcal/mol
    rev_err: [N] array
        Energies std errors computed in reverse direction, in units of kcal/mol
    energy_type: string
        The type of free energy that is being plotted, typically '∆∆G' or '∆G'

    Returns
    -------
    energy_convergence_plot_bytes: bytes
    """
    assert len(fwd) == len(rev)
    assert len(fwd) == len(fwd_err)
    assert len(rev) == len(rev_err)

    # If the values aren't close warn, but don't fail as MBAR can produce relatively large differences
    # when there is poor convergence
    if not np.allclose(fwd[-1], rev[-1]):
        warnings.warn(f"Final energies are not close: Fwd {fwd[-1]:.3f} Rev {rev[-1]:.3f}")
    if np.isfinite(fwd_err).all() and np.isfinite(rev_err).all():
        if not np.allclose(fwd_err[-1], rev_err[-1]):
            warnings.warn(f"Final errors are not close: Fwd err {fwd_err[-1]:.3f} Rev err {rev_err[-1]:.3f}")
    fwd_mask = np.isfinite(fwd_err)
    rev_mask = np.isfinite(rev_err)
    xs = np.linspace(1.0 / len(fwd), 1.0, len(fwd))

    plt.figure(figsize=(6, 6))
    plt.title(f"{energy_type} Convergence Over Time")
    plt.plot(xs, fwd, label=f"Forward {energy_type}", marker="o")
    plt.fill_between(xs[fwd_mask], fwd[fwd_mask] - fwd_err[fwd_mask], fwd[fwd_mask] + fwd_err[fwd_mask], alpha=0.25)
    plt.plot(xs, rev, label=f"Reverse {energy_type}", marker="o")
    plt.fill_between(xs[rev_mask], rev[rev_mask] - rev_err[rev_mask], rev[rev_mask] + rev_err[rev_mask], alpha=0.25)
    plt.axhline(fwd[-1], linestyle="--")
    plt.xlabel("Fraction of simulation time")
    plt.ylabel(f"{energy_type} (kcal/mol)")
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plot_png = buffer.read()

    return plot_png


def plot_hrex_swap_acceptance_rates(swap_acceptance_rates: NDArray):
    _, ax = plt.subplots()
    ax.plot(swap_acceptance_rates, ".-")
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(r"left state index")
    ax.set_ylabel(r"swap acceptance rate")
    ax.xaxis.get_major_locator().set_params(integer=True)


def plot_hrex_swap_acceptance_rates_convergence(cumulative_swap_acceptance_rates: NDArray):
    _, n_pairs = cumulative_swap_acceptance_rates.shape
    _, ax = plt.subplots()
    ax.plot(cumulative_swap_acceptance_rates)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cumulative swap acceptance rate")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(labels=[str(i) for i in range(n_pairs)], title="left state index")


def plot_hrex_replica_state_distribution(cumulative_replica_state_counts: NDArray):
    n_iters, n_replicas, n_states = cumulative_replica_state_counts.shape
    count_by_replica_by_state = cumulative_replica_state_counts[-1]  # (replica, state) -> int
    fraction_by_replica_by_state = count_by_replica_by_state / n_iters  # (replica, state) -> float
    fraction_by_state_by_replica = fraction_by_replica_by_state.T  # (state, replica) -> float

    bottom = np.zeros(n_states)
    _, ax = plt.subplots()
    for state_idx, fraction_by_state in enumerate(fraction_by_state_by_replica):
        ax.bar(np.arange(n_replicas), fraction_by_state, bottom=bottom, width=0.5, label=str(state_idx))
        bottom += fraction_by_state

    ax.set_xlabel("replica")
    ax.set_ylabel("fraction of iterations")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(title="state")


def plot_hrex_replica_state_distribution_heatmap(cumulative_replica_state_counts: NDArray):
    n_iters, n_replicas, n_states = cumulative_replica_state_counts.shape
    count_by_replica_by_state = cumulative_replica_state_counts[-1]  # (replica, state) -> int
    fraction_by_replica_by_state = count_by_replica_by_state / n_iters  # (replica, state) -> float
    fraction_by_state_by_replica = fraction_by_replica_by_state.T  # (state, replica) -> float

    fig, ax = plt.subplots()
    p = ax.pcolor(np.arange(n_replicas), np.arange(n_states), fraction_by_state_by_replica, vmin=0.0, vmax=1.0)
    ax.set_xlabel("replica")
    ax.set_ylabel("state")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.colorbar(p, label="fraction of iterations")


def plot_hrex_replica_state_distribution_convergence(cumulative_replica_state_counts: NDArray):

    n_iters, _, n_states = cumulative_replica_state_counts.shape
    fraction_by_iter_by_replica_by_state = (
        cumulative_replica_state_counts / np.arange(n_iters)[:, None, None]
    )  # (iter, replica, state) -> float
    fraction_by_replica_by_iter_by_state = np.swapaxes(
        fraction_by_iter_by_replica_by_state, 0, 1
    )  # (replica, iter, state) -> float

    fig, axs = plt.subplots(nrows=n_states)

    for replica_idx, (fraction_by_iter_by_state, ax) in enumerate(zip(fraction_by_replica_by_iter_by_state, axs)):
        p = ax.pcolor(np.arange(n_iters), np.arange(n_states), fraction_by_iter_by_state.T, vmin=0.0, vmax=1.0)
        ax.set_xlabel("iteration")
        ax.set_ylabel("state")
        ax.set_title(f"replica = {replica_idx}")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(p, cax=cbar_ax, label="fraction of iterations")
