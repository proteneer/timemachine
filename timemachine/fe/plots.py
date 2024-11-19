import io
import warnings
from typing import Callable, Sequence, Tuple, cast

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from timemachine.constants import BOLTZ, DEFAULT_TEMP, KCAL_TO_KJ
from timemachine.fe.bar import compute_fwd_and_reverse_df_over_time

DEFAULT_HEATMAP_ANNOTATE_THRESHOLD = 20


def plot_work(w_forward, w_reverse, axes):
    """histograms of +forward and -reverse works"""

    w_all = np.concatenate([+w_forward, -w_reverse])
    # Tear out any non-finite works
    w_all = w_all[np.isfinite(w_all)]
    a_min, a_max = np.amin(w_all), np.amax(w_all)

    axes.hist(+w_forward, alpha=0.5, label="fwd", density=True, bins=20, range=(a_min, a_max))
    axes.hist(-w_reverse, alpha=0.5, label="-rev", density=True, bins=20, range=(a_min, a_max))
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


def plot_dG_errs_subfigure(ax, components, lambdas, dG_errs):
    # one line per energy component
    for component, ys in zip(components, dG_errs):
        ax.plot(lambdas[:-1], ys, marker=".", label=component)

    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$\lambda_i$")
    ax.set_ylabel(r"$\Delta G$ error ($\lambda_i$, $\lambda_{i+1}$) / (kJ / mol)")
    ax.legend()


def plot_dG_errs_figure(components, lambdas, dG_err_by_lambda, dG_err_by_component_by_lambda):
    _, (ax_top, ax_btm) = plt.subplots(2, 1, figsize=(7, 9))
    plot_dG_errs_subfigure(ax_top, ["Overall"], lambdas, [dG_err_by_lambda])
    plot_dG_errs_subfigure(ax_btm, components, lambdas, dG_err_by_component_by_lambda.T)


def plot_overlap_summary_subfigure(ax, components, lambdas, overlaps):
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


def plot_overlap_summary_figure(components, lambdas, overlap_by_lambda, overlap_by_component_by_lambda):
    _, (ax_top, ax_btm) = plt.subplots(2, 1, figsize=(7, 9))
    plot_overlap_summary_subfigure(ax_top, ["Overall"], lambdas, [overlap_by_lambda])
    plot_overlap_summary_subfigure(ax_btm, components, lambdas, overlap_by_component_by_lambda.T)


def plot_overlap_detail_figure(
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
    u_kln_by_component_by_lambda: [n_lambdas - 1,P,2,2,T] array
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

    num_rows = len(u_kln_by_component_by_lambda)  # n_lambdas - 1 adjacent pairs
    num_cols = num_energy_components + 1  # one per component + one for overall energy

    fig, all_axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 3))
    fig.tight_layout(pad=4.0)
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
        plot_axis = all_axes[lamb_idx][num_energy_components]
        plot_title = f"{prefix}_{lamb_idx}_to_{lamb_idx + 1}"
        plot_BAR(df, df_err, w_fwd, w_rev, plot_title, plot_axis)

    # [n_lambdas x num_energy_components] plots (relying on energy decomposition)
    for lamb_idx, u_kln_by_component in enumerate(u_kln_by_component_by_lambda):
        w_fwd_by_component = u_kln_by_component[:, 1, 0] - u_kln_by_component[:, 0, 0]
        w_rev_by_component = u_kln_by_component[:, 0, 1] - u_kln_by_component[:, 1, 1]

        # loop over bond, angle, torsion, nonbonded terms etc.
        for u_idx in range(num_energy_components):
            plot_axis = all_axes[lamb_idx][u_idx]

            plot_work(w_fwd_by_component[u_idx], w_rev_by_component[u_idx], plot_axis)
            plot_axis.set_title(components[u_idx])


def plot_forward_and_reverse_ddg(
    solvent_ukln_by_lambda: NDArray,
    complex_ukln_by_lambda: NDArray,
    temperature: float = DEFAULT_TEMP,
    frames_per_step: int = 100,
    prefix: str = "",
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
    prefix: string
        Title prefix

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

    return plot_as_png_fxn(plot_fwd_reverse_predictions, fwd, fwd_err, rev, rev_err, prefix=prefix)


def plot_forward_and_reverse_dg(
    ukln_by_lambda: NDArray, temperature: float = DEFAULT_TEMP, frames_per_step: int = 100, prefix: str = ""
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
    prefix: string
        Title prefix

    Returns
    -------
    dg_convergence_plot_bytes: bytes
    """

    fwd, fwd_err, rev, rev_err = compute_fwd_and_reverse_df_over_time(ukln_by_lambda, frames_per_step=frames_per_step)

    kBT = BOLTZ * temperature

    return plot_as_png_fxn(
        plot_fwd_reverse_predictions,
        fwd * kBT / KCAL_TO_KJ,
        fwd_err * kBT / KCAL_TO_KJ,
        rev * kBT / KCAL_TO_KJ,
        rev_err * kBT / KCAL_TO_KJ,
        energy_type="∆G",
        prefix=prefix,
    )


def plot_fwd_reverse_predictions(
    fwd: NDArray, fwd_err: NDArray, rev: NDArray, rev_err: NDArray, energy_type: str = "∆∆G", prefix: str = ""
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
    prefix: string
        Title prefix
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

    fig = plt.figure(figsize=(6, 6))

    # Compute the upper and lower bounds and add +/- 1 kcal/mol to the y axis limits
    combined_vals = np.concatenate([fwd, rev])
    y_max = np.max(combined_vals)
    y_min = np.min(combined_vals)
    plt.ylim(y_min - 1.0, y_max + 1.0)

    max_error = np.abs(np.concatenate([fwd_err, rev_err])).max()
    fig.text(0.55, 0.15, f"Max error = {max_error:.2g} kcal/mol")

    plt.title(f"{prefix} {energy_type} Convergence Over Time")
    plt.plot(xs, fwd, label=f"Forward {energy_type}", marker="o")
    plt.fill_between(xs[fwd_mask], fwd[fwd_mask] - fwd_err[fwd_mask], fwd[fwd_mask] + fwd_err[fwd_mask], alpha=0.25)
    plt.plot(xs, rev, label=f"Reverse {energy_type}", marker="o")
    plt.fill_between(xs[rev_mask], rev[rev_mask] - rev_err[rev_mask], rev[rev_mask] + rev_err[rev_mask], alpha=0.25)
    plt.axhline(fwd[-1], linestyle="--")
    plt.xlabel("Fraction of simulation time")
    plt.ylabel(f"{energy_type} (kcal/mol)")
    plt.legend()
    plt.tight_layout()


def plot_chiral_restraint_energies(chiral_energies: NDArray, figsize: Tuple[float, float] = (13, 10), prefix: str = ""):
    """Plot matrix of chiral restraint energies as a heatmap.

    For use with the outputs of timemachine.fe.chiral_utils.make_chiral_flip_heatmaps.
    """
    n_states, n_frames = chiral_energies.shape
    states = np.arange(n_states)
    frames = np.arange(n_frames)

    fig, ax = plt.subplots(figsize=figsize)
    p = ax.pcolormesh(frames, states, chiral_energies, vmin=0.0)

    ax.set_xlabel("frame")
    ax.set_ylabel("state")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.get_major_locator().set_params(integer=True)

    fig.colorbar(p, label="chiral restraint energy")
    fig.suptitle(f"{prefix} Chiral Restraint Energies")
    fig.tight_layout()


def plot_hrex_transition_matrix(
    transition_probability: NDArray,
    figsize: Tuple[float, float] = (13, 10),
    annotate_threshold: int = DEFAULT_HEATMAP_ANNOTATE_THRESHOLD,
    format_annotation: Callable[[float], str] = lambda x: f"{100.0*x:.2g}",
    format_cbar_tick: Callable[[float], str] = lambda x: f"{100.0*x:.2g}%",
    prefix: str = "",
):
    """Plot matrix of estimated transition probabilities for permutation moves as a heatmap."""
    n_states, _ = transition_probability.shape
    states = np.arange(n_states)

    fig, ax = plt.subplots(figsize=figsize)
    p = ax.pcolormesh(states, states, transition_probability)

    # Skip text annotations when number of states is large
    if n_states <= annotate_threshold:
        for from_state in states:
            for to_state in states:
                prob = transition_probability[to_state, from_state]
                if prob > 0.0:
                    label = format_annotation(cast(float, prob))
                    ax.text(from_state, to_state, label, ha="center", va="center", color="w", fontsize=8)

    ax.set_xlabel("from state")
    ax.set_ylabel("to state")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_aspect("equal")

    fig.colorbar(p, label="fraction of iterations", format=lambda x, _: format_cbar_tick(x))
    fig.suptitle(f"{prefix} HREX Transition Matrix")
    fig.tight_layout()


def plot_hrex_swap_acceptance_rates_convergence(cumulative_swap_acceptance_rates: NDArray, prefix: str = ""):
    """Plot swap acceptance rates averaged over previous iterations as a function of iteration for each pair of
    neighbors."""
    _, n_pairs = cumulative_swap_acceptance_rates.shape
    fig, ax = plt.subplots()
    ax.plot(cumulative_swap_acceptance_rates)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cumulative swap acceptance rate")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(
        labels=[str(i) for i in range(n_pairs)],
        title="left state index",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    fig.suptitle(f"{prefix} HREX Swap Acceptance Rates")
    fig.tight_layout()


def plot_hrex_replica_state_distribution_heatmap(
    cumulative_replica_state_counts: NDArray,
    lambdas: Sequence[float],
    figsize: Tuple[float, float] = (13, 10),
    annotate_threshold: int = DEFAULT_HEATMAP_ANNOTATE_THRESHOLD,
    format_annotation: Callable[[float], str] = lambda x: f"{100.0*x:.2g}",
    format_cbar_tick: Callable[[float], str] = lambda x: f"{100.0*x:.2g}%",
    prefix: str = "",
):
    """Plot distribution of (replica, state) pairs as a heatmap."""
    n_iters, n_states, n_replicas = cumulative_replica_state_counts.shape
    assert n_states == len(lambdas), "Number of lambdas and states must match"
    replicas = np.arange(n_replicas)
    states = np.arange(n_states)
    count_by_replica_by_state = cumulative_replica_state_counts[-1]  # (state, replica) -> int
    fraction_by_replica_by_state = count_by_replica_by_state / n_iters  # (state, replica) -> float

    fig, ax = plt.subplots(figsize=figsize)
    p = ax.pcolormesh(replicas, states, fraction_by_replica_by_state)

    # Skip text annotations when number of states is large
    if n_states <= annotate_threshold:
        for replica in replicas:
            for state in states:
                fraction = fraction_by_replica_by_state[state, replica]
                label = format_annotation(fraction)
                ax.text(replica, state, label, ha="center", va="center", color="w", fontsize=8)

    ax.set_xlabel("replica")
    ax.set_ylabel("state")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_aspect("equal")

    # Add labels at the top of the x axis that include the replica lambda values
    top_ax = ax.secondary_yaxis("right")
    top_ax.set_ylabel(r"$\lambda$")
    top_ax.set_yticks(replicas, labels=[f"{lamb:.2f}" for lamb in lambdas])

    fig.colorbar(p, label="fraction of iterations", format=lambda x, _: format_cbar_tick(x))

    fig.suptitle(f"{prefix} HREX Replica-State Distribution Heatmap")
    fig.tight_layout()


def plot_as_png_fxn(f, *args, **kwargs) -> bytes:
    """
    Given a function which generates a plot, return the plot as png bytes.
    """
    plt.clf()
    f(*args, **kwargs)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    plot_png = buffer.read()
    return plot_png


def _plot_bond_interpolation(st, xs, systems, filter_fn, axs, row):
    bond_ks = []  # K x # bonds
    bond_bs = []  # K x # bonds
    for sys in systems:
        bond_idxs = sys.bond.potential.idxs
        keep_idxs = []
        for b_idx, idxs in enumerate(bond_idxs):
            if filter_fn(idxs):
                keep_idxs.append(b_idx)
        keep_idxs = np.array(keep_idxs, dtype=np.int32)
        bond_params = sys.bond.params
        bond_ks.append(bond_params[keep_idxs, 0])
        bond_bs.append(bond_params[keep_idxs, 1])

    bond_idxs = bond_idxs[keep_idxs]
    bond_ks = np.array(bond_ks).T  # bonds x K
    bond_bs = np.array(bond_bs).T  # bonds x K
    num_bonds = bond_ks.shape[0]
    for b_idx in range(num_bonds):
        if bond_ks[b_idx][0] != bond_ks[b_idx][-1]:
            if st._bond_idxs_belong_to_chiral_volume_turning_on(
                tuple(bond_idxs[b_idx])
            ) or st._bond_idxs_belong_to_chiral_volume_turning_off(tuple(bond_idxs[b_idx])):
                linestyle = "dashed"
            else:
                linestyle = "solid"
            label = f"{bond_idxs[b_idx]}"
            alpha = 1.0
        else:
            linestyle = "dotted"
            label = None
            alpha = 0.1

        axs[row, 0].plot(xs, bond_ks[b_idx], linestyle=linestyle, label=label, alpha=alpha)
        axs[row, 1].plot(xs, bond_bs[b_idx], linestyle=linestyle, label=label, alpha=alpha)

    axs[row, 0].set_title("bond force constants")
    axs[row, 1].set_title("bond lengths")

    axs[row, 0].set_ylabel("force constant")
    axs[row, 1].set_ylabel("bond length")

    axs[row, 0].set_xlabel("lambda window")
    axs[row, 1].set_xlabel("lambda window")

    # axs[row, 0].legend()
    # axs[row, 1].legend()


def _plot_angle_interpolation(st, xs, systems, filter_fn, axs, row):
    angle_ks = []  # K x # angles
    angle_bs = []  # K x # angles
    for sys in systems:
        angle_idxs = sys.angle.potential.idxs
        angle_params = sys.angle.params
        keep_idxs = []
        for b_idx, idxs in enumerate(angle_idxs):
            if filter_fn(idxs):
                keep_idxs.append(b_idx)
        keep_idxs = np.array(keep_idxs, dtype=np.int32)
        angle_ks.append(angle_params[keep_idxs, 0])
        angle_bs.append(angle_params[keep_idxs, 1])

    angle_idxs = angle_idxs[keep_idxs]
    angle_ks = np.array(angle_ks).T  # angles x K
    angle_bs = np.array(angle_bs).T * (180 / np.pi)  # angles x K *
    num_angles = angle_ks.shape[0]

    for b_idx in range(num_angles):
        if angle_ks[b_idx][0] != angle_ks[b_idx][-1]:
            if st._angle_idxs_belong_to_chiral_volume_turning_on(
                tuple(angle_idxs[b_idx])
            ) or st._angle_idxs_belong_to_chiral_volume_turning_off(tuple(angle_idxs[b_idx])):
                linestyle = "dashed"
            else:
                linestyle = "solid"
            label = f"{angle_idxs[b_idx]}"
            alpha = 0.75
        else:
            linestyle = "dotted"
            label = None
            alpha = 0.1
        axs[row, 0].plot(xs, angle_ks[b_idx], linestyle=linestyle, label=label, alpha=alpha)
        axs[row, 1].plot(xs, angle_bs[b_idx], linestyle=linestyle, label=label, alpha=alpha)

    axs[row, 0].set_title("angle force constants")
    axs[row, 1].set_title("angle degrees")

    axs[row, 0].set_ylabel("force constant")
    axs[row, 1].set_ylabel("angle degree")

    axs[row, 0].set_xlabel("lambda window")
    axs[row, 1].set_xlabel("lambda window")

    # axs[row, 0].legend()
    # axs[row, 1].legend()


def _plot_chiral_atom_interpolation(xs, systems, filter_fn, axs, row):
    # process_bonds
    chiral_atom_ks = []  # K x # chiral_atoms
    for sys in systems:
        chiral_atom_idxs = sys.chiral_atom.potential.idxs
        chiral_atom_params = sys.chiral_atom.params
        keep_idxs = []
        for b_idx, idxs in enumerate(chiral_atom_idxs):
            if filter_fn(idxs):
                keep_idxs.append(b_idx)
        keep_idxs = np.array(keep_idxs, dtype=np.int32)
        chiral_atom_ks.append(chiral_atom_params[keep_idxs])

    chiral_atom_idxs = chiral_atom_idxs[keep_idxs]
    chiral_atom_ks = np.array(chiral_atom_ks).T  # chiral_atoms x K
    num_chiral_atoms = chiral_atom_ks.shape[0]

    for b_idx in range(num_chiral_atoms):
        if chiral_atom_ks[b_idx][0] != chiral_atom_ks[b_idx][-1]:
            linestyle = "dashed"
            label = f"{chiral_atom_idxs[b_idx]}"
            alpha = 1.0
        else:
            linestyle = "dotted"
            label = None
            alpha = 0.1

        axs[row, 0].plot(xs, chiral_atom_ks[b_idx], linestyle=linestyle, label=label, alpha=alpha)

    axs[row, 0].set_title("chiral_atom force constants")
    axs[row, 0].set_ylabel("force constant")
    axs[row, 0].set_ylim(-100, 1200)
    axs[row, 0].set_xlabel("lambda window")
    # axs[row, 0].legend()


def _plot_proper_interpolation(xs, systems, filter_fn, axs, row):
    proper_ks = []  # K x # propers
    proper_bs = []  # K x # propers
    for sys in systems:
        proper_idxs = sys.proper.potential.idxs
        proper_params = sys.proper.params
        keep_idxs = []
        for b_idx, idxs in enumerate(proper_idxs):
            if filter_fn(idxs):
                keep_idxs.append(b_idx)
        keep_idxs = np.array(keep_idxs, dtype=np.int32)
        proper_ks.append(proper_params[keep_idxs, 0])
        proper_bs.append(proper_params[keep_idxs, 1])

    proper_idxs = proper_idxs[keep_idxs]
    proper_ks = np.array(proper_ks).T  # propers x K
    proper_bs = np.array(proper_bs).T * (180 / np.pi)  # propers x K (in degrees)
    num_propers = proper_ks.shape[0]

    for b_idx in range(num_propers):
        if proper_ks[b_idx][0] != proper_ks[b_idx][-1]:
            linestyle = "solid"
            label = f"{proper_idxs[b_idx]}"
            alpha = 1.0
        else:
            linestyle = "dotted"
            label = None
            alpha = 0.1

        axs[row, 0].plot(xs, proper_ks[b_idx], linestyle=linestyle, label=label, alpha=alpha)
        axs[row, 1].plot(xs, proper_bs[b_idx], linestyle=linestyle, label=label, alpha=alpha)

    axs[row, 0].set_title("proper force constants")
    axs[row, 1].set_title("proper degrees")

    axs[row, 0].set_ylabel("force constant")
    axs[row, 1].set_ylabel("proper degree")

    axs[row, 0].set_xlabel("lambda window")
    axs[row, 1].set_xlabel("lambda window")

    # axs[row, 0].legend(bbox_to_anchor=(1, 0.5))
    # axs[row, 1].legend(bbox_to_anchor=(1, 0.5))


def _plot_improper_interpolation(xs, systems, filter_fn, axs, row):
    improper_ks = []  # K x # impropers
    improper_bs = []  # K x # impropers
    for sys in systems:
        improper_idxs = sys.improper.potential.idxs
        improper_params = sys.improper.params
        keep_idxs = []
        for b_idx, idxs in enumerate(improper_idxs):
            if filter_fn(idxs):
                keep_idxs.append(b_idx)
        keep_idxs = np.array(keep_idxs, dtype=np.int32)
        improper_ks.append(improper_params[keep_idxs, 0])
        improper_bs.append(improper_params[keep_idxs, 1])

    improper_idxs = improper_idxs[keep_idxs]
    improper_ks = np.array(improper_ks).T  # impropers x K
    improper_bs = np.array(improper_bs).T * (180 / np.pi)  # of impropers x K (in degrees)
    num_impropers = improper_ks.shape[0]

    for b_idx in range(num_impropers):
        if improper_ks[b_idx][0] != improper_ks[b_idx][-1]:
            linestyle = "solid"
            label = f"{improper_idxs[b_idx]}"
            alpha = 1.0
        else:
            linestyle = "dotted"
            label = None
            alpha = 0.1

        axs[row, 0].plot(xs, improper_ks[b_idx], linestyle=linestyle, label=label, alpha=alpha)
        axs[row, 1].plot(xs, improper_bs[b_idx], linestyle=linestyle, label=label, alpha=alpha)

    axs[row, 0].set_title("improper force constants")
    axs[row, 1].set_title("improper degrees")

    axs[row, 0].set_ylabel("force constant")
    axs[row, 1].set_ylabel("improper degree")

    axs[row, 0].set_xlabel("lambda window")
    axs[row, 1].set_xlabel("lambda window")


def plot_interpolation_schedule(st, filter_fn, fig_title, n_windows):
    fig, axs = plt.subplots(5, 2, figsize=(9, 12))
    # plot the force constant and equilibrium bond lengths along lambda
    lambdas = np.linspace(0, 1.0, n_windows)
    systems = []
    for lam in lambdas:
        systems.append(st.setup_intermediate_state(lam))
    _plot_bond_interpolation(st, lambdas, systems, filter_fn, axs, row=0)
    _plot_chiral_atom_interpolation(lambdas, systems, filter_fn, axs, row=1)
    _plot_angle_interpolation(st, lambdas, systems, filter_fn, axs, row=2)
    _plot_proper_interpolation(lambdas, systems, filter_fn, axs, row=3)
    _plot_improper_interpolation(lambdas, systems, filter_fn, axs, row=4)
    fig.suptitle(fig_title, fontsize=12)
    plt.tight_layout()
    # plt.show()


def plot_core_interpolation_schedule(st, n_windows=48):
    plot_interpolation_schedule(st, st.all_idxs_belong_to_core, "Core Interpolation Schedule", n_windows)


def plot_dummy_a_interpolation_schedule(st, n_windows=48):
    plot_interpolation_schedule(st, st.any_idxs_belong_to_dummy_a, "Dummy Group A Interpolation Schedule", n_windows)


def plot_dummy_b_interpolation_schedule(st, n_windows=48):
    plot_interpolation_schedule(st, st.any_idxs_belong_to_dummy_b, "Dummy Group B Interpolation Schedule", n_windows)
