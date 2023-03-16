import io

import numpy as np
from matplotlib import pyplot as plt

from timemachine.constants import BOLTZ


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
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    overlap_detail_png = buffer.read()

    return overlap_detail_png
