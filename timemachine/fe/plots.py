import io

import numpy as np
from matplotlib import pyplot as plt


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


def make_dG_errs_figure(components, lambdas, dG_errs_by_lambda, dG_errs_by_lambda_by_component):
    _, (ax_top, ax_btm) = plt.subplots(2, 1, figsize=(7, 9))
    plot_dG_errs(ax_top, ["Overall"], lambdas, [dG_errs_by_lambda])
    plot_dG_errs(ax_btm, components, lambdas, dG_errs_by_lambda_by_component)
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


def make_overlap_summary_figure(components, lambdas, overlaps_by_lambda, overlaps_by_lambda_by_component):
    _, (ax_top, ax_btm) = plt.subplots(2, 1, figsize=(7, 9))
    plot_overlap_summary(ax_top, ["Overall"], lambdas, [overlaps_by_lambda])
    plot_overlap_summary(ax_btm, components, lambdas, overlaps_by_lambda_by_component)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer.read()
