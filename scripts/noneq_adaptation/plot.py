import matplotlib.pyplot as plt
import numpy as np
import os
from fe.free_energy import construct_lambda_schedule
from deploy import interpolate_lambda_schedule

hist_kwargs = dict(density=True, alpha=0.5)


# TODO: reduce duplication of titles and axis labels

def plot_lambda_schedules(n_md_steps, optimized_lam_traj):
    x = np.arange(n_md_steps)
    default = construct_lambda_schedule(n_md_steps)
    optimized = interpolate_lambda_schedule(optimized_lam_traj, n_md_steps)
    plt.title(f'default and optimized protocols\n# MD steps = {n_md_steps}')

    plt.ylabel(r'$\lambda_t$')
    plt.xlabel(r'$t$ (noneq MD step)')

    plt.plot(x, default, label='manual')
    plt.plot(x, optimized, label='optimized')

    # plt.legend(title='protocol')


def plot_work_accumulation_stddev(du_dl_trajs_default, du_dl_trajs_optimized, optimized_lam_traj):
    n_md_steps = du_dl_trajs_default.shape[1]

    x = np.arange(n_md_steps)
    default = construct_lambda_schedule(n_md_steps)
    optimized = interpolate_lambda_schedule(optimized_lam_traj, n_md_steps)
    plt.title('stddev of work\naccumulated per step')

    plt.ylabel(r'$\frac{\partial u}{\partial \lambda} \cdot \frac{d \lambda}{d t}$')
    plt.xlabel(r'$t$ (noneq MD step)')

    plt.plot(x[1:], du_dl_trajs_default.std(0)[1:] * np.diff(default), label='manual')
    plt.plot(x[1:], du_dl_trajs_optimized.std(0)[1:] * np.diff(optimized), label='optimized')

    # plt.legend(title='protocol')


def plot_work_histograms(works_default, works_optimized):
    plt.title('forward work distributions')

    plt.xlabel('work ($k_B T$)')
    plt.ylabel('probability density')
    plt.yticks([])

    plt.hist(works_default, label='manual', **hist_kwargs)
    plt.hist(works_optimized, label='optimized', **hist_kwargs)

    # plt.legend(title='protocol')


if __name__ == '__main__':

    optimized_lam_trajs = np.load('results/optimized_lam_trajs.npz')

    du_dl_trajs_default = np.load('results/du_dl_trajs_default.npz')
    du_dl_trajs_optimized = np.load('results/du_dl_trajs_optimized.npz')

    results = np.load('results/works_via_du_dl.npz')

    n_conditions = len(results['total_md_step_range'])

    # (3 rows) x (n_conditions columns)
    plt.figure(figsize=(3 * n_conditions, 3 * 3))

    # row 1: plot optimized protocols
    ax = None
    for i in range(n_conditions):
        n_md_steps = results['total_md_step_range'][i]
        ax = plt.subplot(3, n_conditions, i + 1, sharey=ax)
        plot_lambda_schedules(n_md_steps, optimized_lam_trajs[str(i)])
        if i == 0:
            plt.legend(title='protocol')

    # row 2: plot work accumulation over course of protocol...
    ax = None
    for i in range(n_conditions):
        n_md_steps = results['total_md_step_range'][i]
        lam_traj = optimized_lam_trajs[str(i)]
        plot_ind = (1 * n_conditions) + i + 1  # (num_previous_plots) + i + (0-indexed -> 1-indexed)
        ax = plt.subplot(3, n_conditions, plot_ind, sharey=ax)
        plot_work_accumulation_stddev(
            du_dl_trajs_default[str(n_md_steps)],
            du_dl_trajs_optimized[str(n_md_steps)],
            optimized_lam_trajs[str(i)]
        )
        if i == 0:
            plt.legend(title='protocol')

    # row 3: plot final work histograms
    the_big_zipper = zip(results['total_md_step_range'], results['works_default'], results['works_optimized'])
    ax = None
    for i, (n_steps, w_default, w_optimized) in enumerate(the_big_zipper):
        plot_ind = (2 * n_conditions) + i + 1
        ax = plt.subplot(3, n_conditions, plot_ind, sharex=ax, sharey=ax)
        plot_work_histograms(w_default, w_optimized)
        if i == 0:
            plt.legend(title='protocol')

    plt.tight_layout()

    figure_path = os.path.join(os.path.dirname(__file__), 'figures/default_vs_optimized.png')
    print(f'saving figure to {figure_path}')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
