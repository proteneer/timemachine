import matplotlib.pyplot as plt

hist_kwargs = dict(density=True, alpha=0.5)

def plot_work_accumulation_stddev(du_dl_trajs_default, du_dl_trajs_optimized, default_schedule, optimized_schedule):
    plt.title('stddev of work accumulation per step')

    plt.ylabel(r'$\frac{\partial u}{\partial \lambda} \cdot \frac{d \lambda}{d t}$')
    plt.xlabel(r'$t$ (noneq MD step)')

    plt.plot(du_dl_trajs_default.std(0)[1:] * np.diff(default_schedule), label='manual')
    plt.plot(du_dl_trajs_optimized.std(0)[1:] * np.diff(optimized_schedule), label='sequentially optimized')

    plt.legend(title='protocol')


def plot_work_histograms(works_default, works_optimized):
    plt.title('forward work distributions')

    plt.xlabel('work ($k_B T$)')
    plt.ylabel('probability density')
    plt.yticks([])

    plt.hist(works_default, label='manual', **hist_kwargs)
    plt.hist(works_optimized, label='sequentially optimized', **hist_kwargs)

    plt.legend(title='protocol')
