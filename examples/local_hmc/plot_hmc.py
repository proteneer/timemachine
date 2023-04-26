from glob import glob

import matplotlib.pyplot as plt
import numpy as np


def get_global_accept_rates():
    global_fnames = glob("hmc_profiling_results/global_*.npz")
    all_global_results = []
    for fname in global_fnames:
        all_global_results.append(np.load(fname))

    dts = []
    accept_rates = []
    for r in all_global_results:
        dt = r["dt"]
        accept_rate = np.exp(r["log_accept_probs"]).mean()

        dts.append(dt)
        accept_rates.append(accept_rate)

    dts = np.array(dts)
    accept_rates = np.array(accept_rates)

    perm = np.argsort(dts)
    global_accept_rates = accept_rates[perm]

    n_global = all_global_results[0]["n_atoms_moved"][0]

    return dts[perm], global_accept_rates, n_global


def get_local_accept_rates():
    local_fnames = glob("hmc_profiling_results/local_*.npz")
    all_local_results = []
    for fname in local_fnames:
        all_local_results.append(np.load(fname))

    radii = list(set([float(r["radius"]) for r in all_local_results]))

    sorted_radii = sorted(list(radii))

    accept_rates_by_radius = []
    avg_n_selected = []
    for radius in sorted_radii:
        dts = []
        accept_rates = []
        _n_selected = []
        for r in all_local_results:
            if float(r["radius"]) == radius:
                dt = r["dt"]
                accept_rate = np.exp(r["log_accept_probs"]).mean()

                dts.append(dt)
                accept_rates.append(accept_rate)
                _n_selected.append((r["n_atoms_moved"].mean()))

        dts = np.array(dts)
        accept_rates = np.array(accept_rates)
        perm = np.argsort(dts)
        accept_rates_by_radius.append(accept_rates[perm])
        avg_n_selected.append(np.mean(_n_selected))

    return dts[perm], sorted_radii, accept_rates_by_radius, avg_n_selected


def make_accept_rate_plot():
    _dts, global_accept_rates, n_global = get_global_accept_rates()
    dts, radii, accept_rates_by_radius, avg_n_selected = get_local_accept_rates()

    assert (_dts == dts).all()

    dts_fs = 1e3 * dts

    cmap = plt.get_cmap("viridis")
    n_conditions = len(radii) + 1
    colors = cmap.colors[:: len(cmap.colors) // n_conditions][:n_conditions]
    colors = colors[::-1]

    plt.figure(figsize=(8, 4))
    for i, (radius, y) in enumerate(zip(radii, accept_rates_by_radius)):
        plt.plot(dts_fs, y, c=colors[i], label=f"local (r = {radius} nm,  avg. n = {int(avg_n_selected[i])})")
    plt.plot(dts_fs, global_accept_rates, "--", label=f"global (n = {n_global})", c=colors[-1])
    plt.xlabel("timestep (fs)")
    plt.ylabel("acceptance rate")
    plt.ylim(0, 1)
    # plt.xscale('log')
    plt.legend(loc=(1, 0))

    plt.title("HMC acceptance rates")

    plt.tight_layout()

    fname = "local_hmc_acceptance_rates.pdf"
    print(f"saving figure to {fname}")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    make_accept_rate_plot()
