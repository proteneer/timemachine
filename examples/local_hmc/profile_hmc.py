from copy import deepcopy
from time import time

import networkx as nx
import numpy as np
from hmc import GlobalHMC, LocalHMC  # TODO: move to timemachine

from timemachine.constants import DEFAULT_TEMP
from timemachine.md.barostat.utils import get_bond_list
from timemachine.md.states import CoordsVelBox
from timemachine.parallel.client import CUDAPoolClient
from timemachine.testsystems.dhfr import setup_dhfr

# TODO: update to use:
# [x] cutoff = 1.2 instead of 1.0
# [x] restraint energy before and after

n_samples = 1000
folder_name = f"hmc_profiling_results_mcmc_{n_samples}"


def measure_mcmc_performance(proposal_fxn, eq_samples, n_samples):
    log_accept_probs = []
    n_atoms_moved = []
    proposal_norms = []
    timings = []
    for _ in range(n_samples):
        xvb_0 = deepcopy(eq_samples[np.random.randint(len(eq_samples))])

        t0 = time()
        xvb_prop, log_accept_prob = proposal_fxn(xvb_0)
        t1 = time()

        diff = xvb_prop.coords - xvb_0.coords

        n_atoms_moved.append(int((diff != 0).any(1).sum()))
        proposal_norms.append(np.linalg.norm(diff))
        log_accept_probs.append(log_accept_prob)
        timings.append(t1 - t0)

        if np.random.rand() < np.exp(log_accept_prob):
            xvb_0 = xvb_prop

    return dict(
        n_atoms_moved=np.array(n_atoms_moved),
        proposal_norms=np.array(proposal_norms),
        log_accept_probs=np.array(log_accept_probs),
        timings=np.array(timings),
    )


def make_global_hmc_proposal_fxn(bound_potentials, dt, masses, temperature, n_steps=100):
    global_hmc_move = GlobalHMC(bound_potentials, dt, masses, temperature)

    def proposal_fxn(xvb):
        xvb_prop, log_accept_prob = global_hmc_move.propose(xvb, n_steps)
        return xvb_prop, log_accept_prob

    return proposal_fxn


def make_local_hmc_proposal_fxn(
    bound_potentials, dt, masses, local_idxs=[0], radius=1.2, k=1000.0, n_steps=100, temperature=DEFAULT_TEMP
):
    local_hmc_move = LocalHMC(bound_potentials, dt, masses, temperature)

    def proposal_fxn(xvb):
        i = local_idxs[np.random.randint(len(local_idxs))]
        xvb_prop, log_accept_prob = local_hmc_move.propose(xvb, n_steps, i, radius, k)
        return xvb_prop, log_accept_prob

    return proposal_fxn


# pre-generated samples using Langevin at default settings
# [x] TODO: probably I should propagate each of the stored Langevin samples using a few steps of HMC?
# --> done: see polish_equilibrium_samples (which runs 100 "polishing" steps of Barker MCMC)
# dhfr_samples = np.load('dhfr_langevin_samples.npz')
dhfr_samples = np.load("dhfr_mcmc_samples.npz")
# TODO: use dhfr_mcmc_samples when completed
xs, boxes = dhfr_samples["xs"], dhfr_samples["boxes"]
eq_samples = [CoordsVelBox(x, None, box) for (x, box) in zip(xs, boxes)]

# system
host_fns, host_masses, host_conf, box = setup_dhfr(1.2)
constant_masses = np.ones_like(host_masses) * np.mean(host_masses)

# grid search setup
n_steps = 100  # fixed n_steps: molecular HMC is not as sensitive to this as other parameters

# dt_grid_fs = np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0])
# dt_grid_fs = 10**np.linspace(-4, np.log10(4.0), 20)

small_dt_grid_fs = 10 ** np.linspace(-4, -1, 5)
large_dt_grid_fs = 10 ** np.linspace(-1, np.log10(5.0), 15)
dt_grid_fs = np.unique(np.hstack([small_dt_grid_fs, large_dt_grid_fs]))

dt_grid_ps = 1e-3 * dt_grid_fs  # picoseconds


def process_global_setting(i):
    dt = dt_grid_ps[i]
    print()
    proposal_fxn = make_global_hmc_proposal_fxn(host_fns, dt, constant_masses, DEFAULT_TEMP, n_steps)
    results = measure_mcmc_performance(proposal_fxn, eq_samples, n_samples=100)

    print(f"dt = {dt}, avg. acceptance rate = {np.exp(results['log_accept_probs']).mean()}")
    fname = f"{folder_name}/global_dt_{i}.npz"
    print("saving result to ", fname)

    np.savez(fname, dt=dt, **results)

    return True


radius_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]
k = 1000.0

# get indices of protein atoms in the system
harmonic_bond_potential = host_fns[0]
bond_list = get_bond_list(harmonic_bond_potential.potential)
g = nx.Graph(bond_list)
connected_components = list(nx.connected_components(g))
sizes = [len(c) for c in connected_components]
protein_idxs = np.array(list(connected_components[np.argmax(sizes)]))


def process_local_settings(i, j):
    dt = dt_grid_ps[i]
    radius = radius_grid[j]

    print(f"processing dt = {dt}, radius = {radius}...")

    proposal_fxn = make_local_hmc_proposal_fxn(
        host_fns, dt, constant_masses, protein_idxs, radius, k, n_steps, DEFAULT_TEMP
    )
    results = measure_mcmc_performance(proposal_fxn, eq_samples, n_samples=n_samples)
    print(
        f"completed dt = {dt}, radius = {radius}! avg. acceptance rate = {np.exp(results['log_accept_probs']).mean()}"
    )

    fname = f"{folder_name}/local_dt_{i}_radius_{j}.npz"
    print("saving result to ", fname)

    np.savez(fname, radius=radius, dt=dt, **results)

    return True


if __name__ == "__main__":

    client = CUDAPoolClient(max_workers=10)
    futures = []

    # global HMC baseline: just dt_grid
    for i in range(len(dt_grid_ps)):
        futures.append(client.submit(process_global_setting, i))

    # local HMC variations: vary both dt and radius
    for i in range(len(dt_grid_ps)):
        for j in range(len(radius_grid)):
            futures.append(client.submit(process_local_settings, i, j))
    successes = [future.result() for future in futures]
    print(successes)
