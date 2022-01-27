from datetime import datetime
from functools import partial
from pickle import dump

import numpy as np

from fe.free_energy_rabfe import construct_pre_optimized_absolute_lambda_schedule_solvent
from md.smc import simple_smc, conditional_multinomial_resample
from parallel.client import AbstractClient, CUDAPoolClient
from testsystems.biphenyl import construct_biphenyl_test_system

# TODO: refactor so that n_md_steps doesn't have to be specified here...
n_md_steps = 5000
potential_energy_fxn, mover, initial_samples = construct_biphenyl_test_system(n_steps=n_md_steps)


def advance(xlam):
    x, lam = xlam
    x_next = mover.move(x, lam)

    return x_next


def u(xlam):
    x, lam = xlam
    return potential_energy_fxn.u(x, lam)


# TODO: move this into parallel utils
def parallel_map(fxn, xs, client: AbstractClient):
    """[fxn(x) for x in xs], parallelized using client"""

    futures = []
    for x in xs:
        futures.append(client.submit(fxn, x))
    return [f.result() for f in futures]


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_walkers", type=int, help="number of walkers", default=1000)
    parser.add_argument("--n_windows", type=int, help="number of lambda windows", default=100)
    parser.add_argument("--resample_thresh", type=float, help="resample when fractional ESS < thresh", default=0.6)
    parser.add_argument("--n_gpus", type=int, help="number of devices that can be used in parallel", default=1)
    cmd_args = parser.parse_args()
    print(cmd_args)

    # parallel set up
    client = CUDAPoolClient(cmd_args.n_gpus)
    pmap = partial(parallel_map, client=client)

    # SMC set up
    n_walkers = cmd_args.n_walkers
    n_windows = cmd_args.n_windows
    resample_thresh = cmd_args.resample_thresh
    config = dict(n_walkers=n_walkers, n_windows=n_windows, resample_thresh=resample_thresh, n_md_steps=n_md_steps)

    # samples = np.random.choice(initial_samples, size=n_walkers)
    sample_inds = np.random.choice(np.arange(len(initial_samples)), size=n_walkers)
    samples = [initial_samples[i] for i in sample_inds]
    lambdas = construct_pre_optimized_absolute_lambda_schedule_solvent(n_windows)[::-1]
    resample = partial(conditional_multinomial_resample, thresh=resample_thresh)

    def propagate(xs, lam):
        xlams = [(x, lam) for x in xs]
        xs_next = pmap(advance, xlams)
        return xs_next

    def log_prob(xs, lam):
        xlams = [(x, lam) for x in xs]
        u_s = np.array(pmap(u, xlams))
        return -u_s

    smc_result = simple_smc(
        samples,
        lambdas,
        propagate,
        log_prob,
        resample,
    )

    with open(f"smc_result_{datetime.now()}.pkl", "wb") as f:
        dump((smc_result, config), f)

    traj, log_weights_traj, ancestry_traj, incremental_log_weights_traj = smc_result

# TODO: use less disk space!
