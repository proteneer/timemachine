from fe.free_energy_rabfe import construct_pre_optimized_absolute_lambda_schedule_solvent
import numpy as np

from smc import simple_smc, conditional_multinomial_resample
from testsystems.biphenyl import construct_biphenyl_test_system

from parallel.client import AbstractClient, CUDAPoolClient
from functools import partial

from pickle import dump
from datetime import datetime

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
    parser.add_argument("--N", type=int, help="number of walkers", default=1000)
    parser.add_argument("--T", type=int, help="number of lambda windows", default=100)
    parser.add_argument("--resample_thresh", type=float, help="resample when fractional ESS < thresh", default=0.6)
    cmd_args = parser.parse_args()
    print(cmd_args)

    # parallel set up
    n_gpus = 10
    client = CUDAPoolClient(n_gpus)
    pmap = partial(parallel_map, client=client)

    # SMC set up
    N = cmd_args.N
    T = cmd_args.T
    resample_thresh = cmd_args.resample_thresh
    config = dict(N=N, T=T, resample_thresh=resample_thresh, n_md_steps=n_md_steps)

    # samples = np.random.choice(initial_samples, size=N)
    sample_inds = np.random.choice(np.arange(len(initial_samples)), size=N)
    samples = [initial_samples[i] for i in sample_inds]
    lambdas = construct_pre_optimized_absolute_lambda_schedule_solvent(T)[::-1]
    resample = partial(conditional_multinomial_resample, thresh=resample_thresh)

    def propagate(xs, lam):
        xlams = [(x, lam) for x in xs]
        results = pmap(advance, xlams)

        # TODO: cleaner way to do this unpacking?
        xs_next = [x_next for (x_next, timing) in results]

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
