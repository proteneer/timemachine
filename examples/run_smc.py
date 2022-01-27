from fe.free_energy_rabfe import construct_pre_optimized_absolute_lambda_schedule_solvent
import numpy as np
from tqdm import tqdm

from testsystems.biphenyl import construct_biphenyl_test_system

from parallel.client import AbstractClient, CUDAPoolClient
from functools import partial

from scipy.special import logsumexp
import numpy as onp
from pickle import dump
from datetime import datetime
from time import time

n_md_steps = 5000

potential_energy_fxn, mover, initial_samples = construct_biphenyl_test_system(n_steps=n_md_steps)

move_timings = []


def advance(xlam):
    x, lam = xlam

    t0 = time()
    x_next = mover.move(x, lam)
    t1 = time()

    ## doesn't work with parallel execution
    # global move_timings
    # move_timings.append(t1 - t0)
    # print(f'moved {n_md_steps} in {t1 - t0:.3f} s')

    ## instead, modify return type
    return x_next, t1 - t0


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


def summarize(values):
    n = len(values)

    discard_equil = 100
    if n > discard_equil:
        v = np.array(values)[discard_equil:]
        return f"(min={min(v):.3f}, max={max(v):.3f}, median={np.median(v):.3f}, stddev={np.std(v):.3f}, n={n})"
    else:
        return "empty"


def simple_smc(
    samples,
    lambdas,
    propagate,
    log_prob,
    resample,
):
    n = len(samples)
    log_weights = np.zeros(n)

    traj = [samples]
    ancestry_traj = [np.arange(n)]
    log_weights_traj = [np.array(log_weights)]
    incremental_log_weights_traj = (
        []
    )  # TODO: this is redundant -- can be reconstructed from log_weights_traj and ancestry...

    trange = tqdm(lambdas[:-2])
    for (lam_initial, lam_target) in zip(trange, lambdas[1:-1]):
        # update log weights
        incremental_log_weights = log_prob(traj[-1], lam_target) - log_prob(traj[-1], lam_initial)
        log_weights += incremental_log_weights

        # resample
        indices, log_weights = resample(log_weights)
        resampled = [traj[-1][i] for i in indices]

        # propagate
        samples = propagate(resampled, lam_target)

        trange.set_postfix(
            # max_log_weight=max(log_weights),
            # min_log_weight=min(log_weights),
            # u_timings=summarize(u_timings),
            move_timings=summarize(move_timings),
            # EXP=-logsumexp(log_weights - np.log(len(log_weights)))
        )

        # append to trajs
        traj.append(samples)
        ancestry_traj.append(indices)
        log_weights_traj.append(np.array(log_weights))
        incremental_log_weights_traj.append(np.array(incremental_log_weights))

    # final result: a collection of samples, with associated log weights
    log_weights += log_prob(samples, lambdas[-1]) - log_prob(samples, lambdas[-2])
    log_weights_traj.append(np.array(log_weights))

    # don't cast xvb list to array, but cast everything else to arrays
    return_lists = [log_weights_traj, ancestry_traj, incremental_log_weights_traj]
    return_arrays = [np.array(t) for t in return_lists]
    return tuple([traj] + return_arrays)


def null_resample(log_weights):
    """No interaction"""
    indices = onp.arange(len(log_weights))
    return indices, log_weights


def multinomial_resample(log_weights):
    weights = np.exp(log_weights - logsumexp(log_weights))
    n = len(log_weights)
    indices = onp.random.choice(np.arange(n), size=n, p=weights)

    # new weights
    avg_log_weights = logsumexp(log_weights - np.log(n)) * np.ones(n)

    return indices, avg_log_weights


def ess(log_weights):
    weights = np.exp(log_weights - logsumexp(log_weights))
    return 1 / np.sum(weights ** 2)


def fractional_ess(log_weights):
    n = len(log_weights)
    return ess(log_weights) / n


def conditional_multinomial_resample(log_weights, thresh=0.5):
    """if fractional_ess(log_weights) < thresh, multinomial_resample"""
    frac_ess = fractional_ess(log_weights)
    if frac_ess < thresh:
        # print(f'fractional ESS = {frac_ess:.3f} < {thresh:.3f} -- resampling!')
        return multinomial_resample(log_weights)
    else:
        return null_resample(log_weights)


# TODO: stratified resampling, with a sorting function

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
        timings = [timing for (x_next, timing) in results]
        move_timings.extend(timings)

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

    np.save("move_timings.npy", np.array(move_timings))

    with open(f"smc_result_{datetime.now()}.pkl", "wb") as f:
        dump((smc_result, config), f)

    traj, log_weights_traj, ancestry_traj, incremental_log_weights_traj = smc_result
    # TODO: analyze me

# TODO: use less disk space!
