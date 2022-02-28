# adapted from https://github.com/proteneer/timemachine/blob/7d6099b0f5b4a2d0b26c3edc7a91c18f7a526c00/md/experimental/smc.py

from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

# type annotations
Sample = Any
Samples = List[Sample]  # e.g. List[CoordsVelBox]

Lambda = LogWeight = float
LogWeights = IndexArray = Array = np.array

BatchPropagator = Callable[[Samples, Lambda], Samples]
BatchLogProb = Callable[[Samples, Lambda], LogWeights]

Resampler = Callable[[LogWeights], Tuple[IndexArray, LogWeights]]

ResultDict = Dict[str, Union[Samples, Array]]


def simple_smc(
    samples: Samples,
    lambdas: Array,
    propagate: BatchPropagator,
    log_prob: BatchLogProb,
    resample: Resampler,
) -> ResultDict:
    """barebones implementation of Sequential Monte Carlo (SMC)

    Parameters
    ----------
    samples: [N,] list
    lambdas: [K,] array
    propagate: function
        [move(x, lam) for x in xs]
        for example, move(x, lam) might mean "run 100 steps of all-atom MD targeting exp(-u(., lam)), initialized at x"
    log_prob: function
        [exp(-u(x, lam)) for x in xs]
    resample: function
        (optionally) perform resampling given an array of log weights

    Returns
    -------
    trajs_dict
        [K-1, N] list of snapshots
        [K-1, N] array of incremental log weights
        [K-1, N] array of ancestor idxs
        [K, N] array of log weights

    References
    ----------
    * Arnaud Doucet's annotated bibliography of SMC
        https://www.stats.ox.ac.uk/~doucet/smc_resources.html
    * [Dai, Heng, Jacob, Whiteley, 2020] An invitation to sequential Monte Carlo samplers
        https://arxiv.org/abs/2007.11936
    """
    n = len(samples)
    log_weights = np.zeros(n)

    # store
    traj = [samples]
    ancestry_traj = [np.arange(n)]
    log_weights_traj = [np.array(log_weights)]
    incremental_log_weights_traj = []  # note: redundant but convenient

    trange = tqdm(lambdas[:-2])

    def logging_stuff(samples, indices, log_weights, incremental_log_weights):
        traj.append(samples)
        ancestry_traj.append(indices)
        log_weights_traj.append(np.array(log_weights))
        incremental_log_weights_traj.append(np.array(incremental_log_weights))
        running_estimate = -logsumexp(log_weights - np.log(len(log_weights)))
        trange.set_postfix(EXP=running_estimate)

    # main loop
    for (lam_initial, lam_target) in zip(trange, lambdas[1:-1]):
        # update log weights
        incremental_log_weights = log_prob(traj[-1], lam_target) - log_prob(traj[-1], lam_initial)
        log_weights += incremental_log_weights

        # resample
        indices, log_weights = resample(log_weights)
        resampled = [traj[-1][i] for i in indices]

        # propagate
        samples = propagate(resampled, lam_target)

        # log
        logging_stuff(samples, indices, log_weights, incremental_log_weights)

    # final result: a collection of samples, with associated log weights
    incremental_log_weights = log_prob(samples, lambdas[-1]) - log_prob(samples, lambdas[-2])
    incremental_log_weights_traj.append(incremental_log_weights)
    log_weights_traj.append(np.array(log_weights + incremental_log_weights))

    # cast everything (except samples list) to arrays
    trajs_dict = dict(
        traj=traj,
        log_weights_traj=np.array(log_weights_traj),
        ancestry_traj=np.array(ancestry_traj),
        incremental_log_weights_traj=np.array(incremental_log_weights_traj),
    )
    return trajs_dict


def null_resample(log_weights):
    """No interaction"""
    indices = np.arange(len(log_weights))
    return indices, log_weights


def multinomial_resample(log_weights):
    """sample proportional to exp(log_weights), with replacement"""
    weights = np.exp(log_weights - logsumexp(log_weights))
    n = len(log_weights)
    indices = np.random.choice(np.arange(n), size=n, p=weights)

    # new weights
    avg_log_weights = logsumexp(log_weights - np.log(n)) * np.ones(n)

    return indices, avg_log_weights


def ess(log_weights):
    """effective sample size, taking values in interval [1, len(log_weights)]"""
    weights = np.exp(log_weights - logsumexp(log_weights))
    return 1 / np.sum(weights ** 2)


def fractional_ess(log_weights):
    """effective sample size, normalized to interval (0, 1]"""
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
