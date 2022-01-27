import numpy as np
import numpy as onp
from scipy.special import logsumexp
from tqdm import tqdm


# based on https://github.com/proteneer/timemachine/blob/7d6099b0f5b4a2d0b26c3edc7a91c18f7a526c00/md/experimental/smc.py

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
            #move_timings=summarize(move_timings),
            EXP=-logsumexp(log_weights - np.log(len(log_weights)))
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
