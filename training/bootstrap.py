# bootstrap TI estimate of free energy

import numpy as np

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

import functools

def integrate(triples):

    stages = {}
    for s, du_dl, lamb in triples:
        if s not in stages:
            stages[s] = []
        stages[s].append((du_dl, lamb))

    dGs = []
    for _, v in stages.items():
        lambdas = []
        avg_du_dls = []
        for du_dl, lamb in v:
            lambdas.append(lamb)
            avg_du_dls.append(du_dl)
        lambdas = np.array(lambdas)

        avg_du_dls = np.array(avg_du_dls)
        perm = np.argsort(lambdas)
        avg_du_dls = avg_du_dls[perm]
        lambdas = lambdas[perm]
        dGs.append(np.trapz(avg_du_dls, lambdas))

    return np.sum(dGs)

def estimate(triples, ssc):

    # Compute the mean
    if triples.ndim == 2:
        return np.array([integrate(triples)]) + ssc

    # Compute the CI
    results = []
    for arr in triples:
        dG = integrate(arr)
        results.append(dG)

    return np.array(results) +   ssc

def ti_ci(all_du_dls, stage_lambdas, ssc, du_dl_cutoff):
    """
    Compute the bootstrap confidence interval under thermodynamic integration.

    Parameters
    ----------
    avg_du_dls: list of np.array
        Average du_dls for each stage

    stage_lambdas: list of np.array
        Lambda schedule for each stage

    ssc: float
        Standard state correction

    Returns
    -------
    mean, [lower 95 CI, upper 95 CI]

    """

    avg_du_dls = []
    for du_dls in all_du_dls:
        avg_du_dls.append(np.sum(lamb_full_du_dls[:, du_dl_cutoff:], axis=0))

    triples = []
    for stage_idx, (stage_lambdas, stage_du_dls) in enumerate(zip(avg_du_dls, stage_lambdas)):
        for d, l in zip(stage_lambdas, stage_du_dls):
            triples.append((stage_idx, d, l))

    stat_fn = functools.partial(estimate, ssc=ssc)

    return bs.bootstrap(np.array(triples), stat_func=stat_fn)


# stage_0_du_dls = np.random.rand(10)
# stage_0_lambdas = np.linspace(0, 1, 10)

# stage_1_du_dls = np.load("/home/yutong/Downloads/67.npy")
# stage_1_lambdas = np.concatenate([
#     np.linspace(0, 0.15, 40, endpoint=False),
#     np.linspace(0.15, 0.33, 160, endpoint=False),
#     np.linspace(0.33, 0.6, 100, endpoint=False),
#     np.linspace(0.6, 1.5, 20),
# ])
