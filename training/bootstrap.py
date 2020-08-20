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
    print(triples.ndim)
    if triples.ndim == 2:
        print("SINGLE")
        return np.array([integrate(triples) + ssc])

    # Compute the CI
    print("TUPLE")
    results = []
    for arr in triples:
        dG = integrate(arr)
        results.append(dG)

    print(np.array(results) + ssc)

    return np.array(results) + ssc

def ti_ci(all_du_dls, ssc, stage_lambdas, du_dl_cutoff):
    """
    Compute the bootstrap confidence interval under thermodynamic integration.

    Parameters
    ----------
    all_du_dls: list of np.array [S, L, F, T]
        all_du_dls for each stage

    stage_lambdas: list of np.array [S, L]
        Lambda schedule for each stage

    ssc: float
        Standard state correction

    Returns
    -------
    mean, [lower 95 CI, upper 95 CI]

    """

    multi_stage_avg_du_dls = []

    for stage_du_dls in all_du_dls:
        avg_du_dls = []
        for lamb_full_du_dls in stage_du_dls:
            avg_du_dls.append(np.mean(np.sum(lamb_full_du_dls[:, du_dl_cutoff:], axis=0)))
        avg_du_dls = np.concatenate([avg_du_dls])
        multi_stage_avg_du_dls.append(avg_du_dls)

    # sample from triples
    triples = []
    for stage_idx, (stage_du_dls, lambdas) in enumerate(zip(multi_stage_avg_du_dls, stage_lambdas)):

        for d, l in zip(stage_du_dls, lambdas):
            triples.append((stage_idx, d, l))

    stat_fn = functools.partial(estimate, ssc=ssc)

    return bs.bootstrap(np.array(triples), stat_func=stat_fn)

