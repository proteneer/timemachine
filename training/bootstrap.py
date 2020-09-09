# (ytz): bootstrap TI estimate of free energy, this is fairly redundant
# we need to cleanup/refactor this with the actual estimater code so
# it's jess janky

import numpy as np

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

import functools


# def integrate(triples):

#     stages = {}
#     for s, du_dl, lamb in triples:
#         if s not in stages:
#             stages[s] = []
#         stages[s].append((du_dl, lamb))

#     dGs = []
#     for _, v in stages.items():
#         lambdas = []
#         avg_du_dls = []
#         for du_dl, lamb in v:
#             lambdas.append(lamb)
#             avg_du_dls.append(du_dl)
#         lambdas = np.array(lambdas)

#         avg_du_dls = np.array(avg_du_dls)
#         perm = np.argsort(lambdas)
#         avg_du_dls = avg_du_dls[perm]
#         lambdas = lambdas[perm]
#         dGs.append(np.trapz(avg_du_dls, lambdas))

#     return np.sum(dGs)

# def estimate(triples, ssc):

#     # Compute the mean
#     if triples.ndim == 2:
#         return np.array([integrate(triples) + ssc])

#     # Compute the CI
#     results = []
#     for arr in triples:
#         dG = integrate(arr)
#         results.append(dG)

#     return np.array(results) + ssc

# tbd: chain bootstrapped estimators

def bs_integrate(tuples):
    """
    Parameters
    ----------

    tuples: tuples of du_dl, lamb

    """
    lambdas = []
    du_dls = []
    for du_dl, lamb in tuples:
        lambdas.append(lamb)
        du_dls.append(du_dl)

    perm = np.argsort(lambdas)
    du_dls = np.array(du_dls)
    du_dls = du_dls[perm]
    lambdas = lambdas[perm]

    return np.trapz(avg_du_dls, lambdas)

def ti_ci(du_dls, lambda_schedule):

    tuples = []
    for du_dl, lamb in zip(du_dls, lambda_schedule):
        tuples.append((du_dl, lamb))

    return bs.bootstrap(np.array(tuples), stat_func=bs_integrate)

# def ti_ci(all_du_dls, ssc, stage_lambdas, du_dl_cutoff):
#     """
#     Compute the bootstrap confidence interval under thermodynamic integration.

#     Parameters
#     ----------
#     all_du_dls: list of np.array [S, L, F, T]
#         all_du_dls for each stage

#     stage_lambdas: list of np.array [S, L]
#         Lambda schedule for each stage

#     ssc: float
#         Standard state correction

#     Returns
#     -------
#     mean, [lower 95 CI, upper 95 CI]

#     """

#     multi_stage_avg_du_dls = []

#     # compute the means
#     for stage_du_dls in all_du_dls:
#         avg_du_dls = []
#         for lamb_full_du_dls in stage_du_dls:
#             avg_du_dls.append(np.mean(np.sum(lamb_full_du_dls[:, du_dl_cutoff:], axis=0)))
#         avg_du_dls = np.concatenate([avg_du_dls])
#         multi_stage_avg_du_dls.append(avg_du_dls)

#     # sample from triples
#     triples = []
#     for stage_idx, (stage_du_dls, lambdas) in enumerate(zip(multi_stage_avg_du_dls, stage_lambdas)):

#         for d, l in zip(stage_du_dls, lambdas):
#             triples.append((stage_idx, d, l))

#     stat_fn = functools.partial(estimate, ssc=ssc)

#     return bs.bootstrap(np.array(triples), stat_func=stat_fn)

