"""Utilities for computing mapped works or mapped u_kn matrix, incorporating invertible map_fxns"""

import numpy as np
from pymbar.utils import kln_to_kn

__all__ = ["compute_mapped_reduced_work", "compute_mapped_u_kn"]


def compute_mapped_reduced_work(src_samples, src_reduced_energy_fxn, dst_reduced_energy_fxn, map_fxn):
    """Compute minus log importance weights, including a change of variables defined by map_fxn

    References
    ----------
    [Jarzynski, 2002] Targeted free energy perturbation
        https://journals.aps.org/pre/abstract/10.1103/PhysRevE.65.046122
    """

    # compare to version without map_fxn
    # return dst_reduced_energy_fxn(src_samples) - src_reduced_energy_fxn(src_samples)

    mapped_samples, logdetjacs = map_fxn(src_samples)
    return dst_reduced_energy_fxn(mapped_samples) - src_reduced_energy_fxn(src_samples) - logdetjacs


def compute_mapped_u_kn(sample_lists, reduced_energy_fxns, map_fxns):
    """Evaluate every sample from state i in every state j,
    incorporating user-defined change of variables supplied in map_fxns[i, j]

    References
    ----------
    [Paliwal, Shirts, 2013] Multistate reweighting and configuration mapping ...
        https://aip.scitation.org/doi/abs/10.1063/1.4801332

    Notes
    -----
    This function assumes (without verifying) that:
        (1) the inverse of map_fxns[i, j] is map_fxns[j, i],
        (2) the domain of map_fxns[i, j] is the support of state i,
        (3) the codomain of map_fxns[i, j] is the support of state j.
    """

    K = len(sample_lists)
    assert len(reduced_energy_fxns) == K

    N_k = [len(xs_k) for xs_k in sample_lists]
    u_kln = np.zeros((K, K, max(N_k)))

    for k in range(K):
        xs_k = sample_lists[k]

        for l in range(K):
            u_l = reduced_energy_fxns[l]

            # compare to version without maps
            # u_kln[k, l, :N_k[k]] = u_l(xs_k)

            map_k_to_l = map_fxns[k, l]
            xs_mapped, logdetjacs = map_k_to_l(xs_k)
            u_kln[k, l, : N_k[k]] = u_l(xs_mapped) - logdetjacs

    u_kn = kln_to_kn(u_kln, N_k)
    assert u_kn.shape == (K, sum(N_k))
    return u_kn
