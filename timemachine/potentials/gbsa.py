# taken from josh fass's code but fixed a couple of bugs
# https://github.com/openforcefield/bayes-implicit-solvent/blob/propertycalculator/bayes_implicit_solvent/gb_models/jax_gb_models.py

import jax.numpy as np
from jax import grad, jit
from timemachine.potentials.jax_utils import delta_r, distance, lambda_to_w, convert_to_4d

def step(x):
    # return (x > 0)
    return 1.0 * (x >= 0)

def gbsa_obc(
    coords,
    params,
    lamb,
    box,
    charge_idxs,
    radii_idxs,
    scale_idxs,
    alpha,
    beta,
    gamma,
    cutoff_radii,
    cutoff_force,
    lambda_idxs,
    dielectric_offset=0.009,
    surface_tension=28.3919551,
    solute_dielectric=1.0,
    solvent_dielectric=78.5,
    probe_radius=0.14):

    assert cutoff_radii == cutoff_force

    coords_4d = convert_to_4d(coords, lamb, lambda_idxs, cutoff_radii)

    N = len(radii_idxs)

    radii = params[radii_idxs]
    scales = params[scale_idxs]

    ri = np.expand_dims(coords_4d, 0)
    rj = np.expand_dims(coords_4d, 1)

    dij = distance(ri, rj, box)

    eye = np.eye(N, dtype=dij.dtype)

    r = dij + eye # so I don't have divide-by-zero nonsense
    or1 = radii.reshape((N, 1)) - dielectric_offset
    or2 = radii.reshape((1, N)) - dielectric_offset
    sr2 = scales.reshape((1, N)) * or2

    L = np.maximum(or1, abs(r - sr2))
    U = r + sr2

    I = 1 / L - 1 / U + 0.25 * (r - sr2 ** 2 / r) * (1 / (U ** 2) - 1 / (L ** 2)) + 0.5 * np.log(
        L / U) / r
    # handle the interior case
    I = np.where(or1 < (sr2 - r), I + 2*(1/or1 - 1/L), I)
    I = step(r + sr2 - or1) * 0.5 * I # note the extra 0.5 here
    I -= np.diag(np.diag(I))

    I = np.where(dij > cutoff_radii, 0, I)
    I = np.sum(I, axis=1)

    # okay, next compute born radii
    offset_radius = radii - dielectric_offset

    psi = I * offset_radius

    psi_coefficient = alpha
    psi2_coefficient = beta
    psi3_coefficient = gamma

    psi_term = (psi_coefficient * psi) - (psi2_coefficient * psi ** 2) + (psi3_coefficient * psi ** 3)

    B = 1 / (1 / offset_radius - np.tanh(psi_term) / radii)

    E = 0.0
    # single particle
    # ACE
    E += np.sum(surface_tension * (radii + probe_radius) ** 2 * (radii / B) ** 6)

    # on-diagonal
    charges = params[charge_idxs]
    E += np.sum(-0.5 * (1 / solute_dielectric - 1 / solvent_dielectric) * charges ** 2 / B)

    # particle pair
    f = np.sqrt(r ** 2 + np.outer(B, B) * np.exp(-r ** 2 / (4 * np.outer(B, B))))
    charge_products = np.outer(charges, charges)

    ixns = - (1 / solute_dielectric - 1 / solvent_dielectric) * charge_products / f

    ixns = np.where(dij > cutoff_force, 0, ixns)

    E += np.sum(np.triu(ixns, k=1))

    return E