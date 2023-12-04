from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc
from numpy.typing import NDArray

from timemachine.potentials.jax_utils import (
    delta_r,
    distance_on_pairs,
    pairs_from_interaction_groups,
    pairwise_distances,
)

Array = Any


def combining_rule_sigma(sig_i, sig_j):
    """Lorentz-Berthelot (sig_i + sig_j) / 2,
    but assuming sig -> (sig / 2) has been applied
    https://github.com/proteneer/timemachine/blob/2d0ef5bb45e034e33074f50e8e0a13189f1b0630/timemachine/ff/handlers/nonbonded.py#L334
    """
    return sig_i + sig_j


def combining_rule_epsilon(eps_i, eps_j):
    """Lorentz-Berthelot sqrt(eps_i * eps_j),
    but assuming eps -> sqrt(eps) has been applied
    (point 2 of https://github.com/proteneer/timemachine#forcefield-gotchas)
    """
    return eps_i * eps_j


def lennard_jones(dij, sig_ij, eps_ij):
    """https://en.wikipedia.org/wiki/Lennard-Jones_potential"""
    sig6 = (sig_ij / dij) ** 6
    sig12 = sig6**2

    return 4 * eps_ij * (sig12 - sig6)


def direct_space_pme(dij, qij, beta):
    """Direct-space contribution from eq 2 of:
    Darden, York, Pedersen, 1993, J. Chem. Phys.
    "Particle mesh Ewald: An N log(N) method for Ewald sums in large systems"
    https://aip.scitation.org/doi/abs/10.1063/1.470117
    """
    return qij * erfc(beta * dij) / dij


def nonbonded_block_unsummed(xi, xj, box, params_i, params_j, beta, cutoff):
    """
    This is a modified version of `nonbonded` that computes a block of
    NxM interactions between two sets of particles x_i and x_j. It is assumed that
    there are no exclusions between the two particle sets. Typical use cases
    include computing the interaction energy between the environment and a
    ligand.

    This is mainly used for testing, as it does not support 4D decoupling or
    alchemical semantics yet.

    Parameters
    ----------
    xi : (N,3) np.ndarray
        Coordinates
    xj : (M,3) np.ndarray
        Coordinates
    box : Optional 3x3 np.ndarray
        Periodic boundary conditions
    params_i : (N, 3) np.ndarray
        3-Tuples of (charge, sigma, epsilons)
    params_j : (M, 3) np.ndarray
        3-Tuples of (charge, sigma, epsilons)
    beta : float
        the charge product q_ij will be multiplied by erfc(beta*d_ij)
    cutoff : Optional float
        a pair of particles (i,j) will be considered non-interacting if the distance d_ij
        between their 3D coordinates exceeds cutoff

    Returns
    -------
    (N,M) np.ndarray
        Interaction energy block

    """
    ri = jnp.expand_dims(xi, axis=1)
    rj = jnp.expand_dims(xj, axis=0)

    dij = jnp.linalg.norm(delta_r(ri, rj, box), axis=-1)
    sig_i = jnp.expand_dims(params_i[:, 1], axis=1)
    sig_j = jnp.expand_dims(params_j[:, 1], axis=0)
    eps_i = jnp.expand_dims(params_i[:, 2], axis=1)
    eps_j = jnp.expand_dims(params_j[:, 2], axis=0)

    sig_ij = combining_rule_sigma(sig_i, sig_j)
    eps_ij = combining_rule_epsilon(eps_i, eps_j)

    qi = jnp.expand_dims(params_i[:, 0], axis=1)
    qj = jnp.expand_dims(params_j[:, 0], axis=0)

    qij = jnp.multiply(qi, qj)

    es = direct_space_pme(dij, qij, beta)
    lj = lennard_jones(dij, sig_ij, eps_ij)

    nrgs = jnp.where(dij < cutoff, es + lj, 0)
    return nrgs


def nonbonded_block(xi, xj, box, params_i, params_j, beta, cutoff):
    """
    This is a summed version of nonbonded_block_unsummed, returning a scalar
    """
    return jnp.sum(nonbonded_block_unsummed(xi, xj, box, params_i, params_j, beta, cutoff))


def convert_exclusions_to_rescale_masks(exclusion_idxs, scales, N):
    """Converts exclusions from list format used in Nonbonded to mask format used in `nonbonded`"""

    # process masks for exclusions properly
    charge_rescale_mask = np.ones((N, N))  # to support item assignment
    for (i, j), exc in zip(exclusion_idxs, scales[:, 0]):
        charge_rescale_mask[i][j] = 1 - exc
        charge_rescale_mask[j][i] = 1 - exc

    lj_rescale_mask = np.ones((N, N))
    for (i, j), exc in zip(exclusion_idxs, scales[:, 1]):
        lj_rescale_mask[i][j] = 1 - exc
        lj_rescale_mask[j][i] = 1 - exc

    return charge_rescale_mask, lj_rescale_mask


def filter_exclusions(
    atom_idxs: NDArray[np.int32],
    exclusion_idxs: NDArray[np.int32],
    scale_factors: NDArray[np.float64],
    update_idxs: bool = False,
):
    """
    Return the exclusions and corresponding scale factors
    with the atoms not in atom_idxs removed.

    Parameters
    ----------
    atom_idxs:
        List of atoms that are considered for interaction.

    exclusion_idxs:
        List of atom pairs to exclude.

    scale_factors:
        Per exclusion charge and lj scale factors.

    update_idxs:
        Set to True to remap the exclusion indexes
        to point to the index of atom_idxs.
        This can be used for the reference JAX potential.
    """
    atom_idxs_set = set(atom_idxs)
    map_to_filtered = {j: i for i, j in enumerate(atom_idxs)}
    filtered_exclusion_idxs_ = []
    filtered_scale_factors_ = []
    for (i, j), sf in zip(exclusion_idxs, scale_factors):
        if i not in atom_idxs_set or j not in atom_idxs_set:
            continue
        if update_idxs:
            i, j = map_to_filtered[i], map_to_filtered[j]
        filtered_exclusion_idxs_.append((i, j))
        filtered_scale_factors_.append(sf)

    filtered_exclusion_idxs = np.array(filtered_exclusion_idxs_, dtype=np.int32)
    filtered_scale_factors = np.array(filtered_scale_factors_)
    if not filtered_scale_factors.shape[0]:
        filtered_scale_factors = filtered_scale_factors.reshape((0, scale_factors.shape[1]))
    return filtered_exclusion_idxs, filtered_scale_factors


def nonbonded(
    conf,
    params,
    box,
    exclusion_idxs: NDArray[np.int32],
    scale_factors: NDArray[np.float64],
    beta,
    cutoff,
    runtime_validate=True,
    atom_idxs=None,
):
    """Lennard-Jones + Coulomb, with a few important twists:
    * distances are computed in 4D using coordinates in params
    * each pairwise LJ and Coulomb term can be multiplied by an adjustable rescale_mask parameter
    * Coulomb terms are multiplied by erfc(beta * distance)

    Parameters
    ----------
    conf : (N, 3) np.ndarray
        3D coordinates
    params : (N, 3) np.ndarray
        columns [charges, sigmas, epsilons, w_coords], one row per particle
    box : Optional 3x3 np.ndarray
    exclusion_idxs:
        List of atom pairs to exclude.
    scale_factors:
        Per exclusion charge and lj scale factors.
    beta : float
        the charge product q_ij will be multiplied by erfc(beta*d_ij)
    cutoff : Optional float
        a pair of particles (i,j) will be considered non-interacting if the distance d_ij
        between their 4D coordinates exceeds cutoff
    runtime_validate: bool
        check whether beta is compatible with cutoff
        (if True, this function will currently not play nice with Jax JIT)
        TODO: is there a way to conditionally print a runtime warning inside
            of a Jax JIT-compiled function, without triggering a Jax ConcretizationTypeError?
    atom_idxs: NDArray[int32]
        Subset of atoms to consider for the interaction or None to consider all atoms.

    Returns
    -------
    energy : float

    References
    ----------
    * Rodinger, Howell, Pomès, 2005, J. Chem. Phys. "Absolute free energy calculations by thermodynamic integration in four spatial
        dimensions" https://aip.scitation.org/doi/abs/10.1063/1.1946750
    * Darden, York, Pedersen, 1993, J. Chem. Phys. "Particle mesh Ewald: An N log(N) method for Ewald sums in large
    systems" https://aip.scitation.org/doi/abs/10.1063/1.470117
        * Coulomb interactions are treated using the direct-space contribution from eq 2
    """
    # If requested, filter to a subset of interacting atoms
    if atom_idxs is not None:
        conf = jnp.array(conf)[atom_idxs, :]
        params = jnp.array(params)[atom_idxs, :]
        exclusion_idxs, scale_factors = filter_exclusions(atom_idxs, exclusion_idxs, scale_factors, update_idxs=True)

    N = conf.shape[0]
    charge_rescale_mask, lj_rescale_mask = convert_exclusions_to_rescale_masks(exclusion_idxs, scale_factors, N)

    if runtime_validate:
        assert (charge_rescale_mask == charge_rescale_mask.T).all()
        assert (lj_rescale_mask == lj_rescale_mask.T).all()

    charges = params[:, 0]
    sig = params[:, 1]
    eps = params[:, 2]
    w_coords = params[:, 3]

    sig_i = jnp.expand_dims(sig, 0)
    sig_j = jnp.expand_dims(sig, 1)
    sig_ij = combining_rule_sigma(sig_i, sig_j)

    eps_i = jnp.expand_dims(eps, 0)
    eps_j = jnp.expand_dims(eps, 1)
    eps_ij = combining_rule_epsilon(eps_i, eps_j)

    dij = pairwise_distances(conf, box, w_coords)

    keep_mask = jnp.ones((N, N)) - jnp.eye(N)
    keep_mask = jnp.where(eps_ij != 0, keep_mask, 0)

    if cutoff is not None:
        if runtime_validate:
            validate_coulomb_cutoff(cutoff, beta, threshold=1e-2)
        eps_ij = jnp.where(dij < cutoff, eps_ij, 0)

    # (ytz): this avoids a nan in the gradient in both jax and tensorflow
    sig_ij = jnp.where(keep_mask, sig_ij, 0)
    eps_ij = jnp.where(keep_mask, eps_ij, 0)

    inv_dij = 1 / dij
    inv_dij = jnp.where(jnp.eye(N), 0, inv_dij)

    sig2 = sig_ij * inv_dij
    sig2 *= sig2
    sig6 = sig2 * sig2 * sig2

    eij_lj = 4 * eps_ij * (sig6 - 1.0) * sig6
    eij_lj = jnp.where(keep_mask, eij_lj, 0)

    qi = jnp.expand_dims(charges, 0)  # (1, N)
    qj = jnp.expand_dims(charges, 1)  # (N, 1)
    qij = jnp.multiply(qi, qj)

    # (ytz): trick used to avoid nans in the diagonal due to the 1/dij term.
    keep_mask = 1 - jnp.eye(N)
    qij = jnp.where(keep_mask, qij, 0)
    dij = jnp.where(keep_mask, dij, 0)

    # funny enough lim_{x->0} erfc(x)/x = 0
    eij_charge = jnp.where(keep_mask, qij * erfc(beta * dij) * inv_dij, 0)  # zero out diagonals
    if cutoff is not None:
        eij_charge = jnp.where(dij < cutoff, eij_charge, 0)

    eij_total = eij_lj * lj_rescale_mask + eij_charge * charge_rescale_mask

    return jnp.sum(eij_total / 2)


def nonbonded_on_specific_pairs(
    conf,
    params,
    box,
    pairs,
    beta: float,
    cutoff: Optional[float] = None,
    rescale_mask=None,
):
    """See `nonbonded` docstring for more details

    Notes
    -----
    * Warning! This function performs no validation of pair indices. If the provided pairs are incomplete (e.g. omitting
        some pairs of atoms that could be within cutoff, or omitting intramolecular pairs, ...), then incorrect results
        can be returned.
    """

    if len(pairs) == 0:
        return np.zeros(1), np.zeros(1)

    inds_l, inds_r = pairs.T

    charges, sig, eps, w_coords = params.T

    # distances and cutoff
    w_offsets = w_coords[pairs[:, 0]] - w_coords[pairs[:, 1]] if w_coords is not None else None
    dij = distance_on_pairs(conf[inds_l], conf[inds_r], box, w_offsets)
    if cutoff is None:
        cutoff = np.inf
    keep_mask = dij <= cutoff

    def apply_cutoff(x):
        return jnp.where(keep_mask, x, 0)

    # vdW by Lennard-Jones
    sig_ij = combining_rule_sigma(sig[inds_l], sig[inds_r])
    sig_ij = apply_cutoff(sig_ij)

    eps_ij = combining_rule_epsilon(eps[inds_l], eps[inds_r])
    eps_ij = apply_cutoff(eps_ij)

    vdW = jnp.where(eps_ij != 0, lennard_jones(dij, sig_ij, eps_ij), 0)

    # Electrostatics by direct-space part of PME
    qij = apply_cutoff(charges[inds_l] * charges[inds_r])
    electrostatics = direct_space_pme(dij, qij, beta)

    if rescale_mask is not None:
        assert rescale_mask.shape == (len(pairs), 2)
        rescale_vdW = rescale_mask[:, 1]
        vdW = jnp.where(rescale_vdW != 0, vdW * rescale_vdW, 0)
        rescale_electrostatics = rescale_mask[:, 0]
        electrostatics = jnp.where(rescale_electrostatics != 0, electrostatics * rescale_electrostatics, 0)

    return vdW, electrostatics


def nonbonded_on_precomputed_pairs(
    conf,
    params,
    box,
    pairs,
    beta: float,
    cutoff: Optional[float] = None,
):
    """
    Similar to pairlist, except that we pre-compute parameters with:

    1) Broadcast parameters using pairs
    2) Apply combining rules to charges, epsilsons, and broadcast
    3) Apply rescale_mask to combined q_ij and eps_ij

    conf: N,3
    params: P,4 (q_ij, s_ij, e_ij, w_offset_ij)
    pairs: P,2 (i,j)
    """

    if len(pairs) == 0:
        return np.zeros(1), np.zeros(1)

    inds_l, inds_r = pairs.T

    # distances and cutoff
    q_ij, sig_ij, eps_ij, offsets = params.T
    dij = distance_on_pairs(conf[inds_l], conf[inds_r], box, offsets)
    if cutoff is None:
        cutoff = np.inf

    keep_mask = dij <= cutoff

    def apply_cutoff(x):
        return jnp.where(keep_mask, x, 0)

    q_ij = apply_cutoff(q_ij)
    sig_ij = apply_cutoff(sig_ij)
    eps_ij = apply_cutoff(eps_ij)

    vdW = jnp.where(eps_ij != 0, lennard_jones(dij, sig_ij, eps_ij), 0)
    electrostatics = jnp.where(q_ij != 0, direct_space_pme(dij, q_ij, beta), 0)

    return vdW, electrostatics


def validate_interaction_group_idxs(n_atoms, a_idxs, b_idxs):
    """assert A and B are disjoint, contain no elements outside range(0, n_atoms), and contain no repeats"""
    A, B = set(a_idxs), set(b_idxs)
    AB = A.union(B)
    assert A.isdisjoint(B)
    assert max(AB) < n_atoms
    assert min(AB) >= 0
    assert len(a_idxs) == len(A)
    assert len(b_idxs) == len(B)


def nonbonded_interaction_groups(
    conf,
    params,
    box,
    a_idxs,
    b_idxs,
    beta: float,
    cutoff: Optional[float] = None,
):
    """Nonbonded interactions between all pairs of atoms $(i, j)$
    where $i$ is in the first set and $j$ in the second.

    See `nonbonded` docstring for more details
    """
    # If not set, col_atom_idxs are all others not in the row
    num_atoms = len(conf)
    if b_idxs is None:
        b_idxs = np.setdiff1d(jnp.arange(num_atoms), a_idxs)
    validate_interaction_group_idxs(num_atoms, a_idxs, b_idxs)
    pairs = pairs_from_interaction_groups(a_idxs, b_idxs)
    vdW, electrostatics = nonbonded_on_specific_pairs(conf, params, box, pairs, beta, cutoff)
    return vdW, electrostatics


def validate_coulomb_cutoff(cutoff=1.0, beta=2.0, threshold=1e-2):
    """check whether f(r) = erfc(beta * r) <= threshold at r = cutoff
    following https://github.com/proteneer/timemachine/pull/424#discussion_r629678467"""
    if erfc(beta * cutoff) > threshold:
        print(UserWarning(f"erfc(beta * cutoff) = {erfc(beta * cutoff)} > threshold = {threshold}"))
