import functools

import jax.numpy as np
import numpy as onp
from jax import vmap
from jax.scipy.special import erfc

from timemachine.potentials import jax_utils
from timemachine.potentials.jax_utils import (
    convert_to_4d,
    delta_r,
    distance,
    distance_on_pairs,
    pairs_from_interaction_groups,
)


def switch_fn(dij, cutoff):
    return np.power(np.cos((np.pi * np.power(dij, 8)) / (2 * cutoff)), 2)


from typing import Optional

Array = np.array


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
    sig12 = sig6 ** 2

    return 4 * eps_ij * (sig12 - sig6)


def direct_space_pme(dij, qij, beta):
    """Direct-space contribution from eq 2 of:
    Darden, York, Pedersen, 1993, J. Chem. Phys.
    "Particle mesh Ewald: An N log(N) method for Ewald sums in large systems"
    https://aip.scitation.org/doi/abs/10.1063/1.470117
    """
    return qij * erfc(beta * dij) / dij


def nonbonded_block(xi, xj, box, params_i, params_j, beta, cutoff):
    """
    This is a modified version of nonbonded_v3 that computes a block of
    interactions between two sets of particles x_i and x_j. It is assumed that
    there are no exclusions between the two particle sets. Typical use cases
    include computing the interaction energy between the environment and a ligand.

    This is mainly used for testing, as it does not support 4D decoupling or
    alchemical semantics yet.

    Parameters
    ----------
    xi : (N,3) np.ndarray
        Coordinates
    xj : (N,3) np.ndarray
        Coordinates
    box : Optional 3x3 np.array
        Periodic boundary conditions
    beta : float
        the charge product q_ij will be multiplied by erfc(beta*d_ij)
    cutoff : Optional float
        a pair of particles (i,j) will be considered non-interacting if the distance d_ij
        between their 3D coordinates exceeds cutoff

    Returns
    -------
    scalar
        Interaction energy

    """
    ri = np.expand_dims(xi, axis=1)
    rj = np.expand_dims(xj, axis=0)
    d2ij = np.sum(np.power(delta_r(ri, rj, box), 2), axis=-1)
    dij = np.sqrt(d2ij)
    sig_i = np.expand_dims(params_i[:, 1], axis=1)
    sig_j = np.expand_dims(params_j[:, 1], axis=0)
    eps_i = np.expand_dims(params_i[:, 2], axis=1)
    eps_j = np.expand_dims(params_j[:, 2], axis=0)

    sig_ij = combining_rule_sigma(sig_i, sig_j)
    eps_ij = combining_rule_epsilon(eps_i, eps_j)

    qi = np.expand_dims(params_i[:, 0], axis=1)
    qj = np.expand_dims(params_j[:, 0], axis=0)

    qij = np.multiply(qi, qj)

    es = direct_space_pme(dij, qij, beta)
    lj = lennard_jones(dij, sig_ij, eps_ij)

    nrg = np.where(dij > cutoff, 0, es + lj)
    return np.sum(nrg)


def convert_exceptions_to_rescale_masks(exclusion_idxs, scales, N):
    """Converts exceptions / exclusions from list format used in Nonbonded to mask format used in nonbonded_v3"""

    # process masks for exclusions properly
    charge_rescale_mask = onp.ones((N, N))  # to support item assignment
    for (i, j), exc in zip(exclusion_idxs, scales[:, 0]):
        charge_rescale_mask[i][j] = 1 - exc
        charge_rescale_mask[j][i] = 1 - exc

    lj_rescale_mask = onp.ones((N, N))
    for (i, j), exc in zip(exclusion_idxs, scales[:, 1]):
        lj_rescale_mask[i][j] = 1 - exc
        lj_rescale_mask[j][i] = 1 - exc

    return charge_rescale_mask, lj_rescale_mask


def nonbonded_v3(
    conf,
    params,
    box,
    lamb,
    charge_rescale_mask,
    lj_rescale_mask,
    beta,
    cutoff,
    lambda_plane_idxs,
    lambda_offset_idxs,
    runtime_validate=True,
):
    """Lennard-Jones + Coulomb, with a few important twists:
    * distances are computed in 4D, controlled by lambda, lambda_plane_idxs, lambda_offset_idxs
    * each pairwise LJ and Coulomb term can be multiplied by an adjustable rescale_mask parameter
    * Coulomb terms are multiplied by erfc(beta * distance)

    Parameters
    ----------
    conf : (N, 3) or (N, 4) np.array
        3D or 4D coordinates
        if 3D, will be converted to 4D using (x,y,z) -> (x,y,z,w)
            where w = cutoff * (lambda_plane_idxs + lambda_offset_idxs * lamb)
    params : (N, 3) np.array
        columns [charges, sigmas, epsilons], one row per particle
    box : Optional 3x3 np.array
    lamb : float
    charge_rescale_mask : (N, N) np.array
        the Coulomb contribution of pair (i,j) will be multiplied by charge_rescale_mask[i,j]
    lj_rescale_mask : (N, N) np.array
        the Lennard-Jones contribution of pair (i,j) will be multiplied by lj_rescale_mask[i,j]
    beta : float
        the charge product q_ij will be multiplied by erfc(beta*d_ij)
    cutoff : Optional float
        a pair of particles (i,j) will be considered non-interacting if the distance d_ij
        between their 4D coordinates exceeds cutoff
    lambda_plane_idxs : Optional (N,) np.array
    lambda_offset_idxs : Optional (N,) np.array
    runtime_validate: bool
        check whether beta is compatible with cutoff
        (if True, this function will currently not play nice with Jax JIT)
        TODO: is there a way to conditionally print a runtime warning inside
            of a Jax JIT-compiled function, without triggering a Jax ConcretizationTypeError?

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
    if runtime_validate:
        assert (charge_rescale_mask == charge_rescale_mask.T).all()
        assert (lj_rescale_mask == lj_rescale_mask.T).all()

    N = conf.shape[0]

    if conf.shape[-1] == 3:
        conf = convert_to_4d(conf, lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)

    # make 4th dimension of box large enough so its roughly aperiodic
    if box is not None:
        if box.shape[-1] == 3:
            box_4d = np.eye(4) * 1000
            box_4d = box_4d.at[:3, :3].set(box)
        else:
            box_4d = box
    else:
        box_4d = None

    box = box_4d

    charges = params[:, 0]
    sig = params[:, 1]
    eps = params[:, 2]

    sig_i = np.expand_dims(sig, 0)
    sig_j = np.expand_dims(sig, 1)
    sig_ij = combining_rule_sigma(sig_i, sig_j)

    eps_i = np.expand_dims(eps, 0)
    eps_j = np.expand_dims(eps, 1)

    eps_ij = combining_rule_epsilon(eps_i, eps_j)

    dij = distance(conf, box)

    keep_mask = np.ones((N, N)) - np.eye(N)
    keep_mask = np.where(eps_ij != 0, keep_mask, 0)

    if cutoff is not None:
        if runtime_validate:
            validate_coulomb_cutoff(cutoff, beta, threshold=1e-2)
        eps_ij = np.where(dij < cutoff, eps_ij, 0)

    # (ytz): this avoids a nan in the gradient in both jax and tensorflow
    sig_ij = np.where(keep_mask, sig_ij, 0)
    eps_ij = np.where(keep_mask, eps_ij, 0)

    inv_dij = 1 / dij
    inv_dij = np.where(np.eye(N), 0, inv_dij)

    sig2 = sig_ij * inv_dij
    sig2 *= sig2
    sig6 = sig2 * sig2 * sig2

    eij_lj = 4 * eps_ij * (sig6 - 1.0) * sig6
    eij_lj = np.where(keep_mask, eij_lj, 0)

    qi = np.expand_dims(charges, 0)  # (1, N)
    qj = np.expand_dims(charges, 1)  # (N, 1)
    qij = np.multiply(qi, qj)

    # (ytz): trick used to avoid nans in the diagonal due to the 1/dij term.
    keep_mask = 1 - np.eye(N)
    qij = np.where(keep_mask, qij, 0)
    dij = np.where(keep_mask, dij, 0)

    # funny enough lim_{x->0} erfc(x)/x = 0
    eij_charge = np.where(keep_mask, qij * erfc(beta * dij) * inv_dij, 0)  # zero out diagonals
    if cutoff is not None:
        eij_charge = np.where(dij > cutoff, 0, eij_charge)

    eij_total = eij_lj * lj_rescale_mask + eij_charge * charge_rescale_mask

    return np.sum(eij_total / 2)


def nonbonded_v3_on_specific_pairs(conf, params, box, pairs, beta: float, cutoff: Optional[float] = None):
    """See nonbonded_v3 docstring for more details

    Notes
    -----
    * Responsibility of caller to ensure pair indices are complete.
    """

    inds_l, inds_r = pairs.T

    # distances and cutoff
    dij = distance_on_pairs(conf[inds_l], conf[inds_r], box)
    if cutoff is None:
        cutoff = np.inf
    keep_mask = dij <= cutoff

    def apply_cutoff(x):
        return np.where(keep_mask, x, 0)

    charges, sig, eps = params.T

    # vdW by Lennard-Jones
    sig_ij = combining_rule_sigma(sig[inds_l], sig[inds_r])
    sig_ij = apply_cutoff(sig_ij)

    eps_ij = combining_rule_epsilon(eps[inds_l], eps[inds_r])
    eps_ij = apply_cutoff(eps_ij)

    vdW = np.where(eps_ij != 0, lennard_jones(dij, sig_ij, eps_ij), 0)

    # Electrostatics by direct-space part of PME
    qij = apply_cutoff(charges[inds_l] * charges[inds_r])
    electrostatics = direct_space_pme(dij, qij, beta)

    return vdW, electrostatics


def nonbonded_v3_interaction_groups(conf, params, box, a_idxs, b_idxs, beta: float, cutoff: Optional[float] = None):
    """Nonbonded interactions between all pairs of atoms $(i, j)$
    where $i$ is in the first set and $j$ in the second.

    See nonbonded_v3 docstring for more details
    """
    pairs = pairs_from_interaction_groups(a_idxs, b_idxs)
    vdW, electrostatics = nonbonded_v3_on_specific_pairs(conf, params, box, pairs, beta, cutoff)
    return vdW, electrostatics


def interpolated(u_fn):
    @functools.wraps(u_fn)
    def wrapper(conf, params, box, lamb):

        # params is expected to be the concatenation of initial
        # (lambda = 0) and final (lambda = 1) parameters, each of
        # length num_atoms
        assert params.size % 2 == 0
        num_atoms = params.shape[0] // 2

        new_params = (1 - lamb) * params[:num_atoms] + lamb * params[num_atoms:]
        return u_fn(conf, new_params, box, lamb)

    return wrapper


def validate_coulomb_cutoff(cutoff=1.0, beta=2.0, threshold=1e-2):
    """check whether f(r) = erfc(beta * r) <= threshold at r = cutoff
    following https://github.com/proteneer/timemachine/pull/424#discussion_r629678467"""
    if erfc(beta * cutoff) > threshold:
        print(UserWarning(f"erfc(beta * cutoff) = {erfc(beta * cutoff)} > threshold = {threshold}"))


# utilities for efficiently recomputing energy as a function of ligand charges
#   (where x_ligand, x_environment, q_environment are all constant, but q_ligand is variable)
#   exploiting the fact that nonbonded_interaction_group(ligand_charges) is a linear function of ligand_charges
#   TODO: avoid repetition between this and lennard-jones


def coulomb_prefactor_on_atom(x_i, x_others, q_others, box=None, beta=2.0, cutoff=np.inf):
    """
    to compute
    contribution_i = sum_j (q_i * q_j) / d_ij * erfc(beta * d_ij)
    efficiently as a function of q_i

    precompute
    prefactor_i = sum_j q_j / d_ij * erfc(beta * d_ij)

    so that
    contribution_i(q_i) = prefactor_i * q_i
    """
    displacements_ij = jax_utils.delta_r(x_i, x_others, box)
    d2_ij = np.sum(displacements_ij ** 2, 1)
    d_ij = np.where(d2_ij <= cutoff ** 2, np.sqrt(d2_ij), np.inf)
    prefactor_i = np.sum((q_others / d_ij) * erfc(beta * d_ij))
    return prefactor_i


def coulomb_prefactors_on_snapshot(x_ligand, x_env, q_env, box=None, beta=2.0, cutoff=np.inf):
    """map coulomb_prefactor_on_atom over each atom in x_ligand"""

    def f_atom(x_i):
        return coulomb_prefactor_on_atom(x_i, x_env, q_env, box, beta, cutoff)

    return vmap(f_atom)(x_ligand)


def coulomb_prefactors_on_traj(traj, boxes, charges, ligand_indices, env_indices, beta=2.0, cutoff=np.inf):
    """map coulomb_prefactors_on_snapshot over each snapshot in a trajectory"""

    q_env = charges[env_indices]

    def f_snapshot(coords, box):
        x_ligand = coords[ligand_indices]
        x_env = coords[env_indices]
        return coulomb_prefactors_on_snapshot(x_ligand, x_env, q_env, box, beta, cutoff)

    return vmap(f_snapshot)(traj, boxes)


def coulomb_interaction_group_energy(q_ligand, q_prefactors):
    """assuming q_prefactors = coulomb_prefactors_on_snapshot(x_ligand, ...),
    cheaply compute the energy of ligand-environment interaction group"""

    return np.dot(q_prefactors, q_ligand)


# utilities for efficiently recomputing energy as a function of ligand epsilon parameters
#   (where x_ligand, x_environment, eps_environment are all constant, but eps_ligand is variable)
#   exploiting the fact that nonbonded_interaction_group(eps_ligand) is a linear function of sqrt(eps_ligand)
#   TODO: avoid repetition between this and coulomb


def lj_prefactor_on_atom(x_i, x_others, sig_i, sig_others, eps_others, box=None, cutoff=np.inf):
    """
    to compute
    contribution_i = sum_j LennardJones(x_i, x_j; (sig_i, eps_i), (sig_j, eps_j))
    efficiently as a function of eps_i

    precompute
    prefactor_i = sum_j 4 * sqrt(eps_j) * ((sig_ij/r_ij)**12 - (sig_ij/r_ij)**6)

    so that
    contribution_i(eps_i) = prefactor_i * sqrt(eps_i)
    """

    displacements_ij = jax_utils.delta_r(x_i, x_others, box)
    d2_ij = np.sum(displacements_ij ** 2, 1)
    d_ij = np.where(d2_ij <= cutoff ** 2, np.sqrt(d2_ij), np.inf)
    sig_ij = combining_rule_sigma(sig_i, sig_others)
    sig6 = (sig_ij / d_ij) ** 6
    sig12 = sig6 ** 2
    # note: eps_others rather than sqrt(eps_others) -- see `combining_rule_epsilon`
    prefactor_i = np.sum(4 * eps_others * (sig12 - sig6))
    return prefactor_i


def lj_prefactors_on_snapshot(x_ligand, x_env, sig_ligand, sig_others, eps_others, box=None, cutoff=np.inf):
    """map lj_prefactor_on_atom over each atom in x_ligand"""

    def f_atom(x_i, sig_i):
        return lj_prefactor_on_atom(x_i, x_env, sig_i, sig_others, eps_others, box, cutoff)

    return vmap(f_atom)(x_ligand, sig_ligand)


def lj_prefactors_on_traj(traj, boxes, sigmas, epsilons, ligand_indices, env_indices, cutoff=np.inf):
    """map lj_prefactors_on_snapshot over each snapshot in a trajectory"""

    sig_ligand = sigmas[ligand_indices]

    eps_env = epsilons[env_indices]
    sig_env = sigmas[env_indices]

    def f_snapshot(coords, box):
        x_ligand = coords[ligand_indices]
        x_env = coords[env_indices]
        return lj_prefactors_on_snapshot(x_ligand, x_env, sig_ligand, sig_env, eps_env, box, cutoff)

    return vmap(f_snapshot)(traj, boxes)


def lj_interaction_group_energy(eps_ligand, eps_prefactors):
    """assuming eps_prefactors = lj_prefactors_on_snapshot(x_ligand, ...),
    cheaply compute the energy of ligand-environment interaction group"""

    return np.dot(eps_prefactors, eps_ligand)
