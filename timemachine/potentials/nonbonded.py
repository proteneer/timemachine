import functools

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.scipy.special import erfc
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from timemachine.potentials import jax_utils
from timemachine.potentials.jax_utils import (
    convert_to_4d,
    delta_r,
    distance,
    distance_on_pairs,
    pairs_from_interaction_groups,
)

Array: TypeAlias = NDArray


def switch_fn(dij, cutoff):
    return jnp.power(jnp.cos((jnp.pi * jnp.power(dij, 8)) / (2 * cutoff)), 2)


from typing import Optional


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
    ri = jnp.expand_dims(xi, axis=1)
    rj = jnp.expand_dims(xj, axis=0)
    d2ij = jnp.sum(jnp.power(delta_r(ri, rj, box), 2), axis=-1)
    dij = jnp.sqrt(d2ij)
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

    nrg = jnp.where(dij > cutoff, 0, es + lj)
    return jnp.sum(nrg)


def convert_exclusions_to_rescale_masks(exclusion_idxs, scales, N):
    """Converts exclusions from list format used in Nonbonded to mask format used in nonbonded_v3"""

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
            box_4d = jnp.eye(4) * 1000
            box_4d = box_4d.at[:3, :3].set(box)
        else:
            box_4d = box
    else:
        box_4d = None

    box = box_4d

    charges = params[:, 0]
    sig = params[:, 1]
    eps = params[:, 2]

    sig_i = jnp.expand_dims(sig, 0)
    sig_j = jnp.expand_dims(sig, 1)
    sig_ij = combining_rule_sigma(sig_i, sig_j)

    eps_i = jnp.expand_dims(eps, 0)
    eps_j = jnp.expand_dims(eps, 1)

    eps_ij = combining_rule_epsilon(eps_i, eps_j)

    dij = distance(conf, box)

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
        eij_charge = jnp.where(dij > cutoff, 0, eij_charge)

    eij_total = eij_lj * lj_rescale_mask + eij_charge * charge_rescale_mask

    return jnp.sum(eij_total / 2)


def nonbonded_v3_on_specific_pairs(
    conf, params, box, pairs, beta: float, cutoff: Optional[float] = None, rescale_mask=None
):
    """See nonbonded_v3 docstring for more details

    Notes
    -----
    * Warning! This function performs no validation of pair indices. If the provided pairs are incomplete (e.g. omitting
        some pairs of atoms that could be within cutoff, or omitting intramolecular pairs, ...), then incorrect results
        can be returned.
    """

    if len(pairs) == 0:
        return np.zeros(1), np.zeros(1)

    inds_l, inds_r = pairs.T

    # distances and cutoff
    dij = distance_on_pairs(conf[inds_l], conf[inds_r], box)
    if cutoff is None:
        cutoff = np.inf
    keep_mask = dij <= cutoff

    def apply_cutoff(x):
        return jnp.where(keep_mask, x, 0)

    charges, sig, eps = params.T

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


def nonbonded_v3_on_precomputed_pairs(conf, params, box, pairs, beta: float, cutoff: Optional[float] = None):
    """
    Similar to pairlist, except that we pre-compute the scaled charges, epsilsons, and broadcast

    conf: N,3
    params: P,3 (q_ij, s_ij, e_ij)
    pairs: P,2 (i,j)
    """

    if len(pairs) == 0:
        return np.zeros(1), np.zeros(1)

    inds_l, inds_r = pairs.T

    # distances and cutoff
    dij = distance_on_pairs(conf[inds_l], conf[inds_r], box)
    if cutoff is None:
        cutoff = np.inf
    keep_mask = dij <= cutoff

    def apply_cutoff(x):
        return jnp.where(keep_mask, x, 0)

    q_ij = apply_cutoff(params[:, 0])
    sig_ij = apply_cutoff(params[:, 1])
    eps_ij = apply_cutoff(params[:, 2])

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


def nonbonded_v3_interaction_groups(conf, params, box, a_idxs, b_idxs, beta: float, cutoff: Optional[float] = None):
    """Nonbonded interactions between all pairs of atoms $(i, j)$
    where $i$ is in the first set and $j$ in the second.

    See nonbonded_v3 docstring for more details
    """
    validate_interaction_group_idxs(len(conf), a_idxs, b_idxs)
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


def coulomb_prefactor_on_atom(x_i, x_others, q_others, box=None, beta=2.0, cutoff=jnp.inf) -> float:
    """Precompute part of (sum_i q_i * q_j / d_ij * rxn_field(d_ij)) that does not depend on q_i

    Parameters
    ----------
    x_i : [D] array
        position of focus atom (in ligand)
    x_others: [N_env, D] array
        positions of all other atoms (in environment)
    q_others: [N_env] array
        charges of all other atoms (in environment)
    box: optional diagonal [D, D] array
    beta: float
    cutoff: float

    Returns
    -------
    prefactor_i : float
        sum_j q_j / d_ij * erfc(beta * d_ij)
    """
    d_ij = jax_utils.distance_from_one_to_others(x_i, x_others, box, cutoff)
    prefactor_i = jnp.sum((q_others / d_ij) * erfc(beta * d_ij))
    return prefactor_i


def coulomb_prefactors_on_snapshot(x_ligand, x_env, q_env, box=None, beta=2.0, cutoff=np.inf) -> Array:
    """Map coulomb_prefactor_on_atom over atoms in x_ligand

    Parameters
    ----------
    x_ligand: [N_lig, D] array
    x_env: [N_env, D] array
    q_env: [N_env] array
    box: optional diagonal [D, D] array
    beta: float
    cutoff: float

    Returns
    -------
    prefactors: [N_lig] array
        prefactors[i] = coulomb_prefactor_on_atom(x_ligand[i], ...)
    """

    def f_atom(x_i):
        return coulomb_prefactor_on_atom(x_i, x_env, q_env, box, beta, cutoff)

    return vmap(f_atom)(x_ligand)


def coulomb_prefactors_on_traj(traj, boxes, charges, ligand_indices, env_indices, beta=2.0, cutoff=jnp.inf):
    """Map coulomb_prefactors_on_snapshot over snapshots in a trajectory

    Parameters
    ----------
    traj: [T, N, D] array
    boxes: diagonal [T, D, D] array (or [T] array of Nones)
    charges: [N] array
    ligand_indices: [N_lig] array of ints
    env_indices: [N_env] array of ints
    beta: float
    cutoff: float

    Returns
    -------
    traj_prefactors: [T, N_lig] array
        traj_prefactors[t] = coulomb_prefactors_on_snapshot(traj[t][ligand_indices], ...)
    """
    validate_interaction_group_idxs(len(traj[0]), ligand_indices, env_indices)

    q_env = charges[env_indices]

    def f_snapshot(coords, box):
        x_ligand = coords[ligand_indices]
        x_env = coords[env_indices]
        return coulomb_prefactors_on_snapshot(x_ligand, x_env, q_env, box, beta, cutoff)

    return vmap(f_snapshot)(traj, boxes)


def coulomb_interaction_group_energy(q_ligand: Array, q_prefactors: Array) -> float:
    """Assuming q_prefactors = coulomb_prefactors_on_snapshot(x_ligand, ...),
    cheaply compute the energy of ligand-environment interaction group

    Parameters
    ----------
    q_ligand: [N_lig] array
    q_prefactors: [N_lig] array

    Returns
    -------
    energy: float
    """

    return jnp.dot(q_prefactors, q_ligand)


# utilities for efficiently recomputing energy as a function of ligand epsilon parameters
#   (where x_ligand, x_environment, eps_environment are all constant, but eps_ligand is variable)
#   exploiting the fact that nonbonded_interaction_group(eps_ligand) is a linear function of sqrt(eps_ligand)
#   TODO: avoid repetition between this and coulomb


def lj_prefactor_on_atom(x_i, x_others, sig_i, sig_others, eps_others, box=None, cutoff=jnp.inf):
    """Precompute part of sum_j LennardJones(x_i, x_j; (sig_i, eps_i), (sig_j, eps_j)) that does not depend on eps_i

    Parameters
    ----------
    x_i : [D] array
        position of focus atom (in ligand)
    x_others: [N_env, D] array
        positions of all other atoms (in environment)
    sig_i: float
        Lennard-Jones sigma parameter of focus atom
    sig_others: [N_env] array
        Lennard-Jones sigma parameters of all other atoms (in environment)
    box: optional diagonal [D, D] array
    beta: float
    cutoff: float

    Returns
    -------
    prefactor_i : float
        sum_j 4 * sqrt(eps_j) * ((sig_ij/r_ij)**12 - (sig_ij/r_ij)**6)
    """
    d_ij = jax_utils.distance_from_one_to_others(x_i, x_others, box, cutoff)

    sig_ij = combining_rule_sigma(sig_i, sig_others)
    sig6 = (sig_ij / d_ij) ** 6
    sig12 = sig6 ** 2
    # note: eps_others rather than sqrt(eps_others) -- see `combining_rule_epsilon`
    prefactor_i = jnp.sum(4 * eps_others * (sig12 - sig6))
    return prefactor_i


def lj_prefactors_on_snapshot(x_ligand, x_env, sig_ligand, sig_env, eps_env, box=None, cutoff=jnp.inf):
    """Map lj_prefactor_on_atom over atoms in x_ligand

    Parameters
    ----------
    x_ligand: [N_lig, D] array
    x_env: [N_env, D] array
    sig_ligand: [N_lig] array
    sig_env: [N_env] array
    eps_env: [N_env] array
    box: optional diagonal [D, D] array
    cutoff: float

    Returns
    -------
    prefactors: [N_lig] array
        prefactors[i] = lj_prefactor_on_atom(x_ligand[i], ...)
    """

    def f_atom(x_i, sig_i):
        return lj_prefactor_on_atom(x_i, x_env, sig_i, sig_env, eps_env, box, cutoff)

    return vmap(f_atom)(x_ligand, sig_ligand)


def lj_prefactors_on_traj(traj, boxes, sigmas, epsilons, ligand_indices, env_indices, cutoff=jnp.inf):
    """Map lj_prefactors_on_snapshot over snapshots in a trajectory

    Parameters
    ----------
    traj: [T, N, D] array
    boxes: diagonal [T, D, D] array (or [T] array of Nones)
    sigmas: [N] array
    epsilons: [N] array
    ligand_indices: [N_lig] array of ints
    env_indices: [N_env] array of ints
    cutoff: float

    Returns
    -------
    traj_prefactors: [T, N_lig] array
        traj_prefactors[t] = lj_prefactors_on_snapshot(traj[t][ligand_indices], ...)
    """
    validate_interaction_group_idxs(len(traj[0]), ligand_indices, env_indices)

    sig_ligand = sigmas[ligand_indices]

    eps_env = epsilons[env_indices]
    sig_env = sigmas[env_indices]

    def f_snapshot(coords, box):
        x_ligand = coords[ligand_indices]
        x_env = coords[env_indices]
        return lj_prefactors_on_snapshot(x_ligand, x_env, sig_ligand, sig_env, eps_env, box, cutoff)

    return vmap(f_snapshot)(traj, boxes)


def lj_interaction_group_energy(eps_ligand, eps_prefactors):
    """Assuming eps_prefactors = lj_prefactors_on_snapshot(x_ligand, ...),
    cheaply compute the energy of ligand-environment interaction group

    Parameters
    ----------
    eps_ligand: [N_lig] array
    eps_prefactors: [N_lig] array

    Returns
    -------
    energy: float
    """

    return jnp.dot(eps_prefactors, eps_ligand)
