from typing import Optional, Tuple, cast

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap
from jax.scipy.special import erfc
from jax.typing import ArrayLike
from numpy.typing import NDArray
from scipy.special import binom

from timemachine.potentials import jax_utils
from timemachine.potentials.jax_utils import (
    DEFAULT_CHUNK_SIZE,
    delta_r,
    distance_on_pairs,
    pairs_from_interaction_groups,
    pairwise_distances,
    process_traj_in_chunks,
)


def switch_fn(dij, cutoff=1.2):
    """heuristic switching function

    intended to:
    * have {f, f', f''} go to 0 at cutoff
    * keep "switch_fn(dij) * erfc(beta * dij)" as close as possible to "erfc(beta * dij)"
        for the range dij in [0, 1.2), for beta = 2.0

    usage notes:
    * not necessarily intended for use with LJ

    TODO
    * respond to user-specified cutoff
    """
    cutoff = 1.2  # NOTE: intentionally overrides user input for now
    f = jnp.power(jnp.cos((jnp.pi * jnp.power(dij / cutoff, 8)) / 2), 3)
    return jnp.where(dij < cutoff, f, 0)


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


def switched_direct_space_pme(dij, qij, beta, cutoff):
    """direct_space_pme * switch_fn"""
    return direct_space_pme(dij, qij, beta) * switch_fn(dij, cutoff)


def nonbonded_block_unsummed(
    xi: NDArray,
    xj: NDArray,
    box: NDArray,
    params_i: NDArray,
    params_j: NDArray,
    beta: float,
    cutoff: float,
) -> Array:
    """
    This is a modified version of `nonbonded` that computes a block of
    NxM interactions between two sets of particles x_i and x_j. It is assumed that
    there are no exclusions between the two particle sets. Typical use cases
    include computing the interaction energy between the environment and a
    ligand.

    This is mainly used for testing.

    Parameters
    ----------
    xi : (N,3) np.ndarray
        Coordinates
    xj : (M,3) np.ndarray
        Coordinates
    box : Optional 3x3 np.ndarray
        Periodic boundary conditions
    params_i : (N,4) np.ndarray
        3-Tuples of (charge, sigma, epsilons, w_offset)
    params_j : (M,4) np.ndarray
        3-Tuples of (charge, sigma, epsilons, w_offset)
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

    w_i = jnp.expand_dims(params_i[:, 3], axis=1)
    w_j = jnp.expand_dims(params_j[:, 3], axis=0)

    dij = delta_r(ri, rj, box)
    dij = jnp.concatenate([dij, (w_i - w_j).reshape(*dij.shape[:-1], 1)], axis=-1)

    dij = jnp.linalg.norm(dij, axis=-1)
    sig_i = jnp.expand_dims(params_i[:, 1], axis=1)
    sig_j = jnp.expand_dims(params_j[:, 1], axis=0)
    eps_i = jnp.expand_dims(params_i[:, 2], axis=1)
    eps_j = jnp.expand_dims(params_j[:, 2], axis=0)

    sig_ij = combining_rule_sigma(sig_i, sig_j)
    eps_ij = combining_rule_epsilon(eps_i, eps_j)

    qi = jnp.expand_dims(params_i[:, 0], axis=1)
    qj = jnp.expand_dims(params_j[:, 0], axis=0)

    qij = jnp.multiply(qi, qj)

    es = switched_direct_space_pme(dij, qij, beta, cutoff)
    lj = lennard_jones(dij, sig_ij, eps_ij)

    nrgs = jnp.where(dij < cutoff, es + lj, 0)
    return cast(Array, nrgs)


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
    eij_charge = jnp.where(keep_mask, switched_direct_space_pme(dij, qij, beta, cutoff), 0)  # zero out diagonals
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
) -> Tuple[Array, Array]:
    """See `nonbonded` docstring for more details

    Notes
    -----
    * Warning! This function performs no validation of pair indices. If the provided pairs are incomplete (e.g. omitting
        some pairs of atoms that could be within cutoff, or omitting intramolecular pairs, ...), then incorrect results
        can be returned.
    """

    if len(pairs) == 0:
        return jnp.zeros(1), jnp.zeros(1)

    inds_l, inds_r = pairs.T

    charges, sig, eps, w_coords = params.T

    # distances and cutoff
    w_offsets = w_coords[pairs[:, 0]] - w_coords[pairs[:, 1]] if w_coords is not None else None
    dij = distance_on_pairs(conf[inds_l], conf[inds_r], box, w_offsets)
    if cutoff is None:
        cutoff = np.inf
    keep_mask = dij < cutoff

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
    electrostatics = switched_direct_space_pme(dij, qij, beta, cutoff)

    if rescale_mask is not None:
        assert rescale_mask.shape == (len(pairs), 2)
        rescale_vdW = rescale_mask[:, 1]
        vdW = jnp.where(rescale_vdW != 0, vdW * rescale_vdW, 0)
        rescale_electrostatics = rescale_mask[:, 0]
        electrostatics = jnp.where(rescale_electrostatics != 0, electrostatics * rescale_electrostatics, 0)

    vdW_arr = cast(Array, vdW)
    electrostatics_arr = cast(Array, electrostatics)

    return vdW_arr, electrostatics_arr


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

    keep_mask = dij < cutoff

    def apply_cutoff(x):
        return jnp.where(keep_mask, x, 0)

    q_ij = apply_cutoff(q_ij)
    sig_ij = apply_cutoff(sig_ij)
    eps_ij = apply_cutoff(eps_ij)

    vdW = jnp.where(eps_ij != 0, lennard_jones(dij, sig_ij, eps_ij), 0)
    electrostatics = jnp.where(q_ij != 0, switched_direct_space_pme(dij, q_ij, beta, cutoff), 0)

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


# utilities for efficiently recomputing energy as a function of ligand charges
#   (where x_ligand, x_environment, q_environment are all constant, but q_ligand is variable)
#   exploiting the fact that nonbonded_interaction_group(ligand_charges) is a linear function of ligand_charges
#   TODO: avoid repetition between this and lennard-jones


def coulomb_prefactor_on_atom(x_i, x_others, q_others, box=None, beta=2.0, cutoff=jnp.inf) -> Array:
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
    prefactor_i = jnp.sum((q_others / d_ij) * erfc(beta * d_ij) * switch_fn(d_ij, cutoff))
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


def coulomb_prefactors_on_traj(
    traj, boxes, charges, ligand_indices, env_indices, beta=2.0, cutoff=jnp.inf, chunk_size=DEFAULT_CHUNK_SIZE
):
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
    chunk_size: int
        process traj in ceil(T / chunk_size) chunks, to limit memory consumption

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
        return jit(coulomb_prefactors_on_snapshot)(x_ligand, x_env, q_env, box, beta, cutoff)

    return process_traj_in_chunks(f_snapshot, traj, boxes, chunk_size)


def coulomb_interaction_group_energy(q_ligand: ArrayLike, q_prefactors: ArrayLike) -> Array:
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


# utilities for efficiently recomputing energy as a function of ligand LJ parameters

# note: two cases:
# (1) only varying LJ eps parameters (can summarize using 1 float per ligand atom)
# (2) simultaneously varying LJ eps and sig parameters (summarized using 20 floats per ligand atom)


# case (1) : only varying ligand LJ eps parameters
def lj_eps_prefactor_on_atom(x_i, x_others, sig_i, sig_others, eps_others, box=None, cutoff=np.inf) -> float:
    """Precompute part of (sum_j LennardJones(x_i, x_j; (sig_i, eps_i), (sig_j, eps_j))) that does not depend on eps_i

    Parameters
    ----------
    x_i : [D] array
        position of focus atom (in ligand)
    x_others: [N_env, D] array
        positions of all other atoms (in environment)
    sig_i: float
        LJ_sig of focus atom
    sig_others: [N_env] array
        LJ_sig of all other atoms (in environment)
    eps_others: [N_env] array
        LJ_eps of all other atoms (in environment)
    box: optional diagonal [D, D] array
    cutoff: float

    Returns
    -------
    prefactor_i : float
        sum_j 4 * eps_j * ((sig_ij/r_ij)**12 - (sig_ij/r_ij)**6)
    """
    d_ij = jax_utils.distance_from_one_to_others(x_i, x_others, box, cutoff)

    sig_ij = combining_rule_sigma(sig_i, sig_others)
    sig6 = (sig_ij / d_ij) ** 6
    sig12 = sig6**2
    # note: eps_others rather than sqrt(eps_others) -- see `combining_rule_epsilon`
    prefactor_i = np.sum(4 * eps_others * (sig12 - sig6))
    return prefactor_i


def lj_eps_prefactors_on_snapshot(x_ligand, x_env, sig_ligand, sig_env, eps_env, box=None, cutoff=np.inf):
    """apply lj_eps_prefactor_on_atom to all atoms in x_ligand"""

    def f_atom(x_i, sig_i):
        return lj_eps_prefactor_on_atom(x_i, x_env, sig_i, sig_env, eps_env, box, cutoff)

    lj_eps_prefactors = vmap(f_atom, (0, 0))(x_ligand, sig_ligand)

    return lj_eps_prefactors


def lj_eps_prefactors_on_traj(
    traj, boxes, sigmas, epsilons, ligand_indices, env_indices, cutoff=np.inf, chunk_size=DEFAULT_CHUNK_SIZE
):
    """apply lj_eps_prefactors_on_snapshot to all snapshots in traj"""
    validate_interaction_group_idxs(len(traj[0]), ligand_indices, env_indices)

    eps_env = epsilons[env_indices]
    sig_env = sigmas[env_indices]

    sig_ligand = sigmas[ligand_indices]

    def f_snapshot(coords, box):
        x_ligand = coords[ligand_indices]
        x_env = coords[env_indices]
        return jit(lj_eps_prefactors_on_snapshot)(x_ligand, x_env, sig_ligand, sig_env, eps_env, box, cutoff)

    return process_traj_in_chunks(f_snapshot, traj, boxes, chunk_size)


def lj_eps_interaction_group_energy(eps_ligand, lj_eps_prefactors):
    """Assuming lj_eps_prefactors = lj_eps_prefactors_on_snapshot(x_ligand, ...),
    ligand-environment interaction group energy as fxn of eps_ligand is a dot product.

    Parameters
    ----------
    eps_ligand: [N_lig] arrays
    lj_eps_prefactors: [N_lig] array

    Returns
    -------
    energy: float
    """
    return jnp.sum(eps_ligand * lj_eps_prefactors)


# case (2) -- (sig, eps) vary simultaneously
#   (where (x_ligand, x_env, sig_env, eps_env) are all constant, but (sig_ligand, eps_ligand) are variable)
#   using [Naden, Shirts]'s linear basis-function approach
#   TODO: avoid repetition between this and coulomb


def _basis_expand_lj_term(sig_env, eps_env, r_env, power):
    """
    Compute expansion of
        sum_i (4 * eps * eps_env[i] * (sig_env[i] / r_env[i])^power)

    (called by basis_expand_lj_env with power=12 and power=6)

    Parameters
    ----------
    sig_env, eps_env : [N_env] arrays
        sigma, epsilon parameters of environment atoms
    r_env : [N_env] array
        distances from trial atom to all environment atoms
    power : int
        12 or 6

    Returns
    -------
    h_n : [power + 1] array

    References
    ----------
    eq. C.1 of Levi Naden's thesis
    """
    r_inv_pow = r_env**-power

    exponents = power - np.arange(power + 1)
    coeffs = binom(power, exponents)

    raised = sig_env ** jnp.expand_dims(exponents, 1)
    h_n_i = r_inv_pow * raised * jnp.expand_dims(coeffs, 1) * jnp.expand_dims(eps_env, 0)

    h_n = jnp.sum(4 * h_n_i, 1)
    return h_n


def basis_expand_lj_env(sig_env, eps_env, r_env):
    """Precomputed part of basis expansion that allows fast computation of

    f(sig, eps)
        = sum_i LJ(r_env[i]; sig + sig_env[i], eps * eps_env[i])
        = dot(
            basis_expand_lj_atom(sig, eps),
            basis_expand_lj_env(sig_env, eps_env, r_env)
        )
    Parameters
    ----------
    sig_env, eps_env : [N_env] arrays
        sigma, epsilon parameters of environment atoms
    r_env : [N_env] array
        distances from variable atom to all environment atoms

    Returns
    -------
    h_n : [20] array
        can be dotted with output of basis_expand_lj_atom(sig, eps)
        to compute energy of one atom interacting with all environment atoms
    """
    jit_basis_expand_lj_term = jit(_basis_expand_lj_term, static_argnums=3)
    h_n_12 = jit_basis_expand_lj_term(sig_env, eps_env, r_env, 12)
    h_n_6 = -jit_basis_expand_lj_term(sig_env, eps_env, r_env, 6)
    return jnp.hstack([h_n_12, h_n_6])


def basis_expand_lj_atom(sig: float, eps: float) -> Array:
    """Variable part of basis expansion that allows fast computation of

    f(sig, eps)
        = sum_i LJ(r_env[i]; sig + sig_env[i], eps * eps_env[i])
        = dot(
            basis_expand_lj_atom(sig, eps),
            basis_expand_lj_env(sig_env, eps_env, r_env)
        )

    Parameters
    ----------
    sig, eps : floats
        LJ parameters of variable atom

    Returns
    -------
    projection : [20] array
        can be dotted with output of basis_expand_lj_env(sig_env, eps_env, r_env)
        to compute energy of one atom interacting with all environment atoms
    """
    exponents = jnp.hstack([jnp.arange(12 + 1), jnp.arange(6 + 1)])
    return eps * (sig**exponents)


def lj_prefactors_on_atom(x, x_others, sig_others, eps_others, box=None, cutoff=jnp.inf):
    """Precompute part of

        sum_j LennardJones(x_i, x_j; (sig_i, eps_i), (sig_j, eps_j))

    that does not depend on (sig_i, eps_i)

    Parameters
    ----------
    x : [D] array
        position of focus atom (in ligand)
    x_others: [N_env, D] array
        positions of all other atoms (in environment)
    sig_others, eps_others: [N_env] arrays
        Lennard-Jones parameters of all other atoms (in environment)
    box: optional diagonal [D, D] array
    cutoff: float

    Returns
    -------
    prefactors : [20] array

    See Also
    --------
    basis_expand_lj_atom : computes a basis expansion of an atom's (sig, eps) that can be dotted with these prefactors
    """
    r_env = jax_utils.distance_from_one_to_others(x, x_others, box, cutoff)
    prefactors = basis_expand_lj_env(sig_others, eps_others, r_env)
    return prefactors


def lj_prefactors_on_snapshot(x_ligand, x_env, sig_env, eps_env, box=None, cutoff=jnp.inf):
    """Map lj_prefactor_on_atom over atoms in x_ligand

    Parameters
    ----------
    x_ligand: [N_lig, D] array
    x_env: [N_env, D] array
    sig_env: [N_env] array
    eps_env: [N_env] array
    box: optional diagonal [D, D] array
    cutoff: float

    Returns
    -------
    prefactors: [N_lig, 20] array
        prefactors[i] = lj_prefactor_on_atom(x_ligand[i], ...)
    """

    def f_atom(x_i):
        return lj_prefactors_on_atom(x_i, x_env, sig_env, eps_env, box, cutoff)

    return vmap(f_atom)(x_ligand)


def lj_prefactors_on_traj(
    traj, boxes, sigmas, epsilons, ligand_indices, env_indices, cutoff=jnp.inf, chunk_size=DEFAULT_CHUNK_SIZE
):
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
    chunk_size: int
        process traj in ceil(T / chunk_size) chunks, to limit memory consumption

    Returns
    -------
    traj_prefactors: [T, N_lig, 20] array
        traj_prefactors[t] = lj_prefactors_on_snapshot(traj[t][ligand_indices], ...)
    """
    validate_interaction_group_idxs(len(traj[0]), ligand_indices, env_indices)

    eps_env = epsilons[env_indices]
    sig_env = sigmas[env_indices]

    def f_snapshot(coords, box):
        x_ligand = coords[ligand_indices]
        x_env = coords[env_indices]
        return jit(lj_prefactors_on_snapshot)(x_ligand, x_env, sig_env, eps_env, box, cutoff)

    return process_traj_in_chunks(f_snapshot, traj, boxes, chunk_size)


def lj_interaction_group_energy(sig_ligand, eps_ligand, lj_prefactors):
    """Assuming lj_prefactors = lj_prefactors_on_snapshot(x_ligand, ...),
    cheaply compute the energy of ligand-environment interaction group

    Parameters
    ----------
    sig_ligand, eps_ligand: [N_lig] arrays
    lj_prefactors: [N_lig, 20] array

    Returns
    -------
    energy: float
    """

    projection = vmap(basis_expand_lj_atom)(sig_ligand, eps_ligand)
    return jnp.sum(projection * lj_prefactors)
