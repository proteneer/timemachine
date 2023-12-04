"""Utilities for efficiently computing interaction-group energies on stored trajectories,
where only a subset of parameters change.

Easiest special cases:
* Varying ligand parameters with multiplicative combining rules
    u(x_i, x_j, theta_i, theta_j) = theta_i * theta_j * f(distance(x_i, x_j))
    * Varying ligand *charges* (but holding environment parameters fixed) for (damped) Coulomb interactions
    * Varying ligand *LJ epsilons* (holding ligand LJ sigmas fixed, and all environment LJ parameters fixed)

Slightly trickier special cases:
* Varying ligand LJ (eps, sig) simultaneously
    * Requires a larger summary, see https://github.com/proteneer/timemachine/pull/931 for refs and details

TODO:
[ ] Reduce code repetition between LJ and charge
[ ] Develop optimizations for case where both ligand and protein params can vary simultaneously
"""

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.scipy.special import erfc
from numpy.typing import NDArray as Array
from scipy.special import binom

from timemachine.potentials import jax_utils
from timemachine.potentials.jax_utils import DEFAULT_CHUNK_SIZE, process_traj_in_chunks
from timemachine.potentials.nonbonded import combining_rule_sigma, validate_interaction_group_idxs


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


# utilities for efficiently recomputing energy as a function of ligand LJ parameters

# note: two cases:
# (1) only varying LJ eps parameters (can summarize using 1 float per ligand atom)
# (2) simultaneously varying LJ eps and sig parameters (summarized using 20 floats per ligand atom)


def lj_eps_prefactor_on_atom(x_i, x_others, sig_i, sig_others, eps_others, box=None, cutoff=jnp.inf) -> float:
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
        sum_j 4 * sqrt(eps_j) * ((sig_ij/r_ij)**12 - (sig_ij/r_ij)**6)
    """
    d_ij = jax_utils.distance_from_one_to_others(x_i, x_others, box, cutoff)

    sig_ij = combining_rule_sigma(sig_i, sig_others)
    sig6 = (sig_ij / d_ij) ** 6
    sig12 = sig6**2
    # note: eps_others rather than sqrt(eps_others) -- see `combining_rule_epsilon`
    prefactor_i = np.sum(4 * eps_others * (sig12 - sig6))
    return prefactor_i


def lj_eps_prefactors_on_snapshot(x_ligand, x_env, sig_ligand, sig_env, eps_env, box=None, cutoff=jnp.inf):
    def f_atom(x_i, sig_i):
        return lj_eps_prefactor_on_atom(x_i, x_env, sig_i, sig_env, eps_env, box, cutoff)

    return vmap(f_atom, (0, 0))(x_ligand, sig_ligand)


def lj_eps_prefactors_on_traj(
    traj, boxes, sigmas, epsilons, ligand_indices, env_indices, cutoff=jnp.inf, chunk_size=DEFAULT_CHUNK_SIZE
):
    validate_interaction_group_idxs(len(traj[0]), ligand_indices, env_indices)

    eps_env = epsilons[env_indices]
    sig_env = sigmas[env_indices]

    sig_ligand = sigmas[ligand_indices]

    def f_snapshot(coords, box):
        x_ligand = coords[ligand_indices]
        x_env = coords[env_indices]
        return jit(lj_eps_prefactors_on_snapshot)(x_ligand, x_env, sig_ligand, sig_env, eps_env, box, cutoff)

    return process_traj_in_chunks(f_snapshot, traj, boxes, chunk_size)


#   (where (x_ligand, x_env, sig_env, eps_env) are all constant, but (sig_ligand, eps_ligand) are variable)
#   using [Naden, Shirts]'s linear basis-function approach


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


class ReweightableTrajectory:
    def __init__(
        self,
        traj,
        boxes,
        nb_params,
        ligand_indices,
        env_indices,
        beta=2.0,
        cutoff=1.2,
        support_ligand_q=True,
        support_ligand_lj_eps=True,
        support_ligand_lj_sig=False,
    ):
        """"""
        n_frames, n_atoms, dim = traj.shape
        self.n_frames = n_frames

        assert dim == 3

        charges, eps, sig = nb_params.T
        assert len(charges) == n_atoms

        self.ligand_q0 = charges[ligand_indices]
        if support_ligand_q:
            self.supports_ligand_q = True
            self.q_prefactors = coulomb_prefactors_on_traj(
                traj, boxes, charges, ligand_indices, env_indices, beta=beta, cutoff=cutoff
            )

        self.ligand_lj_eps = eps[ligand_indices]
        if support_ligand_lj_eps:
            self.supports_ligand_lj_eps = True
            self.lj_eps_prefactors = lj_eps_prefactors_on_traj(
                traj, boxes, sig, eps, ligand_indices, env_indices, cutoff
            )

        self.ligand_lj_sig = sig[ligand_indices]
        if support_ligand_lj_sig:
            self.supports_ligand_lj_sig = True
            self.lj_sig_prefactors = lj_prefactors_on_traj(traj, boxes, sig, eps, ligand_indices, env_indices, cutoff)

    def compute_potential_energy(self, ligand_q, ligand_lj_eps, ligand_lj_sig):
        q_changed = not np.allclose(ligand_q, self.ligand_q0)
        lj_eps_changed = not np.allclose(ligand_lj_eps, self.ligand_lj_eps)
        lj_sig_changed = not np.allclose(ligand_lj_sig, self.ligand_lj_sig)

        Us = jnp.zeros(self.n_frames)
        if q_changed:
            if not self.supports_ligand_q:
                raise RuntimeError("did not construct this object to support varying ligand q")
            U_q = vmap(coulomb_interaction_group_energy, (0, 0))(ligand_q, self.q_prefactors)
            Us += U_q

        # lennard-jones: 2 code paths: costlier if LJ sigma is changing, cheaper if only LJ eps is changing
        if lj_sig_changed:
            if not self.supports_ligand_lj_sig:  # or (lj_eps_changed and not self.supports_ligand_lj_eps):
                raise RuntimeError("did not construct this object to support varying ligand LJ sig")
            U_lj = lj_interaction_group_energy(ligand_lj_sig, ligand_lj_eps, self.lj_sig_prefactors)
            Us += U_lj

        elif lj_eps_changed:
            if not self.supports_ligand_lj_eps:
                raise RuntimeError("did not construct this object to support varying ligand LJ eps")

            raise NotImplementedError("this branch is incomplete...")
            Us += U_lj
        assert 0, "incomplete"

        return Us
