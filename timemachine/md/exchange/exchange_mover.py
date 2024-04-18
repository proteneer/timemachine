from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import special_ortho_group

from timemachine.constants import BOLTZ
from timemachine.md import moves
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import nonbonded


def get_water_idxs(mol_groups: List[NDArray], ligand_idxs: Optional[NDArray] = None) -> List[NDArray]:
    """Given a list of lists that make up the individual molecules in a system, return the subset that is only the waters.

    Contains additional logic to handle the case where ligand_idxs is also of size 3.
    """
    water_groups = [g for g in mol_groups if len(g) == 3]
    if ligand_idxs is not None and len(ligand_idxs) == 3:
        ligand_atom_set = set(ligand_idxs)
        water_groups = [g for g in water_groups if set(g) != ligand_atom_set]
    return water_groups


def randomly_rotate_and_translate(coords, new_loc):
    """
    Randomly rotate and translate coords such that the centroid of the displaced
    coordinates is equal to new_loc.
    """
    # coords: shape 3(OHH) x 3(xyz), water coordinates
    centroid = np.mean(coords, axis=0, keepdims=True)
    centered_coords = coords - centroid
    rot_mat = special_ortho_group.rvs(3)

    # equivalent to standard form where R is first:
    # (R @ A.T).T = A @ R.T
    rotated_coords = centered_coords @ rot_mat.T
    return rotated_coords + new_loc


def translate_coordinates(coords, new_loc):
    """
    Translate coords such that the centroid of the displaced
    coordinates is equal to new_loc.
    """
    centroid = np.mean(coords, axis=0, keepdims=True)
    centered_coords = coords - centroid
    return centered_coords + new_loc


class BDExchangeMove(moves.MonteCarloMove):
    """
    Untargeted, biased deletion move where we selectively prefer certain waters over others.
    """

    def __init__(
        self,
        nb_beta: float,
        nb_cutoff: float,
        nb_params: NDArray,
        water_idxs: NDArray,
        temperature: float,
    ):
        super().__init__()
        self.nb_beta = nb_beta
        self.nb_cutoff = nb_cutoff
        self.nb_params = jnp.array(nb_params)
        self.num_waters = len(water_idxs)

        assert self.num_waters > 0
        last_water_end = water_idxs[0][0] - 1
        for wi, wj, wk in water_idxs:
            assert wi == last_water_end + 1
            assert wi + 1 == wj
            assert wi + 2 == wk
            last_water_end = wk

        # Determine where the start of the waters begin so we can compute the water-water ixn energies
        # in the incremental version
        self.starting_water_position = water_idxs[0][0]
        self.water_idxs_jnp = jnp.array(water_idxs)  # make jit happy
        self.water_idxs_np = np.array(water_idxs)

        kT = BOLTZ * temperature
        self.beta = 1 / kT

        self.n_atoms = len(nb_params)
        all_a_idxs = []
        all_b_idxs = []
        for a_idxs in water_idxs:
            all_a_idxs.append(np.array(a_idxs))
            all_b_idxs.append(np.delete(np.arange(self.n_atoms), a_idxs))

        self.all_a_idxs = np.array(all_a_idxs)
        self.all_b_idxs = np.array(all_b_idxs)

        self.last_conf = None
        self.last_bw = None

        @jax.jit
        def U_fn_unsummed(conf, box, a_idxs, b_idxs):
            # compute the energy of an interaction group
            conf_i = conf[a_idxs]
            conf_j = conf[b_idxs]
            params_i = self.nb_params[a_idxs]
            params_j = self.nb_params[b_idxs]
            nrgs = nonbonded.nonbonded_block_unsummed(
                conf_i, conf_j, box, params_i, params_j, self.nb_beta, self.nb_cutoff
            )

            return jnp.where(jnp.isnan(nrgs), np.inf, nrgs)

        self.U_fn_unsummed = U_fn_unsummed

        @jax.jit
        def U_fn(conf, box, a_idxs, b_idxs):
            return jnp.sum(U_fn_unsummed(conf, box, a_idxs, b_idxs))

        def batch_U_fn(conf, box, all_a_idxs, all_b_idxs):
            nrgs = []
            for a_idxs, b_idxs in zip(all_a_idxs, all_b_idxs):
                nrgs.append(U_fn(conf, box, a_idxs, b_idxs))
            return jnp.array(nrgs)

        self.batch_U_fn = batch_U_fn

        def batch_log_weights(conf, box):
            """
            Return a list of energies equal to len(water_idxs)

            # Cached based on conf
            """
            if not np.array_equal(self.last_conf, conf):
                self.last_conf = conf
                tmp = self.beta * self.batch_U_fn(conf, box, self.all_a_idxs, self.all_b_idxs)
                self.last_bw = np.array(tmp)
            return self.last_bw

        self.batch_log_weights = batch_log_weights

        # @profile
        @jax.jit
        def batch_log_weights_incremental(conf, box, water_idx, new_pos, initial_weights):
            """Compute Z(x') incrementally using Z(x)"""
            conf = jnp.array(conf)

            assert len(initial_weights) == self.num_waters

            a_idxs = self.water_idxs_jnp[water_idx]
            b_idxs = jnp.delete(
                jnp.arange(self.n_atoms), a_idxs, assume_unique_indices=True
            )  # aui used to allow for jit

            # sum interaction energy of all the atoms in a water molecule
            old_water_ixn_nrgs = jnp.sum(self.beta * U_fn_unsummed(conf, box, a_idxs, b_idxs), axis=0)
            old_water_water_ixn_nrgs = jnp.sum(
                old_water_ixn_nrgs[self.starting_water_position :][: (self.num_waters - 1) * 3].reshape(
                    (self.num_waters - 1), 3
                ),
                axis=1,
            )

            assert len(old_water_water_ixn_nrgs) == self.num_waters - 1

            old_water_water_ixn_nrgs_full = jnp.insert(old_water_water_ixn_nrgs, water_idx, 0)
            new_conf = conf.copy()
            new_conf = new_conf.at[a_idxs].set(new_pos)
            new_water_ixn_nrgs = jnp.sum(self.beta * U_fn_unsummed(new_conf, box, a_idxs, b_idxs), axis=0)
            new_water_water_ixn_nrgs = jnp.sum(
                new_water_ixn_nrgs[self.starting_water_position :][: (self.num_waters - 1) * 3].reshape(
                    (self.num_waters - 1), 3
                ),
                axis=1,
            )

            new_water_water_ixn_nrgs_full = jnp.insert(new_water_water_ixn_nrgs, water_idx, 0)
            final_weights = initial_weights - old_water_water_ixn_nrgs_full + new_water_water_ixn_nrgs_full
            final_weights = final_weights.at[water_idx].set(jnp.sum(new_water_ixn_nrgs))

            # (ytz): sanity check to ensure we're doing incremental log weights correctly
            # note that jax 64bit needs to be enabled first.
            # ref_final_weights = self.batch_log_weights(new_conf, box)
            # np.testing.assert_almost_equal(np.array(final_weights), np.array(ref_final_weights))

            return final_weights, new_conf

        self.batch_log_weights_incremental = batch_log_weights_incremental
        self.batch_log_weights = batch_log_weights

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        coords = x.coords
        box = x.box
        log_weights_before = self.batch_log_weights(coords, box)
        log_probs_before = log_weights_before - logsumexp(log_weights_before)
        probs_before = np.exp(log_probs_before)
        chosen_water = np.random.choice(np.arange(self.num_waters), p=probs_before)
        chosen_water_atoms = self.water_idxs_np[chosen_water]

        # compute delta_U of insertion
        trial_chosen_coords = coords[chosen_water_atoms]
        trial_translation = np.diag(box) * np.random.rand(3)
        # tbd - what should we do  with velocities?

        moved_coords = randomly_rotate_and_translate(trial_chosen_coords, trial_translation)
        # moved_coords = translate_coordinates(trial_chosen_coords, trial_translation)

        # optimized version using double transposition
        log_weights_after, trial_coords = self.batch_log_weights_incremental(
            coords, box, chosen_water, moved_coords, log_weights_before
        )

        # reference version
        # trial_coords = coords.copy()  # can optimize this later if needed
        # trial_coords[chosen_water_atoms] = moved_coords
        # log_weights_after = self.batch_log_weights(trial_coords, box)

        log_acceptance_probability = np.minimum(logsumexp(log_weights_before) - logsumexp(log_weights_after), 0.0)
        new_state = CoordsVelBox(trial_coords, x.velocities, x.box)

        return new_state, log_acceptance_probability


# numpy version of delta_r
def delta_r_np(ri, rj, box):
    diff = ri - rj  # this can be either N,N,3 or B,3
    if box is not None:
        box_diag = np.diag(box)
        diff -= box_diag * np.floor(diff / box_diag + 0.5)
    return diff


def inner_insertion(radius, center, box):
    # (ytz): sample a point inside the sphere uniformly
    # source: https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
    xyz = np.random.randn(3)
    n = np.linalg.norm(xyz)
    xyz = xyz / n
    # (ytz): this might not be inside the box, but it doesn't matter
    # since all of our distances are computed using PBCs
    c = np.cbrt(np.random.rand())
    new_xyz = xyz * c * radius + center
    assert np.linalg.norm(delta_r_np(new_xyz, center, box)) < radius

    return new_xyz


def outer_insertion(radius, center, box):
    # generate random proposals outside of the sphere but inside the box
    for i in range(1000000):
        xyz = np.random.rand(3) * np.diag(box)
        if np.linalg.norm(delta_r_np(xyz, center, box)) >= radius:
            return xyz

    assert 0, "outer_insertion_failed"


def get_water_groups(coords, box, center, water_idxs, radius):
    """
    Partition water molecules into two groups, depending if it's inside the sphere or outside the sphere.
    """
    mol_centroids = np.mean(coords[water_idxs], axis=1)
    dijs = np.linalg.norm(delta_r_np(mol_centroids, center, box), axis=1)
    inner_mols = np.argwhere(dijs < radius).reshape(-1)
    outer_mols = np.argwhere(dijs >= radius).reshape(-1)

    assert len(inner_mols) + len(outer_mols) == len(water_idxs)
    return inner_mols, outer_mols


def compute_proposal_probabilities_given_counts(n_a, n_b):
    assert n_a >= 0
    assert n_b >= 0

    if n_a > 0 and n_b > 0:
        return 0.5
    elif n_a > 0 and n_b == 0:
        return 1.0
    elif n_a == 0 and n_b > 0:
        return 1.0
    else:
        # invalid corner
        assert 0


def compute_raw_ratio_given_weights(log_weights_before, log_weights_after, vi_mols, vj_mols, vol_i, vol_j):
    assert len(vi_mols) > 0

    # fwd counts
    fwd_n_i = len(vi_mols)
    fwd_n_j = len(vj_mols)

    # compute fwd_probability
    g_fwd = compute_proposal_probabilities_given_counts(fwd_n_i, fwd_n_j)

    # modify counts after water has been from vol_i -> vol_j
    rev_n_i = fwd_n_i - 1
    rev_n_j = fwd_n_j + 1

    g_rev = compute_proposal_probabilities_given_counts(rev_n_i, rev_n_j)

    raw_log_p = (
        logsumexp(log_weights_before)
        - logsumexp(log_weights_after)
        + np.log(vol_j)
        - np.log(vol_i)
        + np.log(g_rev)
        - np.log(g_fwd)
    )

    return raw_log_p


class TIBDExchangeMove(BDExchangeMove):
    r"""
    Targeted Insertion and Biased Deletion Exchange Move

    Insertions are targeted over two regions V1 and V2 such that V1 is a sphere whose origin
    is at the centroid of a set of indices, and V2 = box - V1.

    Deletions are biased such that each water x_i has a weight equal to exp(u_ixn_nrg(x_i, x \ x_i)).
    """

    def __init__(
        self,
        nb_beta: float,
        nb_cutoff: float,
        nb_params: NDArray,
        water_idxs: NDArray,
        temperature: float,
        ligand_idxs: List[int],
        radius: float,
    ):
        """
        Parameters
        ----
        nb_beta: float
            nonbonded beta parameters used in direct space pme

        nb_cutoff: float
            cutoff in the nonbonded kernel

        nb_params: NDArray (N,4)
            N is the total number of atoms in the system.
            Each 4-tuple of nonbonded parameters corresponds to (charge, sigma, epsilon, w coords)

        water_idxs: NDArray (W,3)
            W is the total number of water molecules in the system.
            Each element is a 3-tuple denoting the index for oxygen, hydrogen, hydrogen (or whatever is
            consistent with the nb_params)

        ligand_idxs: List[int]
            Indices corresponding to the atoms that should be used to compute the centroid

        temperature: float
            Temperature, in Kelvin

        radius: float
            Radius to use for the ligand_idxs

        """
        super().__init__(nb_beta, nb_cutoff, nb_params, water_idxs, temperature)

        self.ligand_idxs = np.array(ligand_idxs)  # used to determine center of sphere
        self.radius = radius

    def swap_vi_into_vj(
        self,
        vi_mols: List[int],
        vj_mols: List[int],
        x: CoordsVelBox,
        vj_site: NDArray,
        vol_i: float,
        vol_j: float,
    ):
        # optimized algorithm:
        # 1. compute batched log weights once for every water, this can be cached.
        # normalization constants are just partial sums over these log weights
        # 2. pick a random water to be inserted/deleted
        # 3. compute updated batched log weights with this new water using the transposition trick

        # The transposition trick allows us to compute the denominator efficiently by computing a 3x(N-3)
        # slice of the nonbonded matrix, as opposed to the full NxN matrix. The numerator is computed once
        # at the start of a batch of MC moves, and only re-computed/replaced with the denominator when a move
        # is accepted.

        # swap a water molecule from region vi to region vj
        coords, box = x.coords, x.box

        # compute weights of waters in the vi region
        # compute weights:
        # p = w_i / sum_j w_j
        # log p = log w_i - log sum_j w_j
        # log p = log exp u_i - log sum_j exp u_j
        # p = exp(u_i - log sum_j exp u_j)
        log_weights_before_full = self.batch_log_weights(coords, box)
        log_weights_before = log_weights_before_full[vi_mols]
        log_probs_before = log_weights_before - logsumexp(log_weights_before)
        probs_before = np.exp(log_probs_before)
        water_idx = np.random.choice(vi_mols, p=probs_before)

        chosen_water_atoms = self.water_idxs_np[water_idx]
        new_coords = coords[chosen_water_atoms]

        # remove centroid and offset into insertion site
        new_coords = randomly_rotate_and_translate(new_coords, vj_site)
        # new_coords = translate_coordinates(new_coords, vj_site) # keep for pedagogical utility

        vj_plus_one_idxs = np.concatenate([[water_idx], vj_mols])
        log_weights_after_full, trial_coords = self.batch_log_weights_incremental(
            coords, box, water_idx, new_coords, log_weights_before_full
        )
        trial_coords = np.array(trial_coords)
        log_weights_after_full = np.array(log_weights_after_full)
        log_weights_after = log_weights_after_full[vj_plus_one_idxs]

        raw_log_p = compute_raw_ratio_given_weights(
            log_weights_before, log_weights_after, vi_mols, vj_mols, vol_i, vol_j
        )

        log_p_accept = min(0, raw_log_p)

        new_state = CoordsVelBox(trial_coords, x.velocities, x.box)

        return new_state, log_p_accept

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        box = x.box
        coords = x.coords
        center = np.mean(coords[self.ligand_idxs], axis=0)
        inner_mols, outer_mols = get_water_groups(coords, box, center, self.water_idxs_np, self.radius)
        n1 = len(inner_mols)
        n2 = len(outer_mols)

        vol_1 = (4 / 3) * np.pi * self.radius**3
        vol_2 = np.prod(np.diag(box)) - vol_1

        # optimize this later
        v1_site = inner_insertion(self.radius, center, box)
        v2_site = outer_insertion(self.radius, center, box)

        if n1 == 0 and n2 == 0:
            assert 0
        elif n1 > 0 and n2 == 0:
            return self.swap_vi_into_vj(inner_mols, outer_mols, x, v2_site, vol_1, vol_2)
        elif n1 == 0 and n2 > 0:
            return self.swap_vi_into_vj(outer_mols, inner_mols, x, v1_site, vol_2, vol_1)
        elif n1 > 0 and n2 > 0:
            if np.random.rand() < 0.5:
                return self.swap_vi_into_vj(inner_mols, outer_mols, x, v2_site, vol_1, vol_2)
            else:
                return self.swap_vi_into_vj(outer_mols, inner_mols, x, v1_site, vol_2, vol_1)
        else:
            # negative numbers, dun goof'd
            assert 0
