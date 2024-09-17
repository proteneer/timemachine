from typing import Callable

import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import vmap
from numpy.typing import NDArray as Array

from timemachine.potentials import nonbonded
from timemachine.potentials.jax_utils import distance2

Position = Array
Param = Array
Box = Array
Energy = float
PairFxn = Callable[[Position, Position, Param, Param, Box], Energy]


# example pair function
def nb_pair_fxn(x_a, x_b, param_a, param_b, box):
    beta = 2.0
    cutoff = 1.2

    # alchemical distance
    r2 = distance2(x_a, x_b, box)
    w_offset = param_b[3] - param_a[3]
    r = jnp.sqrt(r2 + w_offset**2)

    # TM reaction field
    q_prod = param_a[0] * param_b[0]
    e_q = nonbonded.switched_direct_space_pme(r, q_prod, beta, cutoff)

    # Lennard-Jones
    sig = nonbonded.combining_rule_sigma(param_a[1], param_b[1])
    eps = nonbonded.combining_rule_epsilon(param_a[2], param_b[2])
    e_lj = nonbonded.lennard_jones(r, sig, eps)

    return jnp.where(r < cutoff, e_q + e_lj, 0.0)


# brute-force pre-computed neighborlist
@jit
def env_mask_within_cutoff(x_env, x_lig, box, cutoff):
    """result[i] = True if any distance(x_env[i], y) < cutoff for y in x_lig"""

    def d2_others(x_i, x_others):
        d2_ij = vmap(distance2, (None, 0, None))(x_i, x_others, box)
        return jnp.where(d2_ij <= cutoff**2, d2_ij, jnp.inf)

    def within_cutoff(point):
        return jnp.any(d2_others(point, x_lig) < cutoff**2)

    return vmap(within_cutoff)(x_env)


class InteractionGroupTraj:
    def __init__(self, pair_fxn: PairFxn, xs: Array, box_diags: Array, ligand_idxs: Array, env_idxs: Array, cutoff=1.2):
        r"""support [U_ig(x; params) for x in traj]

        where U_ig = \sum_i \sum_j pair_fxn(||x_j - x_j||; params_i, params_j)

        (with i summing over ligand_idxs, j summing over env_idxs, and pair_fxn(r) == 0 when r >= cutoff)
        """
        self.n_frames = len(xs)
        self.pair_fxn = pair_fxn
        self.ligand_idxs = ligand_idxs
        self.all_env_idxs = env_idxs
        num_lig, num_env = len(ligand_idxs), len(env_idxs)

        # apply pair_fxn(x_i, x_j, param_i, param_j, box)
        axes_a = (0, None, 0, None, None)
        axes_b = (None, 0, None, 0, None)
        self.all_pairs_fxn = vmap(vmap(pair_fxn, axes_a), axes_b)

        # TODO[symmetry]:
        #   maybe generalize so we select a subset of ligand atoms too...
        #   (currently: assume we always look at all ligand atoms)
        self.xs_lig = xs[:, ligand_idxs]
        _xs_env = xs[:, env_idxs]  # will select subsets...

        print(f"precomputing neighborlist on ({num_lig}, {num_env}) interaction group, at cutoff={cutoff}")

        # note: vmap here can consume excessive memory if len(env_idxs) * len(xs) is large -> python loop
        # mask = vmap(env_mask_within_cutoff, (0,0,0,None))(_xs_env, self.xs_lig, boxes, cutoff)
        def f(x_env, x_lig, box):
            return env_mask_within_cutoff(x_env, x_lig, box, cutoff)

        mask = np.array([f(_xs_env[i], self.xs_lig[i], np.diag(box_diags[i])) for i in range(self.n_frames)])

        padded_num_env_atoms = mask.sum(1).max()
        num_stored = padded_num_env_atoms + len(ligand_idxs)
        print(f"\tsaving {(xs.shape[1] / num_stored):.2f}x on storage\n\t(relative to storing all env atoms)")
        max_nbrs, mean_nbrs = padded_num_env_atoms, mask.sum(1).mean()
        print(f"\tpadding to max_nbrs = {max_nbrs}\n\t(~{max_nbrs/mean_nbrs:.2f}x larger than unpadded neighbor list)")

        idxs_within_env_block = np.argsort(mask, axis=1)[:, -padded_num_env_atoms:]
        self.selected_env_idxs = jnp.array(self.all_env_idxs[idxs_within_env_block], dtype=jnp.uint32)

        self.xs_env = np.array([_x_env[idxs] for (_x_env, idxs) in zip(_xs_env, idxs_within_env_block)])
        self.box_diags = box_diags

    def compute_Us(self, nb_params):
        nb_params = jnp.array(nb_params)
        lig_params = nb_params[self.ligand_idxs]

        @jit
        def U_snapshot(x_ligand, x_env, env_idxs, box_diag):
            env_params = nb_params[env_idxs]
            return jnp.sum(self.all_pairs_fxn(x_ligand, x_env, lig_params, env_params, jnp.diag(box_diag)))

        Us = vmap(U_snapshot, (0, 0, 0, 0))(self.xs_lig, self.xs_env, self.selected_env_idxs, self.box_diags)
        assert Us.shape == (self.n_frames,)
        return Us
