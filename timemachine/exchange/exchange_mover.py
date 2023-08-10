# Prototype ExchangeMover that implements an instantaneous water swap move.
# disable for ~2x speed-up
# from jax import config
# config.update("jax_enable_x64", True)

import argparse
import os
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from openmm import app
from rdkit import Chem
from scipy.stats import special_ortho_group

from timemachine.constants import AVOGADRO, DEFAULT_KT, DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.datasets.water_exchange import charges
from timemachine.fe import cif_writer
from timemachine.fe.free_energy import AbsoluteFreeEnergy, HostConfig, InitialState, image_frames
from timemachine.fe.topology import BaseTopology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.ff.handlers.nonbonded import PrecomputedChargeHandler
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md import moves
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.builders import strip_units
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import nonbonded

# uncomment if we want to re-enable minimization
# from timemachine.md.minimizer import minimize_host_4d


def build_system(host_pdbfile: str, water_ff: str, padding: float):
    if isinstance(host_pdbfile, str):
        assert os.path.exists(host_pdbfile)
        host_pdb = app.PDBFile(host_pdbfile)
    elif isinstance(host_pdbfile, app.PDBFile):
        host_pdb = host_pdbfile
    else:
        raise TypeError("host_pdbfile must be a string or an openmm PDBFile object")

    host_coords = strip_units(host_pdb.positions)

    box_lengths = np.amax(host_coords, axis=0) - np.amin(host_coords, axis=0)

    box_lengths = box_lengths + padding
    box = np.eye(3, dtype=np.float64) * box_lengths

    host_ff = app.ForceField(f"{water_ff}.xml")
    nwa = len(host_coords)
    solvated_host_system = host_ff.createSystem(
        host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    solvated_host_coords = host_coords
    solvated_topology = host_pdb.topology

    return solvated_host_system, solvated_host_coords, box, solvated_topology, nwa


def randomly_rotate_and_translate(coords, offset):
    # coords: shape 3(OHH) x 3(xyz), water coordinates
    centroid = np.mean(coords, axis=0, keepdims=True)
    centered_coords = coords - centroid
    rot_mat = special_ortho_group.rvs(3)
    rotated_coords = np.matmul(centered_coords, rot_mat)
    return rotated_coords + offset


def randomly_translate(coords, offset):
    centroid = np.mean(coords, axis=0, keepdims=True)
    centered_coords = coords - centroid
    return centered_coords + offset


def get_initial_state(water_pdb, mol, ff, seed, nb_cutoff):
    assert nb_cutoff == 1.2  # hardcoded in prepare_host_edge

    # read water system
    solvent_sys, solvent_conf, solvent_box, solvent_topology, num_water_atoms = build_system(
        water_pdb, ff.water_ff, padding=0.1
    )
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes

    assert num_water_atoms == len(solvent_conf)

    bt = BaseTopology(mol, ff)
    afe = AbsoluteFreeEnergy(mol, bt)
    host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, num_water_atoms)
    potentials, params, combined_masses = afe.prepare_host_edge(ff.get_params(), host_config, 0.0)

    host_bps = []
    for p, bp in zip(params, potentials):
        host_bps.append(bp.bind(p))

    temperature = DEFAULT_TEMP
    dt = 1e-3

    integrator = LangevinIntegrator(temperature, dt, 1.0, combined_masses, seed)

    bond_list = get_bond_list(host_bps[0].potential)
    group_idxs = get_group_indices(bond_list, len(combined_masses))
    barostat_interval = 25

    barostat = MonteCarloBarostat(
        len(combined_masses), DEFAULT_PRESSURE, temperature, group_idxs, barostat_interval, seed + 1
    )

    # (YTZ): This is disabled because the initial ligand and starting waters are pre-minimized
    # print("Minimizing the host system...", end="", flush=True)
    # host_conf = minimize_host_4d([mol], host_config, ff)
    # print("Done", flush=True)

    host_conf = solvent_conf
    combined_conf = np.concatenate([host_conf, get_romol_conf(mol)], axis=0)

    ligand_idxs = np.arange(num_water_atoms, num_water_atoms + mol.GetNumAtoms())

    initial_state = InitialState(
        host_bps, integrator, barostat, combined_conf, np.zeros_like(combined_conf), solvent_box, 0.0, ligand_idxs
    )

    num_water_mols = num_water_atoms // 3

    return initial_state, num_water_mols, solvent_topology


from scipy.special import logsumexp


class ExchangeMove(moves.MonteCarloMove):
    def __init__(self, nb_beta, nb_cutoff, nb_params, water_idxs):
        self.nb_beta = nb_beta
        self.nb_cutoff = nb_cutoff
        self.nb_params = jnp.array(nb_params)
        self.num_waters = len(water_idxs)
        self.water_idxs = water_idxs
        self.beta = 1 / DEFAULT_KT

        self.n_atoms = len(nb_params)
        # for a_idxs, b_idxs in all_a_idxs, all_b_idxs:
        self.all_a_idxs = []
        self.all_b_idxs = []
        for a_idxs in water_idxs:
            self.all_a_idxs.append(np.array(a_idxs))
            self.all_b_idxs.append(np.delete(np.arange(self.n_atoms), a_idxs))
        self.all_a_idxs = np.array(self.all_a_idxs)
        self.all_b_idxs = np.array(self.all_b_idxs)

        @jax.jit
        def U_fn(conf, box, a_idxs, b_idxs):
            # compute the energy of an interaction group
            conf_i = conf[a_idxs]
            conf_j = conf[b_idxs]
            params_i = self.nb_params[a_idxs]
            params_j = self.nb_params[b_idxs]
            return nonbonded.nonbonded_block(conf_i, conf_j, box, params_i, params_j, self.nb_beta, self.nb_cutoff)

        @jax.jit
        def U_fn_unsummed(conf, box, a_idxs, b_idxs):
            # compute the energy of an interaction group
            conf_i = conf[a_idxs]
            conf_j = conf[b_idxs]
            params_i = self.nb_params[a_idxs]
            params_j = self.nb_params[b_idxs]
            return nonbonded.nonbonded_block_unsummed(
                conf_i, conf_j, box, params_i, params_j, self.nb_beta, self.nb_cutoff
            )

        self.batch_U_fn = jax.jit(jax.vmap(U_fn, (None, None, 0, 0)))

        def batch_log_weights(conf, box):
            """
            Return a list of energies equal to len(water_idxs)
            """
            nrgs = self.batch_U_fn(conf, box, self.all_a_idxs, self.all_b_idxs)
            log_weights = self.beta * nrgs  # note the positive energy!
            return log_weights

        def batch_log_weights_incremental(conf, box, water_idx, new_pos):
            # compute the incremental weights
            initial_weights = self.batch_log_weights(conf, box)

            assert len(initial_weights) == self.num_waters

            print("NUM WATERS", len(self.water_idxs))
            print("IWS", initial_weights.shape)

            a_idxs = self.water_idxs[water_idx]
            b_idxs = np.delete(np.arange(self.n_atoms), a_idxs)

            print("A_IDXS", len(a_idxs))
            print("B_IDXS", len(b_idxs))

            old_water_ixn_nrgs = np.sum(self.beta * U_fn_unsummed(conf, box, a_idxs, b_idxs), axis=0)
            old_water_water_ixn_nrgs = np.sum(
                old_water_ixn_nrgs[: (self.num_waters - 1) * 3].reshape((self.num_waters - 1), 3), axis=1
            )
            old_water_water_ixn_nrgs_full = np.zeros(len(self.water_idxs))
            old_water_water_ixn_nrgs_full[:water_idx] = old_water_water_ixn_nrgs[:water_idx]
            old_water_water_ixn_nrgs_full[water_idx + 1 :] = old_water_water_ixn_nrgs[water_idx:]

            # we have one fewer waters now - subtract differential
            new_conf = conf.copy()
            new_conf[a_idxs] = new_pos
            new_water_ixn_nrgs = np.sum(self.beta * U_fn_unsummed(new_conf, box, a_idxs, b_idxs), axis=0)
            new_water_water_ixn_nrgs = np.sum(
                new_water_ixn_nrgs[: (self.num_waters - 1) * 3].reshape((self.num_waters - 1), 3), axis=1
            )
            new_water_water_ixn_nrgs_full = np.zeros(len(self.water_idxs))
            new_water_water_ixn_nrgs_full[:water_idx] = new_water_water_ixn_nrgs[:water_idx]
            new_water_water_ixn_nrgs_full[water_idx + 1 :] = new_water_water_ixn_nrgs[water_idx:]

            final_weights = initial_weights - old_water_water_ixn_nrgs_full + new_water_water_ixn_nrgs_full
            # final_weights = initial_weights - new_water_water_ixn_nrgs

            ref_final_weights = self.batch_log_weights(new_conf, box)

            # print(type(final_weights))
            # print(type(ref_final_weights))
            np.testing.assert_almost_equal(np.array(final_weights), np.array(ref_final_weights))

            return final_weights, new_conf

        self.batch_log_weights_incremental = batch_log_weights_incremental
        self.batch_log_weights = batch_log_weights

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        coords = x.coords
        box = x.box
        log_weights_before = self.batch_log_weights(coords, box)
        # print(log_weights_before)
        log_probs_before = log_weights_before - logsumexp(log_weights_before)
        probs_before = np.exp(log_probs_before)

        # chosen_water = np.random.randint(self.num_waters)
        # print("LPB", log_probs_before)
        chosen_water = np.random.choice(np.arange(self.num_waters), p=probs_before)

        # print("picked", chosen_water)

        chosen_water_atoms = self.water_idxs[chosen_water]

        # compute delta_U of insertion
        trial_chosen_coords = coords[chosen_water_atoms]
        trial_translation = np.diag(box) * np.random.rand(3)
        # tbd - what should we do  with velocities?

        moved_coords = randomly_rotate_and_translate(trial_chosen_coords, trial_translation)
        # trial_coords = coords.copy()  # can optimize this later if needed
        # trial_coords[chosen_water_atoms] = moved_coords

        log_weights_after, trial_coords = self.batch_log_weights_incremental(coords, box, chosen_water, moved_coords)
        # log_weights_after = self.batch_log_weights(trial_coords, box)
        # this can be computed using a simple transposition later on

        log_p_accept = min(0, logsumexp(log_weights_before) - logsumexp(log_weights_after))

        # convert to inf if we get a nan
        # if np.isnan(delta_U_total):
        # delta_U_total = np.inf
        # log_p_accept = min(0, -self.beta * delta_U_total)

        new_state = CoordsVelBox(trial_coords, x.velocities, x.box)

        return new_state, log_p_accept


# numpy version of delta_r
def delta_r_np(ri, rj, box):
    diff = ri - rj  # this can be either N,N,3 or B,3
    if box is not None:
        box_diag = np.diag(box)
        diff -= box_diag * np.floor(diff / box_diag + 0.5)
    return diff


class InsideOutsideExchangeMove(moves.MonteCarloMove):
    """
    Special case of DualRegion swaps where we have two regions V1 and V2 such that
    V1 = spherical shell centered on a set of indices, and V2 = box - V1.

    This class explicitly attempts swaps between the two regions with equal probability.
    """

    def __init__(self, nb_beta, nb_cutoff, nb_params, water_idxs, ligand_idxs, beta, radius):
        self.nb_beta = nb_beta
        self.nb_cutoff = nb_cutoff
        self.nb_params = jnp.array(nb_params)
        self.num_waters = len(water_idxs)
        self.water_idxs = water_idxs
        self.ligand_idxs = ligand_idxs  # used to determine center of sphere
        self.beta = beta
        self.radius = radius

        # @jax.jit
        def U_fn(conf, box, a_idxs, b_idxs):
            # compute the energy of an interaction group
            conf_i = conf[a_idxs]
            conf_j = conf[b_idxs]
            params_i = self.nb_params[a_idxs]
            params_j = self.nb_params[b_idxs]
            return nonbonded.nonbonded_block(conf_i, conf_j, box, params_i, params_j, self.nb_beta, self.nb_cutoff)

        @jax.jit
        def delta_U_total_fn(trial_coords, coords, box, a_idxs, b_idxs):
            delta_U_insert = U_fn(trial_coords, box, a_idxs, b_idxs)
            delta_U_delete = -U_fn(coords, box, a_idxs, b_idxs)
            delta_U_total = delta_U_delete + delta_U_insert
            return delta_U_total

        self.batch_U_fn = jax.jit(jax.vmap(U_fn, (None, None, 0, 0)))

        # vectorized over multiple sets of a_idxs and b_idxs
        def batch_log_weights(conf, box, all_a_idxs, all_b_idxs):
            """
            Return a list of energies equal to len(water_idxs)
            """
            nrgs = self.batch_U_fn(conf, box, np.array(all_a_idxs), np.array(all_b_idxs))
            log_weights = self.beta * nrgs  # note the positive interaction energy!
            return log_weights

        self.batch_log_weights = batch_log_weights

        self.delta_U_total_fn = delta_U_total_fn

    def get_water_groups(self, coords, box, center):
        dijs = np.linalg.norm(delta_r_np(np.mean(coords[self.water_idxs], axis=1), center, box), axis=1)
        v1_mols = np.argwhere(dijs < self.radius).reshape(-1)
        v2_mols = np.argwhere(dijs >= self.radius).reshape(-1)
        return v1_mols, v2_mols

    # @profile
    def swap_vi_into_vj(
        self,
        vi_mols: List[int],
        vj_mols: List[int],
        x: CoordsVelBox,
        vj_insertion_fn: Callable,
        vol_i: float,
        vol_j: float,
    ):
        # swap a water molecule from region vi to region vj
        coords, box = x.coords, x.box

        n_atoms = len(coords)

        # compute weights of waters in the vi region
        before_all_a_idxs = []
        before_all_b_idxs = []
        for vi_mol in vi_mols:
            a_idxs = self.water_idxs[vi_mol]
            before_all_a_idxs.append(a_idxs)
            before_all_b_idxs.append(np.delete(np.arange(n_atoms), a_idxs))

        # compute weights:
        # p = w_i / sum_j w_j
        # log p = log w_i - log sum_j w_j
        # log p = log exp u_i - log sum_j exp u_j
        # p = exp(u_i - log sum_j exp u_j)
        log_weights_before = self.batch_log_weights(coords, box, before_all_a_idxs, before_all_b_idxs)
        log_probs_before = log_weights_before - logsumexp(log_weights_before)
        probs_before = np.exp(log_probs_before)
        choiche = np.random.choice(np.arange(len(before_all_a_idxs)), p=probs_before)

        chosen_water_atoms = before_all_a_idxs[choiche]
        new_coords = coords[chosen_water_atoms]
        # remove centroid and offset into insertion site
        new_coords = randomly_rotate_and_translate(new_coords, vj_insertion_fn())

        trial_coords = coords.copy()  # can optimize this later if needed
        trial_coords[chosen_water_atoms] = new_coords

        # move water into v_j, and compute Z(x').
        # this can in theory be computed using a simple transposition
        after_all_a_idxs = [chosen_water_atoms]
        after_all_b_idxs = [np.delete(np.arange(n_atoms), chosen_water_atoms)]
        for vj_mol in vj_mols:
            a_idxs = self.water_idxs[vj_mol]
            after_all_a_idxs.append(a_idxs)
            after_all_b_idxs.append(np.delete(np.arange(n_atoms), a_idxs))

        log_weights_after = self.batch_log_weights(trial_coords, box, after_all_a_idxs, after_all_b_idxs)

        log_p_accept = min(
            0, logsumexp(log_weights_before) - logsumexp(log_weights_after) + np.log(vol_j) - np.log(vol_i)
        )

        new_state = CoordsVelBox(trial_coords, x.velocities, x.box)

        return new_state, log_p_accept

    # def swap_vi_into_vj(
    #     self,
    #     vi_mols: List[int],
    #     vj_mols: List[int],
    #     x: CoordsVelBox,
    #     vj_insertion_fn: Callable,
    #     vol_i: float,
    #     vol_j: float,
    # ):
    #     chosen_water = np.random.choice(vi_mols)
    #     N_i = len(vi_mols)
    #     N_j = len(vj_mols)
    #     insertion_site = vj_insertion_fn()
    #     new_coords, log_p_accept = self.swap_vi_into_vj_impl(
    #         chosen_water, N_i, N_j, x.coords, x.box, insertion_site, vol_i, vol_j
    #     )

    #     return CoordsVelBox(new_coords, x.velocities, x.box), log_p_accept

    # def swap_vi_into_vj_impl(self, chosen_water, N_i, N_j, coords, box, insertion_site, vol_i: float, vol_j: float):
    #     assert N_i + N_j == len(self.water_idxs)
    #     # swap a water molecule from region vi to region vj
    #     # coords, box = x.coords, x.box
    #     # chosen_water = np.random.choice(vi_mols)
    #     chosen_water_atoms = self.water_idxs[chosen_water]
    #     new_coords = coords[chosen_water_atoms]
    #     # remove centroid and offset into insertion site
    #     new_coords = new_coords - np.mean(new_coords, axis=0) + insertion_site

    #     # debug
    #     # insertion_into_buckyball = len(vi_mols) > len(vj_mols)
    #     # if insertion_into_buckyball:
    #     # new_coords

    #     trial_coords = coords.copy()  # can optimize this later if needed
    #     trial_coords[chosen_water_atoms] = new_coords

    #     n_atoms = len(coords)
    #     a_idxs = chosen_water_atoms
    #     b_idxs = np.delete(np.arange(n_atoms), a_idxs)

    #     # we can probably speed this up even more if we use an incremental voxel_map
    #     # (reduce complexity from O(N) to O(K))
    #     # delta_U_delete = -self.U_fn(coords, box, a_idxs, b_idxs)
    #     # delta_U_insert = self.U_fn(trial_coords, box, a_idxs, b_idxs)
    #     # delta_U_total = delta_U_delete + delta_U_insert

    #     delta_U_total = self.delta_U_total_fn(trial_coords, coords, box, a_idxs, b_idxs)
    #     delta_U_total = np.asarray(delta_U_total)

    #     # convert to inf if we get a nan
    #     if np.isnan(delta_U_total):
    #         delta_U_total = np.inf

    #     # ni = len(vi_mols)
    #     # nj = len(vj_mols)
    #     hastings_factor = np.log((N_i * vol_j) / ((N_j + 1) * vol_i))
    #     # print("REF HF", hastings_factor)
    #     log_p_accept = min(0, -self.beta * delta_U_total + hastings_factor)
    #     # new_state = CoordsVelBox(trial_coords, x.velocities, x.box)

    #     return trial_coords, log_p_accept

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        box = x.box
        coords = x.coords
        center = np.mean(coords[self.ligand_idxs], axis=0)
        v1_mols, v2_mols = self.get_water_groups(coords, box, center)
        n1 = len(v1_mols)
        n2 = len(v2_mols)

        # optimized version
        def v1_insertion():
            # sample a point inside the sphere uniformly
            # source: https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
            xyz = np.random.randn(3)
            n = np.linalg.norm(xyz)
            xyz = xyz / n
            # (ytz): this might not be inside the box, but it doesnt' matter
            # since all of our distances are computed using PBCs
            c = np.cbrt(np.random.rand())
            new_xyz = xyz * c * self.radius + center

            # print("without box", np.linalg.norm(new_xyz - center), self.radius)
            # print("with box", np.linalg.norm(delta_r_np(new_xyz, center, box)), self.radius)

            assert np.linalg.norm(delta_r_np(new_xyz, center, box)) < self.radius

            return new_xyz

        # v1 << v2 so monte carlo is really slow (avg counter ~5000)
        # def v1_insertion():
        #     # generate random proposals inside the sphere
        #     # counter = 0
        #     while True:
        #         # counter += 1
        #         xyz = np.random.randn(3) * np.diag(box)
        #         if np.linalg.norm(delta_r_np(xyz, center, box)) < self.radius:
        #             # print("inserted into v1 after", counter)
        #             return xyz

        # v2 >> v1 so monte carlo works really well
        def v2_insertion():
            # generate random proposals outside of the sphere but inside the box
            while True:
                xyz = np.random.rand(3) * np.diag(box)
                if np.linalg.norm(delta_r_np(xyz, center, box)) >= self.radius:
                    return xyz

        vol_1 = (4 / 3) * np.pi * self.radius ** 3
        vol_2 = np.prod(np.diag(box)) - vol_1

        if n1 == 0 and n2 == 0:
            assert 0
        elif n1 > 0 and n2 == 0:
            return self.swap_vi_into_vj(v1_mols, v2_mols, x, v2_insertion, vol_1, vol_2)
        elif n1 == 0 and n2 > 0:
            return self.swap_vi_into_vj(v2_mols, v1_mols, x, v1_insertion, vol_2, vol_1)
        elif n1 > 0 and n2 > 0:
            if np.random.rand() < 0.5:
                return self.swap_vi_into_vj(v1_mols, v2_mols, x, v2_insertion, vol_1, vol_2)
            else:
                return self.swap_vi_into_vj(v2_mols, v1_mols, x, v1_insertion, vol_2, vol_1)
        else:
            # negative numbers, dun goof'd
            assert 0


from timemachine.lib.custom_ops import InsideOutsideExchangeMover


def compute_density(n_waters, box):
    box_vol = np.prod(np.diag(box))
    numerator = n_waters * 18.01528 * 1e27
    denominator = box_vol * AVOGADRO * 1000
    return numerator / denominator


def compute_occupancy(x_t, box_t, ligand_idxs, threshold):
    # compute the number of waters inside the buckyball ligand

    num_ligand_atoms = len(ligand_idxs)
    num_host_atoms = len(x_t) - num_ligand_atoms
    host_coords = x_t[:num_host_atoms]
    ligand_coords = x_t[num_host_atoms:]
    ligand_centroid = np.mean(ligand_coords, axis=0)
    count = 0
    for rj in host_coords:
        diff = delta_r_np(ligand_centroid, rj, box_t)
        dij = np.linalg.norm(diff)
        if dij < threshold:
            count += 1
    return count


def setup_forcefield(charges):
    # use a simple charge handler on the ligand to avoid running AM1 on a buckyball
    ff = Forcefield.load_default()
    q_handle = PrecomputedChargeHandler(charges)
    q_handle_intra = PrecomputedChargeHandler(charges)
    q_handle_solv = PrecomputedChargeHandler(charges)
    return Forcefield(
        ff.hb_handle,
        ff.ha_handle,
        ff.pt_handle,
        ff.it_handle,
        q_handle=q_handle,
        q_handle_solv=q_handle_solv,
        q_handle_intra=q_handle_intra,
        lj_handle=ff.lj_handle,
        lj_handle_solv=ff.lj_handle_solv,
        lj_handle_intra=ff.lj_handle_intra,
        protein_ff=ff.protein_ff,
        water_ff=ff.water_ff,
    )


def image_xvb(initial_state, xvb_t):
    new_coords = image_frames(initial_state, [xvb_t.coords], [xvb_t.box])[0]
    return CoordsVelBox(new_coords, xvb_t.velocities, xvb_t.box)


class CppMCMover(InsideOutsideExchangeMover):
    n_proposed: int = 0
    n_accepted: int = 0

    # def __init__(self, impl):
    # self._impl = impl

    # def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
    #     """return proposed state and log acceptance probability"""
    #     raise NotImplementedError

    def move(self, x: CoordsVelBox) -> CoordsVelBox:
        new_coords, log_acceptance_probability = self.propose(x.coords, x.box)
        self.n_proposed += 1

        alpha = np.random.rand()
        acceptance_probability = np.exp(log_acceptance_probability)
        if alpha < acceptance_probability:
            self.n_accepted += 1
            proposal = CoordsVelBox(new_coords, x.velocities, x.box)
            return proposal
        else:
            return x


def test_exchange():
    parser = argparse.ArgumentParser(description="Test the exchange protocol in a box of water.")
    parser.add_argument("--water_pdb", type=str, help="Location of the water PDB", required=True)
    parser.add_argument("--ligand_sdf", type=str, help="SDF file containing the ligand of interest", required=True)
    parser.add_argument("--out_cif", type=str, help="Output cif file", required=True)
    parser.add_argument(
        "--ligand_charges",
        type=str,
        help='Allowed values: "zero" or "espaloma"',
        required=True,
    )

    args = parser.parse_args()

    suppl = list(Chem.SDMolSupplier(args.ligand_sdf, removeHs=False))
    mol = suppl[0]

    if args.ligand_charges == "espaloma":
        esp = charges.espaloma_charges()  # charged system via espaloma
    elif args.ligand_charges == "zero":
        esp = np.zeros(mol.GetNumAtoms())  # decharged system
    else:
        assert 0, "Unknown charge model for the ligand"

    ff = setup_forcefield(esp)

    seed = 2024

    np.random.seed(seed)

    nb_cutoff = 1.2  # this has to be 1.2 since the builders hard code this in (should fix later)
    initial_state, nwm, topology = get_initial_state(args.water_pdb, mol, ff, seed, nb_cutoff)

    # set up water indices, assumes that waters are placed at the front of the coordinates.
    water_idxs = []
    for wai in range(nwm):
        water_idxs.append([wai * 3 + 0, wai * 3 + 1, wai * 3 + 2])

    water_idxs = np.array(water_idxs)
    bps = initial_state.potentials

    # [0] nb_all_pairs, [1] nb_ligand_water, [2] nb_ligand_protein
    nb_beta = bps[-1].potential.potentials[1].beta
    nb_cutoff = bps[-1].potential.potentials[1].cutoff
    nb_water_ligand_params = bps[-1].potential.params_init[1]

    print("number of water atoms", nwm * 3, "number of ligand atoms", mol.GetNumAtoms())
    print("water_ligand parameters", nb_water_ligand_params)

    bb_radius = 0.46
    # exc_mover = CppMCMover(
    #     nb_beta,
    #     nb_cutoff,
    #     nb_water_ligand_params.reshape(-1).tolist(),
    #     water_idxs.reshape(-1).tolist(),
    #     initial_state.ligand_idxs.reshape(-1).tolist(),
    #     1 / DEFAULT_KT,
    #     bb_radius,
    # )

    # reference
    # exc_mover = InsideOutsideExchangeMove(
    # nb_beta, nb_cutoff, nb_water_ligand_params, water_idxs, initial_state.ligand_idxs, 1 / DEFAULT_KT, bb_radius
    # )

    # vanilla reference
    exc_mover = ExchangeMove(nb_beta, nb_cutoff, nb_water_ligand_params, water_idxs)

    cur_box = initial_state.box0
    cur_x_t = initial_state.x0
    cur_v_t = np.zeros_like(cur_x_t)

    seed = 2023
    writer = cif_writer.CIFWriter([topology, mol], args.out_cif)
    cur_x_t = image_frames(initial_state, [cur_x_t], [cur_box])[0]
    writer.write_frame(cur_x_t * 10)

    npt_mover = moves.NPTMove(
        bps=initial_state.potentials,
        masses=initial_state.integrator.masses,
        temperature=initial_state.integrator.temperature,
        pressure=initial_state.barostat.pressure,
        n_steps=None,
        seed=seed,
        dt=initial_state.integrator.dt,
        friction=initial_state.integrator.friction,
        barostat_interval=initial_state.barostat.interval,
    )

    # equilibration
    print("Equilibrating the system... ", end="", flush=True)

    equilibration_steps = 50000

    # equilibrate using the npt mover
    npt_mover.n_steps = equilibration_steps
    xvb_t = CoordsVelBox(cur_x_t, cur_v_t, cur_box)
    # xvb_t = npt_mover.move(xvb_t)
    print("done")
    md_steps_per_batch = 2000
    npt_mover.n_steps = md_steps_per_batch

    # TBD: cache the minimized and equilibrated initial structure later on to iterate faster.
    mc_steps_per_batch = 10

    # (ytz): If I start with pure MC, and no MD, it's actually very easy to remove the waters.
    # since the starting waters have very very high energy. If I re-run MD, then it becomes progressively harder
    # remove the water since we will re-equilibriate the waters.

    # for idx in range(100000):
    for idx in range(100000):
        density = compute_density(nwm, xvb_t.box)

        xvb_t = image_xvb(initial_state, xvb_t)
        occ = compute_occupancy(xvb_t.coords, xvb_t.box, initial_state.ligand_idxs, threshold=bb_radius)

        print(
            f"{exc_mover.n_accepted} / {exc_mover.n_proposed} | density {density} | # of waters in bb {occ // 3} | md step: {idx * md_steps_per_batch}",
            flush=True,
        )

        # start_time = time.time()
        for _ in range(mc_steps_per_batch):
            assert np.amax(np.abs(xvb_t.coords)) < 1e3
            xvb_t = exc_mover.move(xvb_t)
        # print("time per mc move", (time.time() - start_time) / mc_steps_per_batch)

        # run MD
        xvb_t = npt_mover.move(xvb_t)  # disabling this gets more moves?


if __name__ == "__main__":
    # A trajectory is written out called water.cif
    # To visualize it, run: pymol water.cif (note that the simulation does not have to be complete to visualize progress)

    # example invocation:

    # start with 6 waters, using espaloma charges:
    # python timemachine/exchange/exchange_mover.py --water_pdb timemachine/datasets/water_exchange/bb_6_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered.sdf --ligand_charges espaloma --out_cif traj_6_waters.cif

    # start with 0 waters, using zero charges:
    # python timemachine/exchange/exchange_mover.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered.sdf --ligand_charges zero --out_cif traj_0_waters.cif
    test_exchange()
