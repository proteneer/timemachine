# Prototype ExchangeMover that implements an instantaneous water swap move.
# disable for ~2x speed-up
# from jax import config

# config.update("jax_enable_x64", True)

import argparse
import os
import sys
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
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
from timemachine.ff.handlers import openmm_deserializer
from timemachine.ff.handlers.nonbonded import PrecomputedChargeHandler
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md import moves
from timemachine.md.barostat.moves import NPTMove
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


def randomly_translate(coords, new_loc):
    """
    Translate coords such that the centroid of the displaced
    coordinates is equal to new_loc.
    """
    centroid = np.mean(coords, axis=0, keepdims=True)
    centered_coords = coords - centroid
    return centered_coords + new_loc


def get_initial_state(water_pdb, mol, ff, seed, nb_cutoff):
    assert nb_cutoff == 1.2  # hardcoded in prepare_host_edge

    # read water system
    solvent_sys, solvent_conf, solvent_box, solvent_topology, num_water_atoms = build_system(
        water_pdb, ff.water_ff, padding=0.1
    )
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes

    assert num_water_atoms == len(solvent_conf)

    if mol:
        bt = BaseTopology(mol, ff)
        afe = AbsoluteFreeEnergy(mol, bt)
        host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, num_water_atoms)
        potentials, params, combined_masses = afe.prepare_host_edge(ff.get_params(), host_config, 0.0)

        host_bps = []
        for p, bp in zip(params, potentials):
            host_bps.append(bp.bind(p))

        # (YTZ): This is disabled because the initial ligand and starting waters are pre-minimized
        # print("Minimizing the host system...", end="", flush=True)
        # host_conf = minimize_host_4d([mol], host_config, ff)
        # print("Done", flush=True)

        ligand_idxs = np.arange(num_water_atoms, num_water_atoms + mol.GetNumAtoms())
        final_masses = combined_masses
        final_conf = np.concatenate([solvent_conf, get_romol_conf(mol)], axis=0)

    else:
        host_bps, host_masses = openmm_deserializer.deserialize_system(solvent_sys, cutoff=nb_cutoff)
        ligand_idxs = [0, 1, 2]  # pick first water molecule to be a "ligand" for targetting purposes
        final_masses = host_masses
        final_conf = solvent_conf

    temperature = DEFAULT_TEMP
    dt = 1e-3
    barostat_interval = 25
    integrator = LangevinIntegrator(temperature, dt, 1.0, final_masses, seed)
    bond_list = get_bond_list(host_bps[0].potential)
    group_idxs = get_group_indices(bond_list, len(final_masses))
    barostat = MonteCarloBarostat(
        len(final_masses), DEFAULT_PRESSURE, temperature, group_idxs, barostat_interval, seed + 1
    )

    initial_state = InitialState(
        host_bps, integrator, barostat, final_conf, np.zeros_like(final_conf), solvent_box, 0.0, ligand_idxs
    )

    num_water_mols = num_water_atoms // 3

    return initial_state, num_water_mols, solvent_topology


from scipy.special import logsumexp


class BDExchangeMove(moves.MetropolisHastingsMove):
    """
    Untargetted, biased deletion move where we selectively prefer certain waters over others.
    """

    def __init__(
        self,
        nb_beta: float,
        nb_cutoff: float,
        nb_params: NDArray,
        water_idxs: NDArray,
        beta: float,
    ):
        super().__init__()
        self.nb_beta = nb_beta
        self.nb_cutoff = nb_cutoff
        self.nb_params = jnp.array(nb_params)
        self.num_waters = len(water_idxs)

        for wi, wj, wk in water_idxs:
            assert wi + 1 == wj
            assert wi + 2 == wk

        self.water_idxs_jnp = jnp.array(water_idxs)  # make jit happy
        self.water_idxs_np = np.array(water_idxs)

        self.beta = beta

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

        @jax.jit
        def U_fn(conf, box, a_idxs, b_idxs):
            return jnp.sum(U_fn_unsummed(conf, box, a_idxs, b_idxs))

        self.batch_U_fn = jax.jit(jax.vmap(U_fn, (None, None, 0, 0)))

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
                old_water_ixn_nrgs[: (self.num_waters - 1) * 3].reshape((self.num_waters - 1), 3), axis=1
            )

            assert len(old_water_water_ixn_nrgs) == self.num_waters - 1

            old_water_water_ixn_nrgs_full = jnp.insert(old_water_water_ixn_nrgs, water_idx, 0)
            new_conf = conf.copy()
            new_conf = new_conf.at[a_idxs].set(new_pos)
            new_water_ixn_nrgs = jnp.sum(self.beta * U_fn_unsummed(new_conf, box, a_idxs, b_idxs), axis=0)
            new_water_water_ixn_nrgs = jnp.sum(
                new_water_ixn_nrgs[: (self.num_waters - 1) * 3].reshape((self.num_waters - 1), 3), axis=1
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

    def propose_with_log_q_diff(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
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
        # moved_coords = randomly_translate(trial_chosen_coords, trial_translation)

        # optimized version using double transposition
        log_weights_after, trial_coords = self.batch_log_weights_incremental(
            coords, box, chosen_water, moved_coords, log_weights_before
        )

        # reference version
        # trial_coords = coords.copy()  # can optimize this later if needed
        # trial_coords[chosen_water_atoms] = moved_coords
        # log_weights_after = self.batch_log_weights(trial_coords, box)

        log_q_diff = logsumexp(log_weights_before) - logsumexp(log_weights_after)
        new_state = CoordsVelBox(trial_coords, x.velocities, x.box)

        return new_state, log_q_diff


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
    # (ytz): this might not be inside the box, but it doesnt' matter
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


class TIBDExchangeMove(BDExchangeMove):
    r"""
    Targetted Insertion and Biased Deletion Exchange Move

    Insertions are targetted over two regions V1 and V2 such that V1 is a sphere whose origin
    is at the centroid of a set of indices, and V2 = box - V1.

    Deletions are biased such that each water x_i has a weight equal to exp(u_ixn_nrg(x_i, x \ x_i)).
    """

    def __init__(
        self,
        nb_beta: float,
        nb_cutoff: float,
        nb_params: NDArray,
        water_idxs: NDArray,
        beta: float,
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

        beta: float
            Thermodynamic beta, 1/kT, in (kJ/mol)^-1

        radius: float
            Radius to use for the ligand_idxs

        """
        super().__init__(nb_beta, nb_cutoff, nb_params, water_idxs, beta)

        self.ligand_idxs = np.array(ligand_idxs)  # used to determine center of sphere
        self.radius = radius

    # def get_water_groups(self, coords, box, center):
    #     """
    #     Partition water molecules into two groups, depending if it's inside the sphere or outside the sphere.
    #     """
    #     mol_centroids = np.mean(coords[self.water_idxs_np], axis=1)
    #     dijs = np.linalg.norm(delta_r_np(mol_centroids, center, box), axis=1)
    #     inner_mols = np.argwhere(dijs < self.radius).reshape(-1)
    #     outer_mols = np.argwhere(dijs >= self.radius).reshape(-1)

    #     assert len(inner_mols) + len(outer_mols) == len(self.water_idxs_np)
    #     return inner_mols, outer_mols

    # @profile
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
        # new_coords = randomly_translate(new_coords, vj_site)

        vj_plus_one_idxs = np.concatenate([[water_idx], vj_mols])
        log_weights_after_full, trial_coords = self.batch_log_weights_incremental(
            coords, box, water_idx, new_coords, log_weights_before_full
        )
        trial_coords = np.array(trial_coords)
        log_weights_after_full = np.array(log_weights_after_full)
        log_weights_after = log_weights_after_full[vj_plus_one_idxs]

        log_p_accept = min(
            0, logsumexp(log_weights_before) - logsumexp(log_weights_after) + np.log(vol_j) - np.log(vol_i)
        )

        new_state = CoordsVelBox(trial_coords, x.velocities, x.box)

        return new_state, log_p_accept

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        box = x.box
        coords = x.coords
        center = np.mean(coords[self.ligand_idxs], axis=0)
        inner_mols, outer_mols = get_water_groups(coords, box, center, self.water_idxs_np, self.radius)
        n1 = len(inner_mols)
        n2 = len(outer_mols)

        vol_1 = (4 / 3) * np.pi * self.radius ** 3
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
    if charges is None:
        return ff

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


def test_exchange():
    parser = argparse.ArgumentParser(description="Test the exchange protocol in a box of water.")
    parser.add_argument("--water_pdb", type=str, help="Location of the water PDB", required=True)
    parser.add_argument(
        "--ligand_sdf",
        type=str,
        help="SDF file containing the ligand of interest. Disable to run bulk water.",
        required=False,
    )
    parser.add_argument("--out_cif", type=str, help="Output cif file", required=True)
    parser.add_argument(
        "--ligand_charges",
        type=str,
        help='Allowed values: "zero" or "espaloma", required if ligand_sdf is not False.',
        required=False,
    )

    parser.add_argument(
        "--md_steps_per_batch",
        type=int,
        help="Number of MD steps per batch",
        required=True,
    )

    parser.add_argument(
        "--mc_steps_per_batch",
        type=int,
        help="Number of MC steps per batch",
        required=True,
    )

    parser.add_argument(
        "--insertion_type",
        type=str,
        help='Allowed values "targeted" and "untargeted"',
        required=True,
    )

    args = parser.parse_args()

    print(" ".join(sys.argv))

    if args.ligand_sdf is not None:
        suppl = list(Chem.SDMolSupplier(args.ligand_sdf, removeHs=False))
        mol = suppl[0]
        if args.ligand_charges == "espaloma":
            esp = charges.espaloma_charges()  # charged system via espaloma
        elif args.ligand_charges == "zero":
            esp = np.zeros(mol.GetNumAtoms())  # decharged system
        else:
            assert 0, "Unknown charge model for the ligand"
    else:
        mol = None
        esp = None

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
    # all_pairs has masked charges
    if mol:
        # uses a summed potential
        nb_beta = bps[-1].potential.potentials[1].beta
        nb_cutoff = bps[-1].potential.potentials[1].cutoff
        nb_water_ligand_params = bps[-1].potential.params_init[1]
        print("number of ligand atoms", mol.GetNumAtoms())
    else:
        # does not use a summed potential
        nb_beta = bps[-1].potential.beta
        nb_cutoff = bps[-1].potential.cutoff
        nb_water_ligand_params = bps[-1].params
    print("number of water atoms", nwm * 3)
    print("water_ligand parameters", nb_water_ligand_params)
    bb_radius = 0.46

    # tibd optimized
    if args.insertion_type == "targeted":
        exc_mover = TIBDExchangeMove(
            nb_beta, nb_cutoff, nb_water_ligand_params, water_idxs, 1 / DEFAULT_KT, initial_state.ligand_idxs, bb_radius
        )
    elif args.insertion_type == "untargeted":
        # vanilla reference
        exc_mover = BDExchangeMove(nb_beta, nb_cutoff, nb_water_ligand_params, water_idxs, 1 / DEFAULT_KT)
    else:
        assert 0

    cur_box = initial_state.box0
    cur_x_t = initial_state.x0
    cur_v_t = np.zeros_like(cur_x_t)

    # debug
    seed = 2023
    if mol:
        writer = cif_writer.CIFWriter([topology, mol], args.out_cif)
    else:
        writer = cif_writer.CIFWriter([topology], args.out_cif)

    cur_x_t = image_frames(initial_state, [cur_x_t], [cur_box])[0]
    writer.write_frame(cur_x_t * 10)

    npt_mover = NPTMove(
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
    xvb_t = npt_mover.move(xvb_t)
    print("done")

    # TBD: cache the minimized and equilibrated initial structure later on to iterate faster.
    npt_mover.n_steps = args.md_steps_per_batch
    # (ytz): If I start with pure MC, and no MD, it's actually very easy to remove the waters.
    # since the starting waters have very very high energy. If I re-run MD, then it becomes progressively harder
    # remove the water since we will re-equilibriate the waters.
    for idx in range(1000000):
        density = compute_density(nwm, xvb_t.box)

        xvb_t = image_xvb(initial_state, xvb_t)

        # start_time = time.time()
        for _ in range(args.mc_steps_per_batch):
            assert np.amax(np.abs(xvb_t.coords)) < 1e3
            xvb_t = exc_mover.move(xvb_t)

        # compute occupancy at the end of MC moves (as opposed to MD moves), as its more sensitive to any possible
        # biases and/or correctness issues.
        occ = compute_occupancy(xvb_t.coords, xvb_t.box, initial_state.ligand_idxs, threshold=bb_radius)
        print(
            f"{exc_mover.n_accepted} / {exc_mover.n_proposed} | density {density} | # of waters in spherical region {occ // 3} | md step: {idx * args.md_steps_per_batch}",
            flush=True,
        )

        if idx % 10 == 0:
            writer.write_frame(xvb_t.coords * 10)

        # print("time per mc move", (time.time() - start_time) / mc_steps_per_batch)

        # run MD
        xvb_t = npt_mover.move(xvb_t)

    writer.close()


if __name__ == "__main__":
    # A trajectory is written out called water.cif
    # To visualize it, run: pymol water.cif (note that the simulation does not have to be complete to visualize progress)

    # example invocation:

    # start with 6 waters, using espaloma charge, 10k mc steps, 10k md steps, targeted insertion:
    # python timemachine/md/exchange/exchange_mover.py --water_pdb timemachine/datasets/water_exchange/bb_6_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered.sdf --ligand_charges espaloma --out_cif traj_6_waters.cif --md_steps_per_batch 10000 --mc_steps_per_batch 10000 --insertion_type targeted

    # start with 0 waters, using zero charges, 10k mc steps, 10k md steps, targeted insertion:
    # python timemachine/md/exchange/exchange_mover.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered.sdf --ligand_charges zero --out_cif traj_0_waters.cif --md_steps_per_batch 10000 --mc_steps_per_batch 10000 --insertion_type targeted

    # running in bulk, 10k mc steps, 10k md steps, untargeted insertion
    # python -u timemachine/md/exchange/exchange_mover.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --out_cif bulk.cif --md_steps_per_batch 10000 --mc_steps_per_batch 10000 --insertion_type untargeted
    test_exchange()
