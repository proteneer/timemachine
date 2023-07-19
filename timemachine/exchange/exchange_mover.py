# Prototype ExchangeMover that implements an instantaneous water swap move.
# disable for ~2x speed-up
# from jax import config
# config.update("jax_enable_x64", True)

import argparse
import os
from typing import Tuple

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

    # host_ff = app.ForceField("/home/yutong/Code/timemachine/ff/params/tip3p_modified.xml")
    host_ff = app.ForceField(f"{water_ff}.xml")
    nwa = len(host_coords)
    solvated_host_system = host_ff.createSystem(
        host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    solvated_host_coords = host_coords
    solvated_topology = host_pdb.topology

    return solvated_host_system, solvated_host_coords, box, solvated_topology, nwa


# currently un-used.
def randomly_rotate_and_translate(coords, offset):
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


class ExchangeMove(moves.MonteCarloMove):
    def __init__(self, nb_params, nb_beta, nb_cutoff, water_idxs):
        self.nb_beta = nb_beta
        self.nb_cutoff = nb_cutoff
        self.nb_params = jnp.array(nb_params)
        self.num_waters = len(water_idxs)
        self.water_idxs = water_idxs
        self.beta = 1 / DEFAULT_KT

        @jax.jit
        def U_fn(conf, box, a_idxs, b_idxs):
            # compute the energy of an interaction group
            conf_i = conf[a_idxs]
            conf_j = conf[b_idxs]
            params_i = self.nb_params[a_idxs]
            params_j = self.nb_params[b_idxs]
            return nonbonded.nonbonded_block(conf_i, conf_j, box, params_i, params_j, self.nb_beta, self.nb_cutoff)

        self.U_fn = U_fn

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        coords = x.coords
        box = x.box
        n_atoms = len(coords)
        chosen_water = np.random.randint(self.num_waters)
        chosen_water_atoms = self.water_idxs[chosen_water]

        # compute delta_U of deletion
        a_idxs = chosen_water_atoms
        b_idxs = np.delete(np.arange(n_atoms), a_idxs)

        # compute delta_U of insertion
        trial_chosen_coords = coords[chosen_water_atoms]
        trial_translation = np.diag(box) * np.random.rand(3)
        # tbd - what should we do  with velocities?

        # this has a higher acceptance probability than if we allowed for rotations
        # (probably because we have a much higher chance of a useless move)
        moved_coords = randomly_translate(trial_chosen_coords, trial_translation)
        trial_coords = coords.copy()  # can optimize this later if needed
        trial_coords[chosen_water_atoms] = moved_coords

        delta_U_insert = self.U_fn(trial_coords, box, a_idxs, b_idxs)
        # If our system is in a clash free state, a deletion move has delta_Us typically
        # on the order of 35kTs. However, if we start in a clashy state, we can no longer
        # guarantee this. So it's safer to disable this micro-optimization for now.
        delta_U_delete = -self.U_fn(coords, box, a_idxs, b_idxs)
        delta_U_total = delta_U_delete + delta_U_insert

        # convert to inf if we get a nan
        if np.isnan(delta_U_total):
            delta_U_total = np.inf

        log_p_accept = min(0, -self.beta * delta_U_total)
        new_state = CoordsVelBox(trial_coords, x.velocities, x.box)

        return new_state, log_p_accept


def compute_density(n_waters, box):
    box_vol = np.product(np.diag(box))
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
        diff = ligand_centroid - rj  # this can be either N,N,3 or B,3
        box_diag = np.diag(box_t)
        diff -= box_diag * np.floor(diff / box_diag + 0.5)
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
    nb_ligand_water_params = bps[-1].potential.params_init[1]

    print("number of water atoms", nwm * 3, "number of ligand atoms", mol.GetNumAtoms())
    print("ligand_water parameters", nb_ligand_water_params)

    exc_mover = ExchangeMove(nb_ligand_water_params, nb_beta, nb_cutoff, water_idxs)
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
    xvb_t = npt_mover.move(xvb_t)
    print("done")
    md_steps_per_batch = 2000
    npt_mover.n_steps = md_steps_per_batch
    # TBD: cache the minimized and equilibrated initial structure later on to iterate faster.

    xvb_t = image_xvb(initial_state, xvb_t)

    for idx in range(100000):
        density = compute_density(nwm, xvb_t.box)

        xvb_t = image_xvb(initial_state, xvb_t)
        occ = compute_occupancy(xvb_t.coords, xvb_t.box, initial_state.ligand_idxs, threshold=0.46)

        print(
            f"{exc_mover.n_accepted} / {exc_mover.n_proposed} | density {density} | # of waters in bb {occ // 3} | md step: {idx * md_steps_per_batch}"
        )
        if idx % 10 == 0:
            writer.write_frame(xvb_t.coords * 10)

        # run MC
        for _ in range(5000):
            assert np.amax(np.abs(xvb_t.coords)) < 1e3
            xvb_t = exc_mover.move(xvb_t)

        # run MD
        xvb_t = npt_mover.move(xvb_t)


if __name__ == "__main__":
    # A trajectory is written out called water.cif
    # To visualize it, run: pymol water.cif (note that the simulation does not have to be complete to visualize progress)

    # example invocation:

    # start with 6 waters, using espaloma charges:
    # python timemachine/exchange/exchange_mover.py --water_pdb timemachine/datasets/water_exchange/bb_6_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered.sdf --ligand_charges espaloma --out_cif traj_6_waters.cif

    # start with 0 waters, using zero charges:
    # python timemachine/exchange/exchange_mover.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered.sdf --ligand_charges zero --out_cif traj_0_waters.cif
    test_exchange()
