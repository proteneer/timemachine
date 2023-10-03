# Water sampling script that tests that we can use an instantaneous mover to insert/delete
# waters from a buckyball

# enable for 2x slow down
# from jax import config
# config.update("jax_enable_x64", True)

import argparse
import os
import sys

import numpy as np
from openmm import app
from rdkit import Chem

from timemachine.constants import AVOGADRO, DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe import cif_writer
from timemachine.fe.free_energy import AbsoluteFreeEnergy, HostConfig, InitialState, image_frames
from timemachine.fe.topology import BaseTopology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.ff.handlers.nonbonded import PrecomputedChargeHandler
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md.barostat.moves import NPTMove
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.builders import strip_units
from timemachine.md.exchange import exchange_mover
from timemachine.md.states import CoordsVelBox

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
        diff = exchange_mover.delta_r_np(ligand_centroid, rj, box_t)
        dij = np.linalg.norm(diff)
        if dij < threshold:
            count += 1
    return count


def setup_forcefield():
    # use a precomputed charge handler on the ligand to avoid running AM1 on a buckyball
    ff = Forcefield.load_default()
    # if charges is None:
    #     return ff

    q_handle = PrecomputedChargeHandler()
    q_handle_intra = PrecomputedChargeHandler()
    q_handle_solv = PrecomputedChargeHandler()
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
    else:
        mol = None

    ff = setup_forcefield()
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
        exc_mover = exchange_mover.TIBDExchangeMove(
            nb_beta, nb_cutoff, nb_water_ligand_params, water_idxs, DEFAULT_TEMP, initial_state.ligand_idxs, bb_radius
        )
    elif args.insertion_type == "untargeted":
        # vanilla reference
        exc_mover = exchange_mover.BDExchangeMove(nb_beta, nb_cutoff, nb_water_ligand_params, water_idxs, DEFAULT_TEMP)
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
    # python examples/water_sampling.py --water_pdb timemachine/datasets/water_exchange/bb_6_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered_espaloma.sdf --out_cif traj_6_waters.cif --md_steps_per_batch 10000 --mc_steps_per_batch 10000 --insertion_type targeted

    # start with 0 waters, using zero charges, 10k mc steps, 10k md steps, targeted insertion:
    # python examples/water_sampling.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centere_neutral.sdf --out_cif traj_0_waters.cif --md_steps_per_batch 10000 --mc_steps_per_batch 10000 --insertion_type targeted

    # running in bulk, 10k mc steps, 10k md steps, untargeted insertion
    # python -u examples/water_sampling.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --out_cif bulk.cif --md_steps_per_batch 10000 --mc_steps_per_batch 10000 --insertion_type untargeted
    test_exchange()
