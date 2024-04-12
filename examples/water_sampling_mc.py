# Water sampling script that tests that we can use an instantaneous monte carlo
# mover to insert/delete waters from a buckyball

import argparse
import sys

# enable for 2x slow down
# from jax import config
# config.update("jax_enable_x64", True)
import time

import numpy as np
from rdkit import Chem
from water_sampling_common import DEFAULT_BB_RADIUS, compute_density, compute_occupancy, get_initial_state

from timemachine.constants import DEFAULT_TEMP
from timemachine.fe import cif_writer
from timemachine.fe.free_energy import image_frames
from timemachine.ff import Forcefield
from timemachine.lib import custom_ops
from timemachine.md.barostat.moves import NPTMove
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import BDExchangeMove as RefBDExchangeMove
from timemachine.md.exchange.exchange_mover import TIBDExchangeMove as RefTIBDExchangeMove
from timemachine.md.exchange.exchange_mover import get_water_idxs
from timemachine.md.moves import MonteCarloMove
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import HarmonicBond


def image_xvb(initial_state, xvb_t):
    new_coords = xvb_t.coords
    # Only image if there is a ligand
    if len(initial_state.ligand_idxs) > 0:
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
        choices=["targeted", "untargeted"],
    )
    parser.add_argument(
        "--use_hmr",
        type=int,
        help="Whether or not we apply HMR. 1 for yes, 0 for no.",
        required=True,
    )
    parser.add_argument("--iterations", type=int, help="Number of iterations", default=1000000)
    parser.add_argument("--equilibration_steps", type=int, help="Number of equilibration steps", default=50000)
    parser.add_argument("--seed", default=2024, type=int, help="Random seed")
    parser.add_argument("--use_reference", action="store_true", help="Use the reference ExchangeMoves")
    parser.add_argument(
        "--save_last_frame", type=str, help="Store last frame as a npz file, used to verify bitwise determinism"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to generate proposals for the MC moves, not used for reference case",
    )

    args = parser.parse_args()

    print(" ".join(sys.argv))

    if args.ligand_sdf is not None:
        suppl = list(Chem.SDMolSupplier(args.ligand_sdf, removeHs=False))
        mol = suppl[0]
    else:
        mol = None

    ff = Forcefield.load_precomputed_default()
    seed = args.seed
    np.random.seed(seed)

    nb_cutoff = 1.2  # this has to be 1.2 since the builders hard code this in (should fix later)
    # nit: use lamb=0.0 to get the fully-interacting end-state
    initial_state, nwm, topology = get_initial_state(args.water_pdb, mol, ff, seed, nb_cutoff, args.use_hmr, lamb=0.0)
    # set up water indices, assumes that waters are placed at the front of the coordinates.
    bps = initial_state.potentials
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential
    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, initial_state.x0.shape[0])
    water_idxs = get_water_idxs(all_group_idxs, initial_state.ligand_idxs)

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

    def run_mc_proposals(mover, state: CoordsVelBox, steps: int) -> CoordsVelBox:
        if isinstance(mover, MonteCarloMove):
            for _ in range(steps):
                state = mover.move(state)
            return state
        else:
            # C++ implementations have a different API
            x_mv, _ = exc_mover.move(xvb_t.coords, xvb_t.box)
            return CoordsVelBox(x_mv, xvb_t.velocities, xvb_t.box)

    # Interval is always 1 as we want this to be called every time we call `move`
    exchange_interval = 1
    # tibd optimized
    if args.insertion_type == "targeted":
        assert mol is not None, "Requires a mol for targeted exchange"
        if args.use_reference:
            exc_mover = RefTIBDExchangeMove(
                nb_beta,
                nb_cutoff,
                nb_water_ligand_params,
                water_idxs,
                DEFAULT_TEMP,
                initial_state.ligand_idxs,
                DEFAULT_BB_RADIUS,
            )
        else:
            exc_mover = custom_ops.TIBDExchangeMove_f32(
                initial_state.x0.shape[0],
                initial_state.ligand_idxs,
                water_idxs,
                nb_water_ligand_params,
                DEFAULT_TEMP,
                nb_beta,
                nb_cutoff,
                DEFAULT_BB_RADIUS,
                seed,
                args.mc_steps_per_batch,
                exchange_interval,
                batch_size=args.batch_size,
            )
    elif args.insertion_type == "untargeted":
        if args.use_reference:
            exc_mover = RefBDExchangeMove(nb_beta, nb_cutoff, nb_water_ligand_params, water_idxs, DEFAULT_TEMP)
        else:
            exc_mover = custom_ops.BDExchangeMove_f32(
                initial_state.x0.shape[0],
                water_idxs,
                nb_water_ligand_params,
                DEFAULT_TEMP,
                nb_beta,
                nb_cutoff,
                seed,
                args.mc_steps_per_batch,
                exchange_interval,
                batch_size=args.batch_size,
            )

    cur_box = initial_state.box0
    cur_x_t = initial_state.x0
    cur_v_t = np.zeros_like(cur_x_t)

    # debug
    seed = 2023
    if mol:
        writer = cif_writer.CIFWriter([topology, mol], args.out_cif)
    else:
        writer = cif_writer.CIFWriter([topology], args.out_cif)

    xvb_t = CoordsVelBox(cur_x_t, cur_v_t, cur_box)
    xvb_t = image_xvb(initial_state, xvb_t)
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

    equilibration_steps = args.equilibration_steps
    # equilibrate using the npt mover
    npt_mover.n_steps = equilibration_steps
    if equilibration_steps > 0:
        xvb_t = npt_mover.move(xvb_t)
    print("done")

    # TBD: cache the minimized and equilibrated initial structure later on to iterate faster.
    npt_mover.n_steps = args.md_steps_per_batch
    # (ytz): If I start with pure MC, and no MD, it's actually very easy to remove the waters.
    # since the starting waters have very very high energy. If I re-run MD, then it becomes progressively harder
    # remove the water since we will re-equilibrate the waters.
    for idx in range(args.iterations):
        density = compute_density(nwm, xvb_t.box)

        xvb_t = image_xvb(initial_state, xvb_t)
        assert np.amax(np.abs(xvb_t.coords)) < 1e3

        start_time = time.perf_counter_ns()
        xvb_t = run_mc_proposals(exc_mover, xvb_t, args.mc_steps_per_batch)
        end_time = time.perf_counter_ns()

        occ = 0
        # (fey) Don't compute occupancy if there is no ligand
        if len(initial_state.ligand_idxs) > 0:
            occ = compute_occupancy(xvb_t.coords, xvb_t.box, initial_state.ligand_idxs, threshold=DEFAULT_BB_RADIUS)
            # compute occupancy at the end of MC moves (as opposed to MD moves), as its more sensitive to any possible
            # biases and/or correctness issues.
        if isinstance(exc_mover, MonteCarloMove):
            accepted = exc_mover.n_accepted
            proposed = exc_mover.n_proposed
        else:
            accepted = exc_mover.n_accepted()
            proposed = exc_mover.n_proposed()
        print(
            f"{accepted} / {proposed} | density {density} | # of waters in spherical region {occ // 3} | md step: {idx * args.md_steps_per_batch} | time per mc move: {(end_time - start_time) / args.mc_steps_per_batch }ns",
            flush=True,
        )

        if idx % 10 == 0:
            writer.write_frame(xvb_t.coords * 10)

        # run MD
        if args.md_steps_per_batch > 0:
            xvb_t = npt_mover.move(xvb_t)

    writer.close()
    if args.save_last_frame:
        np.savez(
            args.save_last_frame,
            coords=xvb_t.coords,
            velocities=xvb_t.velocities,
            box=xvb_t.box,
            iterations=args.iterations,
        )


if __name__ == "__main__":
    # A trajectory is written out called water.cif
    # To visualize it, run: pymol water.cif (note that the simulation does not have to be complete to visualize progress)

    # example invocation:

    # start with 0 waters, with hmr, using espaloma charges, 10k mc steps, 10k md steps, targeted insertion:
    # python -u examples/water_sampling_mc.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered_espaloma.sdf --out_cif traj_0_waters.cif --md_steps_per_batch 10000 --mc_steps_per_batch 10000 --insertion_type targeted --use_hmr 1

    # start with 6 waters, with hmr, using espaloma charges, 10k mc steps, 10k md steps, targeted insertion:
    # python -u examples/water_sampling_mc.py --water_pdb timemachine/datasets/water_exchange/bb_6_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered_espaloma.sdf --out_cif traj_6_waters.cif --md_steps_per_batch 10000 --mc_steps_per_batch 10000 --insertion_type targeted --use_hmr 1

    # start with 0 waters, with hmr, using zero charges, 10k mc steps, 10k md steps, targeted insertion:
    # python -u examples/water_sampling_mc.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered_neutral.sdf --out_cif traj_0_waters.cif --md_steps_per_batch 10000 --mc_steps_per_batch 10000 --insertion_type targeted --use_hmr 1

    # running in bulk, 10k mc steps, 10k md steps, untargeted insertion
    # python -u examples/water_sampling_mc.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --out_cif bulk.cif --md_steps_per_batch 10000 --mc_steps_per_batch 10000 --insertion_type untargeted --use_hmr 1
    test_exchange()
