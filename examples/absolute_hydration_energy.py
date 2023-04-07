import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from rdkit import Chem

from timemachine.constants import DEFAULT_FF
from timemachine.fe import pdb_writer
from timemachine.fe.absolute_hydration import run_solvent
from timemachine.fe.free_energy import MDParams, image_frames
from timemachine.fe.utils import get_mol_name, read_sdf
from timemachine.ff import Forcefield
from timemachine.parallel.client import CUDAPoolClient, FileClient
from timemachine.parallel.utils import get_gpu_count

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def write_ligand_trajectory_as_pdb(file_client, mol, res, prefix):
    # initial states all have same ligand idxs/bonds, which is what is impotant here.
    initial_state = res.final_result.initial_states[0]
    for i, (frames, boxes) in enumerate(zip(res.frames, res.boxes)):
        # Only writing out ligands, but image_frames also centers ligands
        frames = image_frames(initial_state, frames, boxes)
        out_path = f"{prefix}_{i}.pdb"
        with NamedTemporaryFile(suffix=".pdb") as temp:
            writer = pdb_writer.PDBWriter([mol], temp.name)
            for frame in frames:
                # Only write out ligand frames
                writer.write_frame(frame[initial_state.ligand_idxs] * 10)
            writer.close()
            file_client.store_stream(out_path, open(temp.name, "rb"))


def run_simulation(file_client, mol, ff, params, seed, windows):
    solvent_res, solvent_top, solvent_host_config = run_solvent(mol, ff, None, seed, params, n_windows=windows)
    dG = np.sum(solvent_res.final_result.dGs)
    dG_err = np.linalg.norm(solvent_res.final_result.dG_errs)
    mol.SetProp("AHFE dG (kJ/mol)", str(dG))
    mol.SetProp("AHFE dG err (kJ/mol)", str(dG_err))
    write_ligand_trajectory_as_pdb(file_client, mol, solvent_res, get_mol_name(mol) + "_ligand_traj")
    return mol


def read_from_args():

    parser = argparse.ArgumentParser(
        description="Estimate Absolute hydration energy free energy with local MD options."
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        help="number of frames to use for the free energy estimate",
        default=2000,
    )
    parser.add_argument(
        "--steps_per_frame",
        type=int,
        help="number of steps between frame collection",
        default=600,
    )
    parser.add_argument(
        "--local_steps",
        type=int,
        help="number of local steps collected in each frame between frame collection. Global steps is steps_per_frame - local_steps",
        default=500,
    )
    parser.add_argument(
        "--k",
        type=int,
        help="value of restraint for local MD",
        default=10000.0,
    )
    parser.add_argument(
        "--n_eq_steps",
        type=int,
        help="Number of equilibration steps",
        default=200000,
    )
    parser.add_argument(
        "--radius",
        type=int,
        help="Radius of selection in nm",
        default=0.5,
    )
    parser.add_argument(
        "--n_windows",
        type=int,
        help="Number of windows to run n_windows",
        default=16,
    )
    parser.add_argument("--ligands", type=str, help="SDF file containing the ligands of interest", required=True)
    parser.add_argument("--forcefield", type=str, help="location of the ligand forcefield", default=DEFAULT_FF)
    parser.add_argument("--seed", type=int, help="Random number seed", default=2023)
    parser.add_argument("--mol_names", type=str, nargs="+", help="Names of mols to run", default=None)
    parser.add_argument("--output_path", help="Path to write out SDF of mols", type=str, required=True)

    args = parser.parse_args()

    assert (
        args.local_steps <= args.steps_per_frame
    ), "Must have less than or equal number of local steps to steps between frames"

    mols = read_sdf(str(Path(args.ligands).resolve()))

    forcefield = Forcefield.load_from_file(args.forcefield)

    params = MDParams(
        n_frames=args.n_frames,
        steps_per_frame=args.steps_per_frame,
        n_eq_steps=args.n_eq_steps,
        local_steps=args.local_steps,
        k=args.k,
        radius=args.radius,
    )
    cli = CUDAPoolClient(get_gpu_count())
    file_client = FileClient()
    futures = {}
    for mol in mols:
        mol_name = get_mol_name(mol)
        if args.mol_names is not None and mol_name not in args.mol_names:
            continue
        if mol_name in futures:
            print(f"Found duplicate name '{mol_name}', skipping...")
            continue
        fut = cli.submit(run_simulation, file_client, mol, forcefield, params, args.seed, args.n_windows)
        futures[mol_name] = (mol, fut)

    with Chem.SDWriter(args.output_path) as writer:
        for name, (mol, fut) in futures.items():
            mol = fut.result()
            writer.write(mol)


if __name__ == "__main__":
    read_from_args()
