import argparse
from pathlib import Path

from timemachine.constants import DEFAULT_FF
from timemachine.fe import pdb_writer
from timemachine.fe.absolute_hydration import run_solvent
from timemachine.fe.free_energy import MDParams
from timemachine.fe.utils import get_mol_name, read_sdf
from timemachine.ff import Forcefield


def write_trajectory_as_pdb(mol, core, all_frames, host_topology, prefix):
    for window_idx, window_frames in enumerate(all_frames):
        out_path = f"{prefix}_{window_idx}.pdb"
        writer = pdb_writer.PDBWriter([host_topology, mol], out_path)
        for frame in window_frames:
            writer.write_frame(frame * 10)
        writer.close()


def read_from_args():

    parser = argparse.ArgumentParser(description="Estimate Absolute hydration energy free energy")
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
        default=1000,
    )
    parser.add_argument(
        "--local_steps",
        type=int,
        help="number of local steps collected in each frame between frame collection",
        default=1000,
    )
    parser.add_argument(
        "--k",
        type=int,
        help="value of restraint for local MD",
        default=10000.0,
    )
    parser.add_argument(
        "--radius",
        type=int,
        help="Radius of selection in nm",
        default=0.5,
    )
    parser.add_argument("--ligands", type=str, help="SDF file containing the ligands of interest", required=True)
    parser.add_argument("--forcefield", type=str, help="location of the ligand forcefield", default=DEFAULT_FF)
    parser.add_argument("--seed", type=int, help="Random number seed", default=2023)
    parser.add_argument("--mol_names", type=str, nargs="+", help="Names of mols to run", default=None)

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
    for mol in mols:
        if args.mol_names is not None and get_mol_name(mol) not in args.mol_names:
            continue
        run_solvent(mol, forcefield, args.protein, args.seed, params)


if __name__ == "__main__":
    read_from_args()
