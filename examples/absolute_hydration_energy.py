import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem

from timemachine.constants import DEFAULT_FF
from timemachine.fe import pdb_writer
from timemachine.fe.absolute_hydration import run_solvent
from timemachine.fe.free_energy import MDParams
from timemachine.fe.utils import get_mol_name, read_sdf
from timemachine.ff import Forcefield
from timemachine.parallel.client import CUDAPoolClient
from timemachine.parallel.utils import get_gpu_count

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def write_trajectory_as_pdb(mol, all_frames, host_topology, prefix):
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
    futures = {}
    for mol in mols:
        mol_name = get_mol_name(mol)
        if args.mol_names is not None and mol_name not in args.mol_names:
            continue
        if mol_name in futures:
            print(f"Found duplicate name '{mol_name}', skipping...")
            continue
        fut = cli.submit(run_solvent, mol, forcefield, None, args.seed, params, n_windows=args.n_windows)
        futures[mol_name] = (mol, fut)

    with Chem.SDWriter(args.output_path) as writer:
        for name, (mol, fut) in futures.items():
            solvent_res, solvent_top, solvent_host_config = fut.result()
            dG = np.sum(solvent_res.final_result.dGs)
            dG_err = np.linalg.norm(solvent_res.final_result.dG_errs)
            mol.SetProp("AHFE dG (kJ/mol)", str(dG))
            mol.SetProp("AHFE dG err (kJ/mol)", str(dG_err))
            writer.write(mol)


if __name__ == "__main__":
    read_from_args()
