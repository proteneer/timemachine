import argparse
import csv
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, SpooledTemporaryFile

import numpy as np
from rdkit import Chem

from timemachine.constants import DEFAULT_FF
from timemachine.fe import atom_mapping, pdb_writer, single_topology
from timemachine.fe.free_energy import HostConfig, MDParams, image_frames
from timemachine.fe.rbfe import estimate_relative_free_energy_via_greedy_bisection
from timemachine.fe.utils import get_mol_name, read_sdf
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.parallel.client import CUDAPoolClient, FileClient
from timemachine.parallel.utils import get_gpu_count

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def write_trajectory_as_pdb(file_client, mol_a, mol_b, core, res, host_topology, prefix):

    atom_map_mixin = single_topology.AtomMapMixin(mol_a, mol_b, core)
    initial_state = res.final_result.initial_states[0]
    for i, (frames, boxes) in enumerate(zip(res.frames, res.boxes)):
        frames = image_frames(initial_state, frames, boxes)
        out_path = f"{prefix}_{i}.pdb"
        with NamedTemporaryFile(suffix=".pdb") as temp:
            writer = pdb_writer.PDBWriter([host_topology, mol_a, mol_b], temp.name)
            for frame in frames:
                host_frame = frame[: host_topology.getNumAtoms()]
                ligand_frame = frame[host_topology.getNumAtoms() :]
                mol_ab_frame = pdb_writer.convert_single_topology_mols(ligand_frame, atom_map_mixin)
                writer.write_frame(np.concatenate([host_frame, mol_ab_frame]) * 10)
            writer.close()
            file_client.store_stream(out_path, open(temp.name, "rb"))


def store_overlap(file_client, res, prefix):
    output = SpooledTemporaryFile()
    np.savez(
        output,
        lambdas=np.array([s.lamb for s in res.final_result.initial_states]),
        component_names=np.array([type(p.potential).__name__ for p in res.final_result.initial_states[0].potentials]),
        overlap_by_component_by_lambda=res.final_result.overlap_by_component_by_lambda,
    )
    output.flush()
    output.seek(0)
    file_client.store_stream(prefix + "overlaps.npz", output)


def store_ukln(file_client, res, prefix):
    output = SpooledTemporaryFile()
    np.savez(
        output,
        lambdas=np.array([s.lamb for s in res.final_result.initial_states]),
        component_names=np.array([type(p.potential).__name__ for p in res.final_result.initial_states[0].potentials]),
        u_kln_by_component_by_lambda=res.final_result.u_kln_by_component_by_lambda,
    )
    output.flush()
    output.seek(0)
    file_client.store_stream(prefix + "ukln.npz", output)


def store_mol(file_client, mol, prefix):
    with NamedTemporaryFile(suffix=".sdf") as temp:
        with Chem.SDWriter(temp.name) as writer:
            writer.write(mol)
        mol_name = get_mol_name(mol)
        file_client.store_stream(prefix + f"{mol_name}.sdf", open(temp.name, "rb"))


def store_summary(file_client, solvent_res, complex_res, prefix):
    solvent_dg = np.sum(solvent_res.final_result.dGs)
    solvent_err = np.linalg.norm(solvent_res.final_result.dG_errs)
    complex_dg = np.sum(complex_res.final_result.dGs)
    complex_err = np.linalg.norm(complex_res.final_result.dG_errs)

    ddg = complex_dg - solvent_dg
    ddg_err = np.linalg.norm([complex_err, solvent_err])
    output = SpooledTemporaryFile()
    np.savez(
        output,
        solvent_dg=np.array([solvent_dg]),
        complex_dg=np.array([complex_dg]),
        solvent_dg_err=np.array([solvent_err]),
        complex_dg_err=np.array([complex_err]),
        ddg=np.array([ddg]),
        ddg_err=np.array([ddg_err]),
    )
    output.flush()
    output.seek(0)
    file_client.store_stream(os.path.join(prefix, "summary.npz"), output)

    return ddg, ddg_err


def run_simulation(file_client, mol_a, mol_b, forcefield, params, seed, windows):
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.12,
        chain_cutoff=0.2,
        max_visits=1e7,
        connected_core=True,
        max_cores=1e6,
        enforce_core_core=True,
        complete_rings=True,
        enforce_chiral=True,
        min_threshold=0,
    )
    core = all_cores[0]

    prefix = os.path.join(get_mol_name(mol_a) + "_" + get_mol_name(mol_b)) + "/"
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(box_width, forcefield.water_ff)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box)

    solvent_res = estimate_relative_free_energy_via_greedy_bisection(
        mol_a, mol_b, core, forcefield, solvent_host_config, seed, n_windows=windows, md_params=params, prefix="solvent"
    )
    store_ukln(file_client, solvent_res, os.path.join(prefix, "solvent_"))

    file_client.store(os.path.join(prefix, "solvent_overlap.png"), solvent_res.plots.overlap_detail_png)

    # this st is only needed to deal with visualization jank
    write_trajectory_as_pdb(
        file_client, mol_a, mol_b, core, solvent_res, solvent_top, os.path.join(prefix, "solvent_traj")
    )

    with NamedTemporaryFile(suffix=".pdb") as temp:
        with open(temp.name, "wb") as ofs:
            temp.write(file_client.load("structure.pdb"))
        complex_sys, complex_conf, complex_box, complex_top = builders.build_protein_system(
            temp.name, forcefield.protein_ff, forcefield.water_ff
        )
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box)
    complex_res = estimate_relative_free_energy_via_greedy_bisection(
        mol_a,
        mol_b,
        core,
        forcefield,
        complex_host_config,
        seed + 1,
        n_windows=windows,
        md_params=params,
        prefix="complex",
    )
    file_client.store(os.path.join(prefix, "complex_overlap.png"), complex_res.plots.overlap_detail_png)
    store_ukln(file_client, complex_res, os.path.join(prefix, "complex_"))
    write_trajectory_as_pdb(
        file_client, mol_a, mol_b, core, complex_res, complex_top, os.path.join(prefix, "complex_traj")
    )

    return store_summary(file_client, solvent_res, complex_res, prefix)


def main():

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
        default=400,
    )
    parser.add_argument(
        "--local_steps",
        type=int,
        help="number of local steps collected in each frame between frame collection. Global steps is steps_per_frame - local_steps",
        default=300,
    )
    parser.add_argument(
        "--k",
        type=int,
        help="value of restraint for local MD",
        default=1000.0,
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
        default=1.0,
    )
    parser.add_argument(
        "--n_windows",
        type=int,
        help="Number of windows to run n_windows",
        default=48,
    )
    parser.add_argument("--protein", type=str, help="PDB of the protein complex", required=True)
    parser.add_argument("--ligands", type=str, help="SDF file containing the ligands of interest", required=True)
    parser.add_argument("--edges", type=str, help="Path to edges", required=True)
    parser.add_argument("--forcefield", type=str, help="location of the ligand forcefield", default=DEFAULT_FF)
    parser.add_argument("--seed", type=int, help="Random number seed", default=2023)
    parser.add_argument("--mol_names", type=str, nargs="+", help="Names of mols to run", default=None)
    parser.add_argument("--output_path", help="Path to write out SDF of mols", type=str, required=True)
    parser.add_argument("--limit", default=None, type=int)
    args = parser.parse_args()

    assert (
        args.local_steps <= args.steps_per_frame
    ), "Must have less than or equal number of local steps to steps between frames"

    mols = read_sdf(str(Path(args.ligands).resolve()))
    mol_name_to_mol = {get_mol_name(mol): mol for mol in mols}

    edges = json.load(open(Path(args.edges).resolve()))

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
    # Store the structure so the file client can access it
    file_client.store("structure.pdb", Path(args.protein).resolve().read_bytes())
    futures = {}
    for (name_a, name_b) in edges:
        mol_a = mol_name_to_mol[name_a]
        mol_b = mol_name_to_mol[name_b]
        fut = cli.submit(run_simulation, file_client, mol_a, mol_b, forcefield, params, args.seed, args.n_windows)
        futures[(name_a, name_b)] = fut
        if args.limit is not None and len(futures) >= args.limit:
            break

    with open(args.output_path, "w", newline="") as ofs:
        writer = csv.writer(ofs)
        writer.writerow(["mol_a", "mol_b", "ddG", "ddG_err"])
        for (name_a, name_b), fut in futures.items():
            ddg, ddg_err = fut.result()
            writer.writerow([name_a, name_b, ddg, ddg_err])
            print(name_a, "->", name_b, "ddG", ddg, ddg_err)


if __name__ == "__main__":
    main()
