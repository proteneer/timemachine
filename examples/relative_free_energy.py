import argparse
import sys

import numpy as np
from rdkit import Chem

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping, cif_writer
from timemachine.fe.free_energy import HREXParams, MDParams, WaterSamplingParams
from timemachine.fe.rbfe import run_complex, run_solvent
from timemachine.fe.single_topology import AtomMapMixin
from timemachine.fe.utils import plot_atom_mapping_grid, read_sdf
from timemachine.ff import Forcefield
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def write_trajectory_as_cif(mol_a, mol_b, core, all_frames, host_topology, prefix):
    atom_map_mixin = AtomMapMixin(mol_a, mol_b, core)
    for window_idx, window_frames in enumerate(all_frames):
        out_path = f"{prefix}_{window_idx}.cif"
        writer = cif_writer.CIFWriter([host_topology, mol_a, mol_b], out_path)
        for frame in window_frames:
            host_frame = frame[: host_topology.getNumAtoms()]
            ligand_frame = frame[host_topology.getNumAtoms() :]
            mol_ab_frame = cif_writer.convert_single_topology_mols(ligand_frame, atom_map_mixin)
            writer.write_frame(np.concatenate([host_frame, mol_ab_frame]) * 10)
        writer.close()


def run_pair(mol_a, mol_b, core, forcefield, md_params, protein_path):
    solvent_res, solvent_top, solvent_host_config = run_solvent(
        mol_a, mol_b, core, forcefield, None, md_params=md_params
    )

    with open("solvent_overlap.png", "wb") as fh:
        fh.write(solvent_res.plots.overlap_detail_png)

    # this st is only needed to deal with visualization jank
    write_trajectory_as_cif(mol_a, mol_b, core, solvent_res.frames, solvent_top, "solvent_traj")

    print(
        f"solvent dG: {np.sum(solvent_res.final_result.dGs):.3f} +- {np.linalg.norm(solvent_res.final_result.dG_errs):.3f} kJ/mol"
    )

    complex_res, complex_top, complex_host_config = run_complex(
        mol_a, mol_b, core, forcefield, protein_path, md_params=md_params
    )
    with open("complex_overlap.png", "wb") as fh:
        fh.write(complex_res.plots.overlap_detail_png)
    write_trajectory_as_cif(mol_a, mol_b, core, complex_res.frames, complex_top, "complex_traj")

    print(
        f"complex dG: {np.sum(complex_res.final_result.dGs):.3f} +- {np.linalg.norm(complex_res.final_result.dG_errs):.3f} kJ/mol"
    )


def hif2a_pair():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    protein_path = "timemachine/testsystems/data/hif2a_nowater_min.pdb"

    md_params = MDParams(n_frames=100, n_eq_steps=200_000, steps_per_frame=400, seed=2023)
    # fast
    run_pair(mol_a, mol_b, core, forcefield, md_params, protein_path=protein_path)


def get_mol_by_name(mols, name):
    for m in mols:
        if m.GetProp("_Name") == name:
            return m

    assert 0, "Mol not found"


def read_from_args():
    parser = argparse.ArgumentParser(
        description="Estimate relative free energy difference between complex and solvent given two ligands mol_a and mol_b."
    )
    parser.add_argument(
        "--n_frames", type=int, help="number of frames to use for the free energy estimate", required=True
    )
    parser.add_argument("--ligands", type=str, help="SDF file containing the ligands of interest", required=True)
    parser.add_argument("--mol_a_name", type=str, help="name of the start molecule", required=True)
    parser.add_argument("--mol_b_name", type=str, help="name of the end molecule", required=True)
    parser.add_argument("--forcefield", type=str, help="location of the ligand forcefield", required=True)
    parser.add_argument("--protein", type=str, help="PDB of the protein complex", required=True)
    parser.add_argument("--seed", type=int, help="Random number seed", required=True)

    args = parser.parse_args()
    mols = read_sdf(str(args.ligands))
    mol_a = get_mol_by_name(mols, args.mol_a_name)  # 43 in test pair
    mol_b = get_mol_by_name(mols, args.mol_b_name)  # 30 in test pair

    print("Searching for the maximum common substructure...")
    print("mol_a SMILES:", Chem.MolToSmiles(mol_a, isomericSmiles=False))
    print("mol_b SMILES:", Chem.MolToSmiles(mol_b, isomericSmiles=False))

    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )

    core = all_cores[0]
    res = plot_atom_mapping_grid(mol_a, mol_b, core)
    fpath = f"atom_mapping_{args.mol_a_name}_to_{args.mol_b_name}.svg"
    print("core mapping written to", fpath)
    with open(fpath, "w") as fh:
        fh.write(res)

    forcefield = Forcefield.load_from_file(args.forcefield)

    md_params = MDParams(
        n_frames=args.n_frames,
        n_eq_steps=200_000,
        steps_per_frame=400,
        seed=args.seed,
        hrex_params=HREXParams(),
        water_sampling_params=WaterSamplingParams(),
    )

    run_pair(mol_a, mol_b, core, forcefield, md_params, args.protein)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        hif2a_pair()
    else:
        read_from_args()
