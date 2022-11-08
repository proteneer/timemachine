import argparse
import sys

import numpy as np
from rdkit import Chem

from timemachine.constants import DEFAULT_FF
from timemachine.fe import atom_mapping, pdb_writer
from timemachine.fe.rbfe import HostConfig, estimate_relative_free_energy
from timemachine.fe.single_topology import AtomMapMixin
from timemachine.fe.utils import plot_atom_mapping_grid, read_sdf
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def write_trajectory_as_pdb(mol_a, mol_b, core, all_frames, host_topology, prefix):

    atom_map_mixin = AtomMapMixin(mol_a, mol_b, core)
    for window_idx, window_frames in enumerate(all_frames):
        out_path = f"{prefix}_{window_idx}.pdb"
        writer = pdb_writer.PDBWriter([host_topology, mol_a, mol_b], out_path)
        for frame in window_frames:
            host_frame = frame[: host_topology.getNumAtoms()]
            ligand_frame = frame[host_topology.getNumAtoms() :]
            mol_ab_frame = pdb_writer.convert_single_topology_mols(ligand_frame, atom_map_mixin)
            writer.write_frame(np.concatenate([host_frame, mol_ab_frame]) * 10)
        writer.close()


def run_pair(mol_a, mol_b, core, forcefield, n_frames, protein_path, seed):

    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(box_width, forcefield.water_ff)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box)

    solvent_res = estimate_relative_free_energy(
        mol_a, mol_b, core, forcefield, solvent_host_config, seed, n_frames=n_frames, prefix="solvent"
    )

    with open("solvent_overlap.png", "wb") as fh:
        fh.write(solvent_res.overlap_detail_png)

    # this st is only needed to deal with visualization jank
    write_trajectory_as_pdb(mol_a, mol_b, core, solvent_res.frames, solvent_top, "solvent_traj")

    print(f"solvent dG: {np.sum(solvent_res.all_dGs):.3f} +- {np.linalg.norm(solvent_res.all_errs):.3f} kJ/mol")

    complex_sys, complex_conf, _, _, complex_box, complex_top = builders.build_protein_system(
        protein_path, forcefield.protein_ff, forcefield.water_ff
    )
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box)
    complex_res = estimate_relative_free_energy(
        mol_a, mol_b, core, forcefield, complex_host_config, seed + 1, n_frames=n_frames, prefix="complex"
    )
    with open("complex_overlap.png", "wb") as fh:
        fh.write(complex_res.overlap_detail_png)
    write_trajectory_as_pdb(mol_a, mol_b, core, complex_res.frames, complex_top, "complex_traj")

    print(f"complex dG: {np.sum(solvent_res.all_dGs):.3f} +- {np.linalg.norm(solvent_res.all_errs):.3f} kJ/mol")


def hif2a_pair():

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    protein_path = "timemachine/testsystems/data/hif2a_nowater_min.pdb"

    # fast
    seed = 2023
    run_pair(mol_a, mol_b, core, forcefield, n_frames=100, protein_path=protein_path, seed=seed)


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
    parser.add_argument("--n_frames", type=int, help="Number of frames to run", required=True)
    parser.add_argument("--seed", type=int, help="Random number seed", required=True)

    args = parser.parse_args()
    mols = read_sdf(str(args.ligands))
    mol_a = get_mol_by_name(mols, args.mol_a_name)  # 43 in test pair
    mol_b = get_mol_by_name(mols, args.mol_b_name)  # 30 in test pair

    print("Searching for the maximum common substructure...")
    mcs_result = atom_mapping.mcs(mol_a, mol_b)
    query_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    print("mol_a SMILES:", Chem.MolToSmiles(mol_a, isomericSmiles=False))
    print("mol_b SMILES:", Chem.MolToSmiles(mol_b, isomericSmiles=False))
    print("core SMARTS:", mcs_result.smartsString)

    core = atom_mapping.get_core_by_mcs(mol_a, mol_b, query_mol, threshold=2.0)
    print("core mapping:", core.tolist())

    res = plot_atom_mapping_grid(mol_a, mol_b, mcs_result.smartsString, core)
    with open(f"atom_mapping_{args.mol_a_name}_to_{args.mol_b_name}.svg", "w") as fh:
        fh.write(res)

    forcefield = Forcefield.load_from_file(args.forcefield)

    run_pair(mol_a, mol_b, core, forcefield, args.n_frames, args.protein, args.seed)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        hif2a_pair()
    else:
        read_from_args()
