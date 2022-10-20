import csv

import numpy as np
import pytest
from rdkit import Chem

from timemachine.constants import DEFAULT_FF, DEFAULT_TEMP
from timemachine.fe import atom_mapping, pdb_writer, utils
from timemachine.fe.rbfe import HostConfig, setup_initial_states
from timemachine.fe.single_topology import AtomMapMixin, SingleTopology
from timemachine.fe.utils import get_mol_name
from timemachine.ff import Forcefield
from timemachine.md import builders


def get_mol_by_name(mols, name):
    for m in mols:
        if get_mol_name(m) == name:
            return m
    assert 0, "Mol not found"


def write_trajectory_as_pdb(mol_a, mol_b, core, all_frames, host_topology, out_path):
    atom_map_mixin = AtomMapMixin(mol_a, mol_b, core)
    writer = pdb_writer.PDBWriter([host_topology, mol_a, mol_b], out_path)
    for frame in all_frames:
        host_frame = frame[: host_topology.getNumAtoms()]
        ligand_frame = frame[host_topology.getNumAtoms() :]
        mol_ab_frame = pdb_writer.convert_single_topology_mols(ligand_frame, atom_map_mixin)
        writer.write_frame(np.concatenate([host_frame, mol_ab_frame]) * 10)
    writer.close()


def run_edge(mol_a, mol_b, protein_path):

    threshold = 2.0
    core, smarts = atom_mapping.get_core_with_alignment(mol_a, mol_b, threshold=threshold)
    res = utils.plot_atom_mapping_grid(mol_a, mol_b, smarts, core)
    with open(f"edge_map_{get_mol_name(mol_a)}_{get_mol_name(mol_b)}.svg", "w") as fh:
        fh.write(res)

    ff = Forcefield.load_from_file(DEFAULT_FF)
    st = SingleTopology(mol_a, mol_b, core, ff)

    lambda_schedule = np.linspace(0, 1, 12)
    seed = 2023

    # solvent
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(box_width, ff.water_ff)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box)
    initial_states = setup_initial_states(st, solvent_host_config, DEFAULT_TEMP, lambda_schedule, seed)
    all_frames = [state.x0 for state in initial_states]
    write_trajectory_as_pdb(
        mol_a,
        mol_b,
        core,
        all_frames,
        solvent_top,
        f"solvent_{get_mol_name(mol_a)}_{get_mol_name(mol_b)}.pdb",
    )

    # complex
    complex_sys, complex_conf, _, _, complex_box, complex_top = builders.build_protein_system(
        protein_path, ff.protein_ff, ff.water_ff
    )
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box)
    initial_states = setup_initial_states(st, complex_host_config, DEFAULT_TEMP, lambda_schedule, seed)
    all_frames = [state.x0 for state in initial_states]
    write_trajectory_as_pdb(
        mol_a,
        mol_b,
        core,
        all_frames,
        complex_top,
        f"complex_{get_mol_name(mol_a)}_{get_mol_name(mol_b)}.pdb",
    )


@pytest.mark.nightly(reason="Takes a while to run")
def test_confgen():

    protein_path = "timemachine/testsystems/data/hif2a_nowater_min.pdb"
    ligands = "timemachine/datasets/fep_benchmark/hif2a/ligands.sdf"
    mols = [mol for mol in Chem.SDMolSupplier(ligands, removeHs=False)]

    results_csv = "timemachine/datasets/fep_benchmark/hif2a/results_edges_5ns.csv"
    with open(results_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        rows = [row for row in reader]
        for row_idx, row in enumerate(rows):
            mol_a_name, mol_b_name, exp_ddg, fep_ddg, fep_ddg_err, ccc_ddg, ccc_ddg_err = row
            print("Processing", mol_a_name, "->", mol_b_name)
            mol_a = get_mol_by_name(mols, mol_a_name)
            mol_b = get_mol_by_name(mols, mol_b_name)

            run_edge(mol_a, mol_b, protein_path)
