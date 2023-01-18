import numpy as np
import pytest

from timemachine.constants import DEFAULT_FF, DEFAULT_TEMP
from timemachine.fe import atom_mapping, pdb_writer, utils
from timemachine.fe.lambda_schedule import construct_pre_optimized_relative_lambda_schedule
from timemachine.fe.rbfe import HostConfig, setup_initial_states
from timemachine.fe.single_topology import AtomMapMixin, SingleTopology
from timemachine.fe.utils import get_mol_name, read_sdf
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


def run_edge(mol_a, mol_b, protein_path, n_windows):

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
    res = utils.plot_atom_mapping_grid(mol_a, mol_b, core)
    with open(f"edge_map_{get_mol_name(mol_a)}_{get_mol_name(mol_b)}.svg", "w") as fh:
        fh.write(res)

    ff = Forcefield.load_from_file(DEFAULT_FF)
    st = SingleTopology(mol_a, mol_b, core, ff)

    lambda_schedule = construct_pre_optimized_relative_lambda_schedule(n_windows)
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
def test_confgen_hard_edges():
    # 1) cherry pick a couple of edges that are hard to setup initial geometries
    # 2) failures come from failed edges in the original hif2a set
    edges = [
        ("30", "25"),  # core-hopping B-ring
        ("227", "266"),  # core-hopping A-ring, bicycle
        ("7a", "224"),  # core-hopping A-ring, 6->5 member,
        ("227", "256"),  # stereo close-proximity down H with up H on B-ring,
        ("289", "61"),  # stereo close-proximity down H with up H on B-ring,
        ("266", "165"),  # core-hopping A-ring, and stereo close-proximity Hs on B-ring,
        ("234", "227"),  # failure, stereo close proximity
        ("1", "155"),  # failure, single nitrile mutation
        ("1", "7a"),  # failure, methyl extension on B-ring
        ("266", "156"),  # failure, A-ring corehopping unicycle -> bicycle, B ring double fluorination
        ("289", "224"),  # failure, A-ring, 6->5 member
        ("165", "42"),  # failure, B-ring, core-hopping,
        ("7b", "42"),  # failure, B-ring, core-hopping,
        ("252", "290"),  # failure, should be a simple transformation
        ("290", "84"),  # failure, B-ring, core-hopping into oxazole
        ("290", "256"),  # failure, should be a simple transformation
        ("164", "163"),  # failure, B-ring has a different conformation
    ]

    protein_path = "timemachine/testsystems/data/hif2a_nowater_min.pdb"
    ligands = "timemachine/datasets/fep_benchmark/hif2a/ligands.sdf"
    mols = read_sdf(ligands)

    n_windows = 12

    for src, dst in edges:
        print("\nProcessing", src, "->", dst, "\n")
        mol_a = get_mol_by_name(mols, src)
        mol_b = get_mol_by_name(mols, dst)
        # try both directions
        run_edge(mol_a, mol_b, protein_path, n_windows)
        run_edge(mol_b, mol_a, protein_path, n_windows)


def test_confgen_spot_edges():
    # spot check so we have something in unit testing.
    edges = [
        ("35", "84"),  # failure, B-ring, core-hopping into oxazole, <-- this fails
    ]

    protein_path = "timemachine/testsystems/data/hif2a_nowater_min.pdb"
    ligands = "timemachine/datasets/fep_benchmark/hif2a/ligands.sdf"
    mols = read_sdf(ligands)

    n_windows = 12

    for src, dst in edges:
        print("\nProcessing", src, "->", dst, "\n")
        mol_a = get_mol_by_name(mols, src)
        mol_b = get_mol_by_name(mols, dst)
        run_edge(mol_a, mol_b, protein_path, n_windows)
