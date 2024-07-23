from importlib import resources

import numpy as np
import pytest

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS, DEFAULT_TEMP
from timemachine.fe import atom_mapping, cif_writer, utils
from timemachine.fe.rbfe import HostConfig, setup_initial_states, setup_optimized_host
from timemachine.fe.single_topology import AtomMapMixin, SingleTopology
from timemachine.fe.utils import get_mol_name, read_sdf
from timemachine.ff import Forcefield
from timemachine.md import builders

SAVE_FRAMES = False


def get_mol_by_name(mols, name):
    for m in mols:
        if get_mol_name(m) == name:
            return m
    assert 0, "Mol not found"


def write_trajectory_as_cif(mol_a, mol_b, core, all_frames, host_topology, out_path):
    atom_map_mixin = AtomMapMixin(mol_a, mol_b, core)
    writer = cif_writer.CIFWriter([host_topology, mol_a, mol_b], out_path)
    for frame in all_frames:
        host_frame = frame[: host_topology.getNumAtoms()]
        ligand_frame = frame[host_topology.getNumAtoms() :]
        mol_ab_frame = cif_writer.convert_single_topology_mols(ligand_frame, atom_map_mixin)
        writer.write_frame(np.concatenate([host_frame, mol_ab_frame]) * 10)
    writer.close()


def run_edge(mol_a, mol_b, protein_path, n_windows):
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )
    core = all_cores[0]
    res = utils.plot_atom_mapping_grid(mol_a, mol_b, core)
    with open(f"edge_map_{get_mol_name(mol_a)}_{get_mol_name(mol_b)}.svg", "w") as fh:
        fh.write(res)

    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)

    # Use the lambda schedule that would be used in bisection
    lambda_schedule = np.linspace(0.0, 1.0, n_windows)
    seed = 2023

    # solvent
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(
        box_width, ff.water_ff, mols=[mol_a, mol_b]
    )
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, solvent_conf.shape[0])
    solvent_host = setup_optimized_host(st, solvent_host_config)
    initial_states = setup_initial_states(st, solvent_host, DEFAULT_TEMP, lambda_schedule, seed)

    if SAVE_FRAMES:
        all_frames = [state.x0 for state in initial_states]
        write_trajectory_as_cif(
            mol_a,
            mol_b,
            core,
            all_frames,
            solvent_top,
            f"solvent_{get_mol_name(mol_a)}_{get_mol_name(mol_b)}.cif",
        )

    # complex
    complex_sys, complex_conf, complex_box, complex_top, num_water_atoms = builders.build_protein_system(
        protein_path, ff.protein_ff, ff.water_ff, mols=[mol_a, mol_b]
    )
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box, num_water_atoms)
    complex_host = setup_optimized_host(st, complex_host_config)
    initial_states = setup_initial_states(st, complex_host, DEFAULT_TEMP, lambda_schedule, seed, min_cutoff=0.7)

    if SAVE_FRAMES:
        all_frames = [state.x0 for state in initial_states]
        write_trajectory_as_cif(
            mol_a,
            mol_b,
            core,
            all_frames,
            complex_top,
            f"complex_{get_mol_name(mol_a)}_{get_mol_name(mol_b)}.cif",
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
    with resources.path("timemachine.datasets.fep_benchmark.hif2a", "ligands.sdf") as ligand_path:
        mols = read_sdf(ligand_path)

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
        ("41", "50"),  # failure, moves beyond 7 A in the solvent leg
    ]

    protein_path = "timemachine/testsystems/data/hif2a_nowater_min.pdb"
    with resources.path("timemachine.datasets.fep_benchmark.hif2a", "ligands.sdf") as ligand_path:
        mols = read_sdf(ligand_path)

    n_windows = 12

    for src, dst in edges:
        print("\nProcessing", src, "->", dst, "\n")
        mol_a = get_mol_by_name(mols, src)
        mol_b = get_mol_by_name(mols, dst)
        run_edge(mol_a, mol_b, protein_path, n_windows)


@pytest.mark.parametrize("pair", [("35", "84")])
@pytest.mark.parametrize("n_windows", [2])
@pytest.mark.parametrize("seed", [2023])
def test_min_cutoff_failure(pair, seed, n_windows):
    """Verify that when minimizing within solvent only the ligand is considered for movement of atoms
    during minimization. Expectation is that all atoms of the molecules will move beyond effectively zero cutoff"""
    src, dst = pair
    box_width = 4.0
    # The cutoff is so small that any ligand pair should trigger the exception
    min_cutoff = 1e-8

    with resources.path("timemachine.datasets.fep_benchmark.hif2a", "ligands.sdf") as ligand_path:
        mols = read_sdf(ligand_path)
    mol_a = get_mol_by_name(mols, src)
    mol_b = get_mol_by_name(mols, dst)

    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )
    core = all_cores[0]

    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)

    # Use the lambda grid as defined for Bisection
    lambda_grid = np.linspace(0.0, 1.0, n_windows)

    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(
        box_width, ff.water_ff, mols=[mol_a, mol_b]
    )
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, solvent_conf.shape[0])
    solvent_host = setup_optimized_host(st, solvent_host_config)
    ligand_idxs = np.arange(st.get_num_atoms()) + solvent_host.conf.shape[0]
    expected_moved = ligand_idxs[st.c_flags != 2]
    with pytest.raises(AssertionError) as res:
        setup_initial_states(st, solvent_host, DEFAULT_TEMP, lambda_grid, seed, min_cutoff=min_cutoff)
    assert f"moved atoms {expected_moved.tolist()} >" in str(res.value)
