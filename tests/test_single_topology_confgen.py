from importlib import resources

import numpy as np
import pytest
from rdkit import Chem

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS, DEFAULT_TEMP
from timemachine.fe import atom_mapping, cif_writer, utils
from timemachine.fe.free_energy import HostConfig
from timemachine.fe.rbfe import (
    get_free_idxs,
    optimize_coords_state,
    setup_initial_state,
    setup_initial_states,
    setup_optimized_host,
)
from timemachine.fe.single_topology import AtomMapMixin, SingleTopology
from timemachine.fe.utils import get_mol_name, read_sdf_mols_by_name
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.potentials.jax_utils import distance_on_pairs

SAVE_FRAMES = False


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
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, solvent_conf.shape[0], solvent_top)
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
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box, num_water_atoms, complex_top)
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


# 1) cherry pick a couple of edges that are hard to setup initial geometries
# 2) failures come from failed edges in the original hif2a set
@pytest.mark.parametrize(
    "src, dst",
    [
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
        ("35", "84"),  # failure, B-ring, core-hopping into oxazole, <-- this fails
        ("41", "50"),  # failure, moves beyond 7 A in the solvent leg
    ],
)
@pytest.mark.nightly(reason="Takes a while to run")
def test_confgen_hard_edges(src, dst):
    protein_path = "timemachine/testsystems/data/hif2a_nowater_min.pdb"
    with resources.path("timemachine.datasets.fep_benchmark.hif2a", "ligands.sdf") as ligand_path:
        mols_by_name = read_sdf_mols_by_name(ligand_path)

    n_windows = 12

    print("\nProcessing", src, "->", dst, "\n")
    mol_a = mols_by_name[src]
    mol_b = mols_by_name[dst]
    # try both directions
    run_edge(mol_a, mol_b, protein_path, n_windows)
    run_edge(mol_b, mol_a, protein_path, n_windows)


@pytest.mark.parametrize(
    "src, dst",
    [
        ("35", "84"),  # failure, B-ring, core-hopping into oxazole, <-- this fails
    ],
)
def test_confgen_spot_edges(src, dst):
    # spot check so we have something in unit testing.
    protein_path = "timemachine/testsystems/data/hif2a_nowater_min.pdb"
    with resources.path("timemachine.datasets.fep_benchmark.hif2a", "ligands.sdf") as ligand_path:
        mols_by_name = read_sdf_mols_by_name(ligand_path)

    n_windows = 3

    print("\nProcessing", src, "->", dst, "\n")
    mol_a = mols_by_name[src]
    mol_b = mols_by_name[dst]
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
        mols_by_name = read_sdf_mols_by_name(ligand_path)

    mol_a = mols_by_name[src]
    mol_b = mols_by_name[dst]

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
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, solvent_conf.shape[0], solvent_top)
    solvent_host = setup_optimized_host(st, solvent_host_config)
    ligand_idxs = np.arange(st.get_num_atoms()) + solvent_host.conf.shape[0]
    expected_moved = ligand_idxs[st.c_flags != 2]
    with pytest.raises(AssertionError) as res:
        setup_initial_states(st, solvent_host, DEFAULT_TEMP, lambda_grid, seed, min_cutoff=min_cutoff)
    assert f"moved atoms {expected_moved.tolist()} >" in str(res.value)


def test_optimize_coords_with_positional_restraint():
    mol = Chem.MolFromMolBlock(
        """Cyclohexane with clashy methyls
RDKit          3D

 24 24  0  0  1  0            999 V2000
    9.9445  -15.4797  197.9278 C   0  0  0  0  0  0
   10.4220  -14.4752  196.8506 C   0  0  2  0  0  0
   11.9342  -14.2011  197.0256 C   0  0  0  0  0  0
   12.7498  -15.5057  197.0172 C   0  0  0  0  0  0
   12.3085  -16.4450  198.1504 C   0  0  0  0  0  0
   10.8012  -16.7661  198.0776 C   0  0  1  0  0  0
   10.0655  -14.8776  195.4099 C   0  0  0  0  0  0
   10.4814  -17.8631  197.0485 C   0  0  0  0  0  0
    8.9099  -15.7577  197.7270 H   0  0  0  0  0  0
    9.8847  -14.9730  198.8910 H   0  0  0  0  0  0
    9.9081  -13.5348  197.0495 H   0  0  0  0  0  0
   12.1011  -13.6750  197.9656 H   0  0  0  0  0  0
   12.2805  -13.5516  196.2216 H   0  0  0  0  0  0
   13.8085  -15.2725  197.1309 H   0  0  0  0  0  0
   12.6208  -16.0081  196.0585 H   0  0  0  0  0  0
   12.5351  -15.9846  199.1120 H   0  0  0  0  0  0
   12.8796  -17.3718  198.0963 H   0  0  0  0  0  0
   10.5398  -17.1853  199.0492 H   0  0  0  0  0  0
   10.4359  -14.1208  194.7184 H   0  0  0  0  0  0
   10.5245  -15.8382  195.1762 H   0  0  0  0  0  0
    8.9829  -14.9595  195.3123 H   0  0  0  0  0  0
   11.1242  -18.7256  197.2243 H   0  0  0  0  0  0
    9.4379  -18.1620  197.1478 H   0  0  0  0  0  0
   10.6549  -17.4802  196.0428 H   0  0  0  0  0  0
  1  2  1  0  0  0
  1  6  1  0  0  0
  1  9  1  0  0  0
  1 10  1  0  0  0
  2  3  1  0  0  0
  2  7  1  0  0  0
  2 11  1  0  0  0
  3  4  1  0  0  0
  3 12  1  0  0  0
  3 13  1  0  0  0
  4  5  1  0  0  0
  4 14  1  0  0  0
  4 15  1  0  0  0
  5  6  1  0  0  0
  5 16  1  0  0  0
  5 17  1  0  0  0
  6  8  1  0  0  0
  6 18  1  0  0  0
  7 19  1  0  0  0
  7 20  1  0  0  0
  7 21  1  0  0  0
  8 22  1  0  0  0
  8 23  1  0  0  0
  8 24  1  0  0  0
M  END
$$$$
""",
        removeHs=False,
    )

    ff = Forcefield.load_default()
    # Core doesn't matter here
    core = np.array([[0, 0]])
    st = SingleTopology(mol, mol, core, ff)

    lambda_grid = np.linspace(0.0, 1.0, 2)

    for lamb in lambda_grid:
        state = setup_initial_state(st, lamb, None, DEFAULT_TEMP, 2024)

        free_idxs = get_free_idxs(state)
        x_opt_unrestrained = optimize_coords_state(state.potentials, state.x0, state.box0, free_idxs, False, k=0.0)
        x_opt_restrained = optimize_coords_state(state.potentials, state.x0, state.box0, free_idxs, False, k=2000.0)
        interacting_atoms = state.interacting_atoms
        displacement_distances_unrestrained = distance_on_pairs(
            state.x0[interacting_atoms], x_opt_unrestrained[interacting_atoms], box=state.box0
        )
        displacement_distances_restrained = distance_on_pairs(
            state.x0[interacting_atoms], x_opt_restrained[interacting_atoms], box=state.box0
        )
        assert np.max(displacement_distances_unrestrained) > np.max(displacement_distances_restrained)
