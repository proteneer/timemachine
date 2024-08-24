# construct a relative transformation
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import special_ortho_group

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping, cif_writer, gpmol
from timemachine.fe.free_energy import HREXParams, MDParams
from timemachine.fe.plots import (  # plot_hrex_replica_state_distribution_convergence,
    plot_hrex_replica_state_distribution,
    plot_hrex_replica_state_distribution_heatmap,
    plot_hrex_swap_acceptance_rates_convergence,
    plot_hrex_transition_matrix,
)
from timemachine.fe.rbfe import estimate_relative_free_energy, estimate_relative_free_energy_bisection_hrex
from timemachine.fe.single_topology import AtomMapMixin, SingleTopology
from timemachine.fe.system import simulate_system
from timemachine.fe.utils import get_romol_conf, plot_atom_mapping_grid, read_sdf, recenter_mol, rotate_mol
from timemachine.ff import Forcefield


def process_core(gp_a, gp_b):
    cur_gp = gp_a
    path_gps = []

    counter = 0
    while True:
        counter += 1
        path_gps.append(cur_gp)

        new_gp = None
        for atom_idx_in_a in cur_gp.find_allowed_core_geometry_mutations(gp_b):
            new_gp = cur_gp.mutate_core_atom(atom_idx_in_a, gp_b)
            break

        if new_gp:
            cur_gp = new_gp
        else:
            break

    return path_gps


def write_trajectory_as_cif(mol_a, mol_b, core, all_frames, prefix):
    atom_map_mixin = AtomMapMixin(mol_a, mol_b, core)
    for window_idx, window_frames in enumerate(all_frames):
        out_path = f"{prefix}_{window_idx}.cif"
        writer = cif_writer.CIFWriter([mol_a, mol_b], out_path)
        for ligand_frame in window_frames:
            mol_ab_frame = cif_writer.convert_single_topology_mols(ligand_frame, atom_map_mixin)
            writer.write_frame(mol_ab_frame * 10)
        writer.close()


def plot_and_save(f, fname, *args, **kwargs) -> bytes:
    """
    Given a function which generates a plot, return the plot as png bytes.
    """
    plt.clf()
    f(*args, **kwargs)
    with open(fname, "wb") as fh:
        plt.savefig(fh, format="png", bbox_inches="tight")


def run_pair_vacuum(mol_a, mol_b, core, forcefield, md_params):
    vacuum_res = estimate_relative_free_energy_bisection_hrex(
        mol_a, mol_b, core, forcefield, None, md_params=md_params, prefix="vacuum", min_overlap=0.6667, n_windows=64
    )

    # vacuum_res = estimate_relative_free_energy(
    #     mol_a, mol_b, core, forcefield, None, prefix="vacuum",
    #     lambda_interval=(0, 1), # breaks
    #     # lambda_interval=(0.61, 0.63),
    #     n_windows=10, md_params=md_params
    # )

    with open("vacuum_overlap.png", "wb") as fh:
        fh.write(vacuum_res.plots.overlap_detail_png)

    plot_and_save(
        plot_hrex_swap_acceptance_rates_convergence,
        "vac_plot_hrex_swap_acceptance_rates_convergence.png",
        vacuum_res.hrex_diagnostics.cumulative_swap_acceptance_rates,
    )
    plot_and_save(
        plot_hrex_transition_matrix,
        "vac_plot_hrex_transition_matrix.png",
        vacuum_res.hrex_diagnostics.transition_matrix,
    )
    plot_and_save(
        plot_hrex_replica_state_distribution,
        "vac_plot_hrex_replica_state_distribution.png",
        vacuum_res.hrex_diagnostics.cumulative_replica_state_counts,
    )
    # plot_and_save(
    #     plot_hrex_replica_state_distribution_convergence,
    #     "vac_plot_hrex_replica_state_distribution_convergence.png",
    #     vacuum_res.hrex_diagnostics.cumulative_replica_state_counts,
    # )
    plot_and_save(
        plot_hrex_replica_state_distribution_heatmap,
        "vac_plot_hrex_replica_state_distribution_heatmap.png",
        vacuum_res.hrex_diagnostics.cumulative_replica_state_counts,
    )


def test_run_vacuum_pair():
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(str(path_to_ligand))
    mol_a = all_mols[1]
    mol_b = all_mols[4]
    cores = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)
    core = cores[0]
    ff = Forcefield.load_default()

    st = gpmol.SingleTopologyV5(mol_a, mol_b, core, ff)
    fpath = "path_all.svg"
    with open(fpath, "w") as fh:
        fh.write(st.draw_path())

    fpath = f"atom_mapping_{mol_a.GetProp('_Name')}_{mol_b.GetProp('_Name')}.svg"
    with open(fpath, "w") as fh:
        from timemachine.fe.utils import plot_atom_mapping_grid

        fh.write(plot_atom_mapping_grid(mol_a, mol_b, core))

    hrex_params = HREXParams(100, 1)
    md_params = MDParams(n_frames=1000, n_eq_steps=10_000, steps_per_frame=400, seed=2024, hrex_params=hrex_params)
    run_pair_vacuum(mol_a, mol_b, core, ff, md_params)


def test_gmol():
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(str(path_to_ligand))
    mol_a = all_mols[8]
    mol_b = all_mols[1]

    cores = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)
    core = cores[0]

    ff = Forcefield.load_default()

    fpath = f"atom_mapping_{mol_a.GetProp('_Name')}_{mol_b.GetProp('_Name')}.svg"
    with open(fpath, "w") as fh:
        from timemachine.fe.utils import plot_atom_mapping_grid

        fh.write(plot_atom_mapping_grid(mol_a, mol_b, core))

    # draw path
    st = gpmol.SingleTopologyV5(mol_a, mol_b, core, ff)
    # pm_all = st.mol_a_path + st.mol_b_path[::-1]
    # pm_all = [recenter_mol(pm.induced_mol()) for pm in pm_all]
    # extra_rotations = generate_good_rotations(pm_all, num_rotations=3)

    # extra_mols = []

    # legends = [f"lamb={x:.2f}" for x in st.get_checkpoint_lambdas()]
    # for rot in extra_rotations:
    #     for pm in pm_all:
    #         extra_mols.append(rotate_mol(pm, rot))
    #         legends.append("")

    # svg = Draw.MolsToGridImage(pm_all + extra_mols, useSVG=True, molsPerRow=len(pm_all), legends=legends)

    fpath = "path_all.svg"
    with open(fpath, "w") as fh:
        fh.write(st.draw_path())

    ref_frame = None

    # for lamb_idx, lamb in enumerate(np.linspace(0, 1, 2)):
    for lamb_idx, lamb in enumerate([0.59, 0.61]):
        # print("Processing lambda", lamb)
        i_state = st.setup_intermediate_state(lamb)
        i_mol, i_kv = st.setup_intermediate_mol_and_kv(lamb)

        # continue
        U_fn = i_state.get_U_fn()
        x_a = get_romol_conf(mol_a)
        x_b = get_romol_conf(mol_b)
        x0 = SingleTopology(mol_a, mol_b, core, ff).combine_confs(x_a, x_b, 0)
        frames = simulate_system(U_fn, x0, num_samples=200)

        with open(f"intermediate_{lamb_idx}.sdf", "w") as fh:
            writer = Chem.SDWriter(fh)
            for frame in frames:
                from scipy.spatial.transform import Rotation

                frame = frame - np.mean(frame, axis=0, keepdims=True)
                if ref_frame is None:
                    ref_frame = frame
                R, _ = Rotation.align_vectors(ref_frame, frame)
                frame = R.apply(frame)

                # frame -= np.mean(frame, axis=0)
                mol_conf = Chem.Conformer(i_mol.GetNumAtoms())
                mol_copy = Chem.Mol(i_mol)
                for a_idx, pos in enumerate(frame):
                    if i_kv[a_idx] != -1:
                        mol_conf.SetAtomPosition(i_kv[a_idx], (pos * 10).astype(np.float64))
                mol_copy.AddConformer(mol_conf)
                writer.write(mol_copy)
            writer.close()

    assert 0

    i_mols, kvs = st.generate_intermediate_mols_and_kvs()
    # 0.3 relative to states 1/2, i.e. => 1.3333
    intermediate_state = gpmol.setup_intermediate_state_standard(0.3, st.checkpoint_states[2], st.checkpoint_states[3])

    # careful: need to use the right idx if we're going from the right!
    i_mol, kv = i_mols[2], kvs[2]

    print("i_mol atoms/bonds", i_mol.GetNumAtoms(), i_mol.GetNumBonds())

    print("Processing")
    U_fn = intermediate_state.get_U_fn()
    x_a = get_romol_conf(mol_a)
    x_b = get_romol_conf(mol_b)
    x0 = SingleTopology(mol_a, mol_b, core, ff).combine_confs(x_a, x_b, 0)
    frames = simulate_system(U_fn, x0, num_samples=200)

    with open(f"intermediate.sdf", "w") as fh:
        writer = Chem.SDWriter(fh)
        for frame in frames:
            frame -= np.mean(frame, axis=0)
            mol_conf = Chem.Conformer(i_mol.GetNumAtoms())
            mol_copy = Chem.Mol(i_mol)
            for a_idx, pos in enumerate(frame):
                if kv[a_idx] != -1:
                    mol_conf.SetAtomPosition(kv[a_idx], (pos * 10).astype(np.float64))
            mol_copy.AddConformer(mol_conf)
            writer.write(mol_copy)
        writer.close()

    # use left one right one depending if we're < left_idx or not.

    for idx, (vs, i_mol, kv) in enumerate(zip(st.checkpoint_states, i_mols, kvs)):
        print("i_mol atoms/bonds", i_mol.GetNumAtoms(), i_mol.GetNumBonds())

        print("Processing checkpoint state", idx)
        U_fn = vs.get_U_fn()
        x_a = get_romol_conf(mol_a)
        x_b = get_romol_conf(mol_b)
        x0 = SingleTopology(mol_a, mol_b, core, ff).combine_confs(x_a, x_b, 0)
        frames = simulate_system(U_fn, x0, num_samples=200)

        with open(f"traj_{idx}.sdf", "w") as fh:
            writer = Chem.SDWriter(fh)
            for frame in frames:
                frame -= np.mean(frame, axis=0)
                mol_conf = Chem.Conformer(i_mol.GetNumAtoms())
                mol_copy = Chem.Mol(i_mol)
                for a_idx, pos in enumerate(frame):
                    if kv[a_idx] != -1:
                        mol_conf.SetAtomPosition(kv[a_idx], (pos * 10).astype(np.float64))
                mol_copy.AddConformer(mol_conf)
                writer.write(mol_copy)
            writer.close()

        # assert 0

    assert 0

    # print(path_mols)
