# usage:
# python examples/validate_solvent_perturbations.py --ligands timemachine/testsystems/data/benzyl.sdf --mol_a_name benzene --mol_b_name benzene_bicycle --n_frames 200 --steps_per_frame 200 --n_eq_steps 5000 --seed 2024

# 1) using smoothcores
# 2) not-using smoothcores

# bisection only, (main goal is to check # of windows emitted in the fast bisection stage)
import argparse
import copy
import pickle

import numpy as np

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping
from timemachine.fe.free_energy import MDParams
from timemachine.fe.rbfe import HostConfig, estimate_relative_free_energy_bisection
from timemachine.fe.utils import plot_atom_mapping_grid, read_sdf
from timemachine.ff import Forcefield
from timemachine.md import builders


def get_mol_by_name(mols, name):
    for m in mols:
        if m.GetProp("_Name") == name:
            return m

    assert 0, "Mol not found"


def run_solvent(mol_a, mol_b, core, forcefield, md_params, use_smoothcore):
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(box_width, forcefield.water_ff)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, len(solvent_conf))
    solvent_res = estimate_relative_free_energy_bisection(
        mol_a,
        mol_b,
        core,
        forcefield,
        solvent_host_config,
        md_params=md_params,
        prefix="solvent",
        min_overlap=0.667,
        use_smoothcore=use_smoothcore,
    )

    print("==statistics==")
    print("lambda_schedule", [x.lamb for x in solvent_res.final_result.initial_states])
    print("dGs", solvent_res.final_result.dGs)
    print("num_windows", len(solvent_res.final_result.dGs) + 1)
    print("sum(dGs)", np.sum(solvent_res.final_result.dGs))
    print("norm(dG_errs)", np.linalg.norm(solvent_res.final_result.dG_errs))

    mol_a_name = mol_a.GetProp("_Name")
    mol_b_name = mol_b.GetProp("_Name")

    with open(f"solvent_summary_overlap_{mol_a_name}_to_{mol_b_name}_{use_smoothcore}.png", "wb") as fh:
        fh.write(solvent_res.plots.overlap_summary_png)

    with open(f"solvent_detail_overlap_{mol_a_name}_to_{mol_b_name}_{use_smoothcore}.png", "wb") as fh:
        fh.write(solvent_res.plots.overlap_detail_png)

    with open(f"solvent_debug_{mol_a_name}_to_{mol_b_name}_{use_smoothcore}.pkl", "wb") as f:
        pickle.dump(solvent_res, f)


def read_from_args():
    parser = argparse.ArgumentParser(description="Estimate relative solvation free energy between two ligands")
    parser.add_argument(
        "--n_frames", type=int, help="number of frames to use for the free energy estimate", required=True
    )
    parser.add_argument("--ligands", type=str, help="SDF file containing the ligands of interest", required=True)
    parser.add_argument("--mol_a_name", type=str, help="name of the start molecule", required=True)
    parser.add_argument("--mol_b_name", type=str, help="name of the end molecule", required=True)
    parser.add_argument("--steps_per_frame", type=int, help="steps per frame", required=True)
    parser.add_argument("--n_eq_steps", type=int, help="n eq steps", required=True)
    parser.add_argument("--seed", type=int, help="Random number seed", required=True)

    args = parser.parse_args()
    mols = read_sdf(str(args.ligands))
    mol_a = get_mol_by_name(mols, args.mol_a_name)
    mol_b = get_mol_by_name(mols, args.mol_b_name)

    md_params = MDParams(
        n_frames=args.n_frames, n_eq_steps=args.n_eq_steps, steps_per_frame=args.steps_per_frame, seed=args.seed
    )

    ff = Forcefield.load_default()

    atom_mapping_kwargs = copy.copy(DEFAULT_ATOM_MAPPING_KWARGS)
    atom_mapping_kwargs["ring_matches_ring_only"] = False

    all_cores = atom_mapping.get_cores(mol_a, mol_b, **atom_mapping_kwargs)

    core = all_cores[0]

    # Hydrogen Only
    # core = np.array([[8, 8]])

    res = plot_atom_mapping_grid(mol_a, mol_b, core)
    fpath = f"atom_mapping_{args.mol_a_name}_to_{args.mol_b_name}.svg"
    print("core mapping written to", fpath)
    with open(fpath, "w") as fh:
        fh.write(res)

    # debug
    from timemachine.fe.single_topology import SingleTopology

    st = SingleTopology(mol_a, mol_b, core, ff, use_smoothcore=True)
    all_charges = []
    all_eps = []
    all_ws = []
    all_mols = []
    for lamb in np.linspace(0, 1, 10):
        guest_ixn_env_params = st._get_smoothcore_guest_params(ff.q_handle, ff.lj_handle, lamb)
        all_charges.append(guest_ixn_env_params[:, 0])
        all_eps.append(guest_ixn_env_params[:, 2])
        all_ws.append(guest_ixn_env_params[:, 3])
        all_mols.append(st.mol(lamb))
    all_charges = np.array(all_charges)
    all_eps = np.array(all_eps)
    all_ws = np.array(all_ws)
    num_atoms = all_charges.shape[1]
    dummy_a_atoms = st.dummy_a_idxs()
    dummy_b_atoms = st.dummy_b_idxs()
    import matplotlib.pyplot as plt

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    for atom_idx in range(num_atoms):
        symbol = all_mols[0].GetAtomWithIdx(atom_idx).GetSymbol()
        if atom_idx in dummy_a_atoms:
            linestyle = "dashed"
        elif atom_idx in dummy_b_atoms:
            linestyle = "dotted"
        else:
            linestyle = "solid"
        ax1.plot(all_charges[:, atom_idx], label=f"{symbol}{atom_idx}", linestyle=linestyle)
        ax2.plot(all_ws[:, atom_idx], label=f"{symbol}{atom_idx}", linestyle=linestyle)
        ax3.plot(all_eps[:, atom_idx], label=f"{symbol}{atom_idx}", linestyle=linestyle)
    ax1.set_ylabel("charge")
    ax1.set_xlabel("lambda_idx")

    ax2.set_ylabel("w")
    ax2.set_xlabel("lambda_idx")

    ax3.set_ylabel("eps")
    ax3.set_xlabel("lambda_idx")

    ax3.legend()
    mol_a_name = mol_a.GetProp("_Name")
    mol_b_name = mol_b.GetProp("_Name")
    plt.tight_layout()
    plt.savefig(f"solvent_parameter_interpolation_{mol_a_name}_to_{mol_b_name}.png")
    # debug

    run_solvent(mol_a, mol_b, core, ff, md_params, use_smoothcore=True)
    run_solvent(mol_a, mol_b, core, ff, md_params, use_smoothcore=False)


if __name__ == "__main__":
    read_from_args()
