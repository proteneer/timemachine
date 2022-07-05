import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pymbar
from rdkit import Chem

from timemachine.constants import BOLTZ
from timemachine.fe import pdb_writer
from timemachine.fe.single_topology_v3 import SingleTopologyV3
from timemachine.fe.system import convert_bps_into_system
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def sample(bound_impls, masses, x0, v0, box0, lamb, group_idxs, seed, n_frames=500, burn_in=10000):

    dt = 1e-3

    temperature = 300.0
    friction = 1.0

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)
    intg_impl = intg.impl()

    baro = MonteCarloBarostat(len(masses), 1.0, temperature, group_idxs, 15, seed + 1)
    baro_impl = baro.impl(bound_impls)

    # tbd: add barostat
    ctxt = custom_ops.Context(x0, v0, box0, intg_impl, bound_impls, baro_impl)

    # burn-in
    ctxt.multiple_steps_U(
        lamb=lamb,
        n_steps=burn_in,
        lambda_windows=[lamb],
        store_u_interval=0,
        store_x_interval=0,
    )

    steps_per_frame = 750
    # production
    n_steps = n_frames * steps_per_frame
    all_nrgs, all_coords, all_boxes = ctxt.multiple_steps_U(
        lamb=lamb,
        n_steps=n_steps,
        lambda_windows=[lamb],
        store_u_interval=steps_per_frame,
        store_x_interval=steps_per_frame,
    )

    assert all_coords.shape[0] == n_frames
    assert all_boxes.shape[0] == n_frames

    return all_coords, all_boxes


def get_batch_U_fn(bps, lamb):

    # return a function that takes in coords, boxes, lambda
    def batch_U_fn(xs, boxes):
        batch_Us = []
        for x, box in zip(xs, boxes):
            Us = []
            for bp in bps:
                _, _, U = bp.execute(x, box, lamb)
                Us.append(U)
            batch_Us.append(np.sum(Us))
        return np.array(batch_Us)

    return batch_U_fn


class HostConfig:
    def __init__(self, omm_system, omm_topology, conf, box):
        self.omm_system = omm_system
        self.omm_topology = omm_topology
        self.conf = conf
        self.box = box


def estimate_relative_free_energy(mol_a, mol_b, core, ff, host_config, n_frames=1000, prefix=""):

    st = SingleTopologyV3(mol_a, mol_b, core, ff)
    host_bps, host_masses = openmm_deserializer.deserialize_system(host_config.omm_system, cutoff=1.2)
    num_host_atoms = len(host_masses)
    host_conf = minimizer.minimize_host_4d(
        [mol_a, mol_b],
        host_config.omm_system,
        host_config.conf,
        ff,
        host_config.box,
    )

    # manually optimized by eye-balling
    # 14-windows, later: how do i use spline interpolation here?
    lambda_schedule = np.array([0.0, 0.06, 0.11, 0.15, 0.20, 0.32, 0.42])
    lambda_schedule = np.concatenate([lambda_schedule, (1 - lambda_schedule[::-1])])

    print("Lambda Schedule:", lambda_schedule)

    batch_U_fns = []
    all_frames = []
    all_boxes = []

    all_dGs = []
    all_errs = []

    for lamb_idx, lamb in enumerate(lambda_schedule):

        hgs = st.combine_with_host(convert_bps_into_system(host_bps), lamb=lamb)
        bound_impls = [p.bound_impl(np.float32) for p in hgs.get_U_fns()]
        batch_U_fns.append(get_batch_U_fn(bound_impls, lamb))

        # minimize water box around the ligand by 4D-decoupling
        ligand_conf = st.combine_confs(
            get_romol_conf(mol_a),
            get_romol_conf(mol_b),
        )

        combined_conf = np.concatenate([host_conf, ligand_conf])

        x0 = combined_conf
        v0 = np.zeros_like(x0)
        box0 = host_config.box

        group_idxs = get_group_indices(get_bond_list(hgs.bond))

        seed = np.random.randint(0, 1000)
        combined_masses = np.concatenate([host_masses, st.combine_masses()])
        frames, boxes = sample(bound_impls, combined_masses, x0, v0, box0, lamb, group_idxs, seed, n_frames=n_frames)
        all_frames.append(frames)
        all_boxes.append(boxes)

        writer = pdb_writer.PDBWriter(
            [host_config.omm_topology, mol_a, mol_b], f"{prefix}_simulation_" + str(lamb_idx) + ".pdb"
        )

        for frame in frames:
            host_frame = frame[:num_host_atoms]
            ligand_frame = frame[num_host_atoms:]
            mol_ab_frame = pdb_writer.convert_single_topology_mols(ligand_frame, st)
            writer.write_frame(np.concatenate([host_frame, mol_ab_frame]) * 10)

        writer.close()

        if lamb_idx > 0:

            prev_frames = all_frames[lamb_idx - 1]
            cur_frames = all_frames[lamb_idx]

            prev_boxes = all_boxes[lamb_idx - 1]
            cur_boxes = all_boxes[lamb_idx]

            prev_U_fn = batch_U_fns[lamb_idx - 1]
            cur_U_fn = batch_U_fns[lamb_idx]

            kT = BOLTZ * 300.0
            beta = 1 / kT

            fwd_delta_u = beta * (cur_U_fn(prev_frames, prev_boxes) - prev_U_fn(prev_frames, prev_boxes))
            rev_delta_u = beta * (prev_U_fn(cur_frames, cur_boxes) - cur_U_fn(cur_frames, cur_boxes))

            dG_exact, exact_bar_err = pymbar.BAR(fwd_delta_u, rev_delta_u)
            dG_exact /= beta
            exact_bar_err /= beta

            plt.clf()
            plt.title(
                f"{prefix} lamb {lambda_schedule[lamb_idx-1]:.3f} -> {lambda_schedule[lamb_idx]:.3f}, dG: {dG_exact:.2f} +- {exact_bar_err:.2f} kJ/mol"
            )
            plt.hist(fwd_delta_u, alpha=0.5, label="fwd", density=True)
            plt.hist(-rev_delta_u, alpha=0.5, label="-rev", density=True)
            plt.legend()
            plt.savefig(f"{prefix}_lambda_{lamb_idx-1}_{lamb_idx}.png")

            all_dGs.append(dG_exact)
            all_errs.append(exact_bar_err)

            print(
                f"{prefix} BAR: lambda {lambda_schedule[lamb_idx-1]:.3f} -> {lambda_schedule[lamb_idx]:.3f} dG: {dG_exact:.3f} dG_err: {exact_bar_err:.3f}"
            )

    print(f"{prefix} total dG: {np.sum(all_dGs):.3f} +- {np.linalg.norm(all_errs):.3f} kJ/mol")


def hif2a_pair():

    st = get_hif2a_ligand_pair_single_topology()
    mol_a = st.mol_a
    mol_b = st.mol_b
    core = st.core
    forcefield = st.ff

    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(box_width)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_top, solvent_conf, solvent_box)

    estimate_relative_free_energy(mol_a, mol_b, core, forcefield, solvent_host_config, n_frames=100, prefix="solvent")

    complex_sys, complex_conf, _, _, complex_box, complex_top = builders.build_protein_system(
        "tests/data/hif2a_nowater_min.pdb"
    )
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    complex_host_config = HostConfig(complex_sys, complex_top, complex_conf, complex_box)

    estimate_relative_free_energy(mol_a, mol_b, core, forcefield, complex_host_config, n_frames=100, prefix="complex")


def get_mol_by_name(mols, name):
    for m in mols:
        if m.GetProp("_Name") == name:
            return m

    assert 0, "Mol not found"


from rdkit.Chem import AllChem, Draw

from timemachine.fe import atom_mapping


def plot_atom_mapping_grid(mol_a, mol_b, core_smarts, core, show_idxs=False):
    mol_a_2d = Chem.Mol(mol_a)
    mol_b_2d = Chem.Mol(mol_b)
    mol_q_2d = Chem.MolFromSmarts(core_smarts)

    AllChem.Compute2DCoords(mol_q_2d)

    q_to_a = [[int(x[0]), int(x[1])] for x in enumerate(core[:, 0])]
    q_to_b = [[int(x[0]), int(x[1])] for x in enumerate(core[:, 1])]

    AllChem.GenerateDepictionMatching2DStructure(mol_a_2d, mol_q_2d, atomMap=q_to_a)
    AllChem.GenerateDepictionMatching2DStructure(mol_b_2d, mol_q_2d, atomMap=q_to_b)

    atom_colors_a = {}
    atom_colors_b = {}
    atom_colors_q = {}
    for c_idx, ((a_idx, b_idx), rgb) in enumerate(zip(core, np.random.random((len(core), 3)))):
        atom_colors_a[int(a_idx)] = tuple(rgb.tolist())
        atom_colors_b[int(b_idx)] = tuple(rgb.tolist())
        atom_colors_q[int(c_idx)] = tuple(rgb.tolist())

    if show_idxs:
        for atom in mol_a_2d.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        for atom in mol_b_2d.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        for atom in mol_q_2d.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

    # print(list(range(mol_q_2d.GetNumAtoms())))
    # print(core[:, 0].tolist())

    return Draw.MolsToGridImage(
        [mol_q_2d, mol_a_2d, mol_b_2d],
        molsPerRow=3,
        highlightAtomLists=[list(range(mol_q_2d.GetNumAtoms())), core[:, 0].tolist(), core[:, 1].tolist()],
        highlightAtomColors=[atom_colors_q, atom_colors_a, atom_colors_b],
        subImgSize=(400, 400),
        legends=["core", mol_a.GetProp("_Name"), mol_b.GetProp("_Name")],
        useSVG=True,
    )


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

    args = parser.parse_args()
    mols = [mol for mol in Chem.SDMolSupplier(str(args.ligands), removeHs=False)]
    mol_a = get_mol_by_name(mols, args.mol_a_name)  # 43 in test pair
    mol_b = get_mol_by_name(mols, args.mol_b_name)  # 30 in test pair

    print("Searching for the maximum common substructure...")
    mcs_threshold = 0.75
    mcs_result = atom_mapping.mcs_map(mol_a, mol_b, threshold=mcs_threshold)
    query_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    print("mol_a SMILES:", Chem.MolToSmiles(mol_a, isomericSmiles=False))
    print("mol_b SMILES:", Chem.MolToSmiles(mol_b, isomericSmiles=False))
    print("core SMARTS:", mcs_result.smartsString)

    core = atom_mapping.get_core_by_mcs(mol_a, mol_b, query_mol, threshold=mcs_threshold)
    print("core mapping:", core.tolist())

    res = plot_atom_mapping_grid(mol_a, mol_b, mcs_result.smartsString, core)
    with open(f"atom_mapping_{args.mol_a_name}_to_{args.mol_b_name}.svg", "w") as fh:
        fh.write(res)

    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(box_width)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_top, solvent_conf, solvent_box)

    forcefield = Forcefield.load_from_file(args.forcefield)

    estimate_relative_free_energy(mol_a, mol_b, core, forcefield, solvent_host_config, n_frames=100, prefix="solvent")

    complex_sys, complex_conf, _, _, complex_box, complex_top = builders.build_protein_system(args.protein)
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    complex_host_config = HostConfig(complex_sys, complex_top, complex_conf, complex_box)

    estimate_relative_free_energy(mol_a, mol_b, core, forcefield, complex_host_config, n_frames=100, prefix="complex")


if __name__ == "__main__":

    if len(sys.argv) == 1:
        hif2a_pair()
    else:
        read_from_args()
