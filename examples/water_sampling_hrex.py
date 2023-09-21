# Water sampling script that uses HREX to generate end-state distributions

# enable for 2x slow down
# from jax import config
# config.update("jax_enable_x64", True)

import argparse
import os
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from openmm import app
from rdkit import Chem

from timemachine.constants import AVOGADRO, DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe.free_energy import (
    AbsoluteFreeEnergy,
    HostConfig,
    InitialState,
    MDParams,
    SimulationResult,
    image_frames,
    make_pair_bar_plots,
    run_sims_bisection,
    run_sims_hrex,
)
from timemachine.fe.plots import (
    plot_hrex_replica_state_distribution,
    plot_hrex_replica_state_distribution_convergence,
    plot_hrex_replica_state_distribution_heatmap,
    plot_hrex_swap_acceptance_rates_convergence,
    plot_hrex_transition_matrix,
)
from timemachine.fe.topology import BaseTopology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.ff.handlers.nonbonded import PrecomputedChargeHandler
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.builders import strip_units
from timemachine.md.exchange import exchange_mover
from timemachine.md.states import CoordsVelBox

# uncomment if we want to re-enable minimization
# from timemachine.md.minimizer import minimize_host_4d


def estimate_relative_free_energy_hrex(
    mol: Chem.rdchem.Mol,
    water_pdb,
    seed,
    lambda_min,
    lambda_max,
    n_windows,
    ff: Forcefield,
    nb_cutoff: float,
    md_params: MDParams,
    n_frames_per_iter: int,
) -> SimulationResult:
    temperature = DEFAULT_TEMP

    combined_prefix = "hrex"

    def make_optimized_initial_state(lamb):
        state, _, _ = get_initial_state(water_pdb, mol, ff, seed, nb_cutoff, lamb)
        return state

    results_bisection, frames_by_state, boxes_by_state, final_velocities_by_state = run_sims_bisection(
        [lambda_min, lambda_max],
        make_optimized_initial_state,
        md_params,  # optimize?
        n_bisections=n_windows - 2,
        temperature=temperature,
        min_overlap=0.666,
    )

    result_bisection = results_bisection[-1]
    initial_states_hrex = result_bisection.initial_states
    lambda_schedule = [i.lamb for i in initial_states_hrex]

    print("Bisected Lambda Schedule", lambda_schedule)

    # Second phase: sample initial states determined by bisection using HREX
    initial_states_hrex = [
        replace(initial_state, x0=frames[-1], v0=final_velocities, box0=boxes[-1])
        for initial_state, frames, boxes, final_velocities in zip(
            results_bisection[-1].initial_states,
            frames_by_state,
            boxes_by_state,
            final_velocities_by_state,
        )
    ]

    pair_bar_result, frames_by_state, boxes_by_state, diagnostics = run_sims_hrex(
        initial_states_hrex,
        replace(md_params, n_eq_steps=0),  # using pre-equilibrated samples
        n_frames_per_iter=n_frames_per_iter,
        temperature=temperature,
    )

    plots = make_pair_bar_plots(pair_bar_result, temperature, combined_prefix)

    stored_frames = []
    stored_boxes = []
    keep_idxs = np.arange(len(initial_states_hrex))
    for i in keep_idxs:
        stored_frames.append(frames_by_state[i])
        stored_boxes.append(boxes_by_state[i])

    return SimulationResult(pair_bar_result, plots, stored_frames, stored_boxes, md_params, None, diagnostics)


def build_system(host_pdbfile: str, water_ff: str, padding: float):
    if isinstance(host_pdbfile, str):
        assert os.path.exists(host_pdbfile)
        host_pdb = app.PDBFile(host_pdbfile)
    elif isinstance(host_pdbfile, app.PDBFile):
        host_pdb = host_pdbfile
    else:
        raise TypeError("host_pdbfile must be a string or an openmm PDBFile object")

    host_coords = strip_units(host_pdb.positions)

    box_lengths = np.amax(host_coords, axis=0) - np.amin(host_coords, axis=0)

    box_lengths = box_lengths + padding
    box = np.eye(3, dtype=np.float64) * box_lengths

    host_ff = app.ForceField(f"{water_ff}.xml")
    nwa = len(host_coords)
    solvated_host_system = host_ff.createSystem(
        host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    solvated_host_coords = host_coords
    solvated_topology = host_pdb.topology

    return solvated_host_system, solvated_host_coords, box, solvated_topology, nwa


def get_initial_state(water_pdb, mol, ff, seed, nb_cutoff, lamb):
    assert nb_cutoff == 1.2  # hardcoded in prepare_host_edge

    # read water system
    solvent_sys, solvent_conf, solvent_box, solvent_topology, num_water_atoms = build_system(
        water_pdb, ff.water_ff, padding=0.1
    )
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes

    assert num_water_atoms == len(solvent_conf)

    bt = BaseTopology(mol, ff)
    afe = AbsoluteFreeEnergy(mol, bt)
    host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, num_water_atoms)
    potentials, params, combined_masses = afe.prepare_host_edge(ff.get_params(), host_config, lamb)

    ligand_idxs = np.arange(num_water_atoms, num_water_atoms + mol.GetNumAtoms())

    # modify state with all ligand atoms decoupled (returned by prepare_host_edge) to
    # state with only half of the ligand atoms decoupled. This roughly decouples a contingous
    # semi-sphere along the "wide" axis as opposed to form random small pores.
    nb_params = params[-1]

    final_masses = combined_masses
    final_conf = np.concatenate([solvent_conf, get_romol_conf(mol)], axis=0)

    component_dim = 4  # q,s,e,w
    num_atoms = len(final_conf)

    ligand_water_flat_idxs = np.s_[component_dim * num_atoms : 2 * component_dim * num_atoms]  # water
    ligand_protein_flat_idxs = np.s_[2 * component_dim * num_atoms : 3 * component_dim * num_atoms :]  # non-water
    ligand_water_params = nb_params[ligand_water_flat_idxs].reshape(-1, 4)
    ligand_protein_params = nb_params[ligand_protein_flat_idxs].reshape(-1, 4)

    fully_coupled_ligand_atoms = ligand_idxs[: len(ligand_idxs) // 2]

    ligand_water_params = ligand_water_params.at[fully_coupled_ligand_atoms, -1].set(0)
    ligand_protein_params = ligand_protein_params.at[fully_coupled_ligand_atoms, -1].set(0)

    nb_params = nb_params.at[ligand_water_flat_idxs].set(ligand_water_params.reshape(-1))
    nb_params = nb_params.at[ligand_protein_flat_idxs].set(ligand_protein_params.reshape(-1))

    # override last potential with the updated nb_params
    host_bps = []
    for p, bp in zip(params[:-1], potentials[:-1]):
        host_bps.append(bp.bind(p))

    host_bps.append(potentials[-1].bind(nb_params))

    # (YTZ): This is disabled because the initial ligand and starting waters are pre-minimized
    # print("Minimizing the host system...", end="", flush=True)
    # host_conf = minimize_host_4d([mol], host_config, ff)
    # print("Done", flush=True)

    temperature = DEFAULT_TEMP
    dt = 1.5e-3
    barostat_interval = 25
    integrator = LangevinIntegrator(temperature, dt, 1.0, final_masses, seed)
    bond_list = get_bond_list(host_bps[0].potential)
    group_idxs = get_group_indices(bond_list, len(final_masses))
    barostat = MonteCarloBarostat(
        len(final_masses), DEFAULT_PRESSURE, temperature, group_idxs, barostat_interval, seed + 1
    )

    initial_state = InitialState(
        host_bps, integrator, barostat, final_conf, np.zeros_like(final_conf), solvent_box, lamb, ligand_idxs
    )

    num_water_mols = num_water_atoms // 3

    return initial_state, num_water_mols, solvent_topology


def compute_density(n_waters, box):
    # returns density of water in kg/m^3
    box_vol = np.prod(np.diag(box))
    numerator = n_waters * 18.01528 * 1e27
    denominator = box_vol * AVOGADRO * 1000
    return numerator / denominator


def compute_occupancy(x_t, box_t, ligand_idxs, threshold):
    # compute the number of waters inside the buckyball ligand
    num_ligand_atoms = len(ligand_idxs)
    num_host_atoms = len(x_t) - num_ligand_atoms
    host_coords = x_t[:num_host_atoms]
    ligand_coords = x_t[num_host_atoms:]
    ligand_centroid = np.mean(ligand_coords, axis=0)
    count = 0
    diffs = exchange_mover.delta_r_np(ligand_centroid[None, :], host_coords, box_t)
    dijs = np.linalg.norm(diffs, axis=-1)
    count = np.sum(dijs < threshold)
    return count


def setup_forcefield():
    # use a precomputed charge handler on the ligand to avoid running AM1 on a buckyball
    ff = Forcefield.load_default()
    q_handle = PrecomputedChargeHandler()
    q_handle_intra = PrecomputedChargeHandler()
    q_handle_solv = PrecomputedChargeHandler()
    return Forcefield(
        ff.hb_handle,
        ff.ha_handle,
        ff.pt_handle,
        ff.it_handle,
        q_handle=q_handle,
        q_handle_solv=q_handle_solv,
        q_handle_intra=q_handle_intra,
        lj_handle=ff.lj_handle,
        lj_handle_solv=ff.lj_handle_solv,
        lj_handle_intra=ff.lj_handle_intra,
        protein_ff=ff.protein_ff,
        water_ff=ff.water_ff,
    )


def image_xvb(initial_state, xvb_t):
    new_coords = image_frames(initial_state, [xvb_t.coords], [xvb_t.box])[0]
    return CoordsVelBox(new_coords, xvb_t.velocities, xvb_t.box)


def test_hrex():
    parser = argparse.ArgumentParser(description="Test the exchange protocol in a box of water.")
    parser.add_argument("--water_pdb", type=str, help="Location of the water PDB", required=True)
    parser.add_argument(
        "--ligand_sdf",
        type=str,
        help="SDF file containing the ligand of interest. Disable to run bulk water.",
        required=True,
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        help="Number of frames to use for bisection and hrex",
        required=True,
    )

    args = parser.parse_args()

    print(" ".join(sys.argv))

    suppl = list(Chem.SDMolSupplier(args.ligand_sdf, removeHs=False))
    mol = suppl[0]

    ff = setup_forcefield()
    seed = 2024
    np.random.seed(seed)

    nb_cutoff = 1.2  # this has to be 1.2 since the builders hard code this in (should fix later)

    # set up water indices, assumes that waters are placed at the front of the coordinates.
    dummy_initial_state, nwm, _ = get_initial_state(args.water_pdb, mol, ff, seed, nb_cutoff, 0.0)
    water_idxs = []
    for wai in range(nwm):
        water_idxs.append([wai * 3 + 0, wai * 3 + 1, wai * 3 + 2])
    water_idxs = np.array(water_idxs)

    print("number of ligand atoms", mol.GetNumAtoms())
    print("number of water atoms", nwm * 3)
    bb_radius = 0.46

    lambda_min = 0.0
    lambda_max = 0.3
    n_windows = 48

    mdp = MDParams(n_frames=args.n_frames, n_eq_steps=10_000, steps_per_frame=400, seed=2023)
    sim_res = estimate_relative_free_energy_hrex(
        mol,
        args.water_pdb,
        seed,
        lambda_min,
        lambda_max,
        n_windows,
        ff,
        nb_cutoff,
        mdp,
        n_frames_per_iter=1,
    )

    pair_bar_result = sim_res.final_result
    lamb_schedule = [state.lamb for state in pair_bar_result.initial_states]
    print("final overlaps", pair_bar_result.overlaps)
    print("final dGs", pair_bar_result.dGs)
    print("final dG_errs", pair_bar_result.dG_errs)

    plt.close()
    plt.cla()
    plt.clf()
    for lamb_idx, lamb in enumerate(lamb_schedule):
        xs = sim_res.frames[lamb_idx]
        boxes = sim_res.boxes[lamb_idx]

        occs = []
        for x, b in zip(xs, boxes):
            occs.append(compute_occupancy(x, b, dummy_initial_state.ligand_idxs, threshold=bb_radius) // 3)

        if lamb_idx == 0:
            lw = 3.5
            alpha = 1.0
        else:
            lw = 1.0
            alpha = 0.25

        plt.plot(occs, label=f"{lamb:.2f}", alpha=alpha, linewidth=lw)
        unique, counts = np.unique(occs, return_counts=True)
        print("lambda", lamb, "occupancies", dict(zip(unique, counts)))

    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("occupancy")
    plt.show()

    plot_hrex_swap_acceptance_rates_convergence(
        sim_res.hrex_diagnostics.cumulative_swap_acceptance_rates, label="cumulative"
    )
    plot_hrex_swap_acceptance_rates_convergence(
        sim_res.hrex_diagnostics.instantaneous_swap_acceptance_rates, label="instantaneous"
    )
    plot_hrex_transition_matrix(sim_res.hrex_diagnostics.transition_matrix)
    plot_hrex_replica_state_distribution(sim_res.hrex_diagnostics.cumulative_replica_state_counts)
    plot_hrex_replica_state_distribution_convergence(sim_res.hrex_diagnostics.cumulative_replica_state_counts)
    plot_hrex_replica_state_distribution_heatmap(sim_res.hrex_diagnostics.cumulative_replica_state_counts)
    plt.show()


if __name__ == "__main__":
    # example invocation:

    # start with 0 waters, using espaloma charges
    # python examples/water_sampling_hrex.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered_espaloma.sdf --n_frames 2000
    test_hrex()
