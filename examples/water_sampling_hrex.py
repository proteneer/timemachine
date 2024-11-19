# Water sampling script that uses HREX to generate fully-interacting end-state distributions
import argparse
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from water_sampling_common import DEFAULT_BB_RADIUS, compute_occupancy, get_initial_state

from timemachine.constants import DEFAULT_TEMP
from timemachine.fe.free_energy import HREXParams, InitialState, MDParams, SimulationResult
from timemachine.fe.plots import (
    plot_hrex_replica_state_distribution_heatmap,
    plot_hrex_swap_acceptance_rates_convergence,
    plot_hrex_transition_matrix,
)
from timemachine.fe.rbfe import estimate_relative_free_energy_bisection_hrex_impl
from timemachine.ff import Forcefield


def estimate_relative_free_energy_hrex_bb(
    mol: Chem.rdchem.Mol,
    water_pdb,
    seed,
    lambda_min,
    lambda_max,
    n_windows,
    use_hmr,
    ff: Forcefield,
    nb_cutoff: float,
    md_params: MDParams,
) -> SimulationResult:
    temperature = DEFAULT_TEMP

    combined_prefix = "hrex"

    def make_optimized_initial_state_fn(lamb: float) -> InitialState:
        state, _, _ = get_initial_state(water_pdb, mol, ff, seed, nb_cutoff, use_hmr, lamb)
        return state

    return estimate_relative_free_energy_bisection_hrex_impl(
        temperature,
        lambda_min,
        lambda_max,
        md_params,
        n_windows,
        make_optimized_initial_state_fn,
        combined_prefix,
        min_overlap=0.667,
    )


def plot_hrex_water_transitions(
    sim_res: SimulationResult,
    lamb_schedule: List[float],
    ligand_idxs: List[int],
):
    plt.figure(figsize=(12, 6))
    for lamb_idx, lamb in enumerate(lamb_schedule):
        xs = sim_res.frames[lamb_idx]
        boxes = sim_res.boxes[lamb_idx]

        occs = []
        for x, b in zip(xs, boxes):
            occs.append(compute_occupancy(x, b, ligand_idxs, threshold=DEFAULT_BB_RADIUS) // 3)

        if lamb_idx == 0:
            lw = 3.0
            ls = "solid"
            alpha = 1.0
        else:
            lw = 1.0
            ls = "dotted"
            alpha = 0.20

        plt.plot(occs, label=f"{lamb:.2f}", alpha=alpha, linewidth=lw, linestyle=ls)
        unique, counts = np.unique(occs, return_counts=True)
        print("raw occs", occs)
        print("lambda", lamb, "occupancies", dict(zip(unique, counts)))

    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("occupancy")


def plot_and_save(f, fname, *args, **kwargs):
    """
    Given a function which generates a plot, saves plot to fname as a png.
    """
    plt.clf()
    f(*args, **kwargs)
    with open(fname, "wb") as fh:
        plt.savefig(fh, format="png", bbox_inches="tight")


def test_hrex():
    parser = argparse.ArgumentParser(
        description="Test Hamiltonian Replica Exchange to enhance water sampling inside a buckyball."
    )
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
    parser.add_argument(
        "--use_hmr",
        type=int,
        help="Whether or not we apply HMR. 1 for yes, 0 for no.",
        required=True,
    )

    args = parser.parse_args()

    print(" ".join(sys.argv))

    suppl = list(Chem.SDMolSupplier(args.ligand_sdf, removeHs=False))
    mol = suppl[0]

    ff = Forcefield.load_precomputed_default()
    seed = 2024
    np.random.seed(seed)

    nb_cutoff = 1.2  # this has to be 1.2 since the builders hard code this in (should fix later)

    # set up water indices, assumes that waters are placed at the front of the coordinates.
    dummy_initial_state, nwm, _ = get_initial_state(args.water_pdb, mol, ff, seed, nb_cutoff, args.use_hmr, 0.0)
    water_idxs = []
    for wai in range(nwm):
        water_idxs.append([wai * 3 + 0, wai * 3 + 1, wai * 3 + 2])
    water_idxs = np.array(water_idxs)

    print("number of ligand atoms", mol.GetNumAtoms())
    print("number of water atoms", nwm * 3)

    lambda_min = 0.0
    lambda_max = 0.4
    n_windows = 48

    hrex_params = HREXParams(n_frames_bisection=min(1000, args.n_frames))
    mdp = MDParams(n_frames=args.n_frames, n_eq_steps=10_000, steps_per_frame=400, seed=2023, hrex_params=hrex_params)
    print("hrex_params:", hrex_params)
    sim_res = estimate_relative_free_energy_hrex_bb(
        mol, args.water_pdb, seed, lambda_min, lambda_max, n_windows, args.use_hmr, ff, nb_cutoff, mdp
    )

    pair_bar_result = sim_res.final_result
    lamb_schedule = [state.lamb for state in pair_bar_result.initial_states]
    print("final overlaps", pair_bar_result.overlaps)
    print("final dGs", pair_bar_result.dGs)
    print("final dG_errs", pair_bar_result.dG_errs)

    plot_and_save(
        plot_hrex_water_transitions,
        "plot_hrex_water_transitions.png",
        sim_res,
        lamb_schedule,
        dummy_initial_state.ligand_idxs,
    )
    plot_and_save(
        plot_hrex_swap_acceptance_rates_convergence,
        "plot_hrex_swap_acceptance_rates_convergence.png",
        sim_res.hrex_diagnostics.cumulative_swap_acceptance_rates,
    )
    plot_and_save(
        plot_hrex_transition_matrix,
        "plot_hrex_transition_matrix.png",
        sim_res.hrex_diagnostics.transition_matrix,
    )
    plot_and_save(
        plot_hrex_replica_state_distribution_heatmap,
        "plot_hrex_replica_state_distribution_heatmap.png",
        sim_res.hrex_diagnostics.cumulative_replica_state_counts,
    )


if __name__ == "__main__":
    # example invocation:

    # start with 0 waters, using espaloma charges, with hmr, for 2000 frames
    # python -u examples/water_sampling_hrex.py --water_pdb timemachine/datasets/water_exchange/bb_0_waters.pdb --ligand_sdf timemachine/datasets/water_exchange/bb_centered_espaloma.sdf --n_frames 2000 --use_hmr 1
    test_hrex()
