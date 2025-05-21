import pickle
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Literal

import jax

# Enable 64 bit jax
jax.config.update("jax_enable_x64", True)


import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

# This is needed for pickled mols to preserve their properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS, DEFAULT_FF
from timemachine.fe import atom_mapping
from timemachine.fe.free_energy import HREXParams, RESTParams, WaterSamplingParams
from timemachine.fe.plots import plot_as_png_fxn, plot_water_proposals_by_state
from timemachine.fe.rbfe import (
    DEFAULT_NUM_WINDOWS,
    HREXSimulationResult,
    MDParams,
    run_complex,
    run_solvent,
    run_vacuum,
)
from timemachine.fe.utils import get_mol_name, plot_atom_mapping_grid, read_sdf_mols_by_name
from timemachine.ff import Forcefield
from timemachine.md.exchange.utils import get_radius_of_mol_pair
from timemachine.parallel.client import AbstractFileClient, CUDAPoolClient, FileClient
from timemachine.parallel.utils import get_gpu_count


def run_leg(
    file_client: AbstractFileClient,
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    core: NDArray,
    leg_name: Literal["complex", "solvent", "vacuum"],
    ff: Forcefield,
    pdb_path: str | None,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
):
    """Run an RBFE leg (vacuum, solvent, or complex).

    Will store results using the file_client to a new directory that has the name of the leg being run.

    Stores the following files:

    * results.npz - Predictions, overlaps and the number of windows
    * lambda*_traj.npz - Store the endstate trajectories
    * simulation_result.pkl - Pickled copy of the SimulationResult object
    * host_config.pkl - Pickled HostConfig, if the leg is not vacuum
    * hrex_transition_matrix.png - PNG of the transition matrix plot
    * hrex_swap_acceptance_rates_convergence.png - PNG of the HREX swap rate convergence
    * hrex_replica_state_distribution_heatmap.png - PNG of the HREX replica state distribution heatmap

    Parameters
    ----------
    file_client : FileClient
        File client for storing results of the simulation
    mol_a : Chem.Mol
        First molecule in the system.
    mol_b : Chem.Mol
        Second molecule in the system.
    core : NDArray
        The atom mapping between the two molecules
    leg_name : str
        Name of the leg to run. Can be "vacuum", "solvent", or "complex".
    ff : Forcefield
        Forcefield
    pdb_path : str, optional
        Path to a PDB file if running a "complex" leg. Optional.
    md_params : MDParams
        Parameters for the RBFE simulation.
    n_windows : int
        Maximum number of windows to generate during bisection.
    min_overlap : float
        Minimum overlap used during bisection.
    """
    np.random.seed(md_params.seed)
    host_config = None
    if leg_name == "vacuum":
        res = run_vacuum(
            mol_a,
            mol_b,
            core,
            ff,
            None,
            md_params,
            n_windows=n_windows,
            min_overlap=min_overlap,
        )
    elif leg_name == "solvent":
        res, host_config = run_solvent(
            mol_a,
            mol_b,
            core,
            ff,
            None,
            md_params,
            n_windows=n_windows,
            min_overlap=min_overlap,
        )
    elif leg_name == "complex":
        assert pdb_path is not None
        res, host_config = run_complex(
            mol_a,
            mol_b,
            core,
            ff,
            str(Path(pdb_path).expanduser()),
            md_params,
            n_windows=n_windows,
            min_overlap=min_overlap,
        )
    else:
        assert 0, f"Invalid leg: {leg_name}"

    pred_dg = float(np.sum(res.final_result.dGs))
    pred_dg_err = float(np.linalg.norm(res.final_result.dG_errs))
    print(
        " | ".join(
            [
                f"{get_mol_name(mol_a)} -> {get_mol_name(mol_b)} (kJ/mol)",
                f"{leg_name} {pred_dg:.2f} +- {pred_dg_err:.2f}",
            ]
        ),
    )
    # Ensure the output directory exists
    Path(file_client.full_path(leg_name)).mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        file_client.full_path(Path(leg_name) / "results.npz"),
        pred_dg=pred_dg,
        pred_dg_err=pred_dg_err,
        overlaps=res.final_result.overlaps,
        n_windows=len(res.final_result.initial_states),
    )

    np.savez_compressed(
        file_client.full_path(Path(leg_name) / "lambda0_traj.npz"),
        coords=np.array(res.trajectories[0].frames),
        boxes=np.asarray(res.trajectories[0].boxes),
    )
    np.savez_compressed(
        file_client.full_path(Path(leg_name) / "lambda1_traj.npz"),
        coords=np.array(res.trajectories[-1].frames),
        boxes=np.asarray(res.trajectories[-1].boxes),
    )
    file_client.store(Path(leg_name) / "simulation_result.pkl", pickle.dumps(res))
    if host_config is not None:
        file_client.store(Path(leg_name) / "host_config.pkl", pickle.dumps(host_config))

    if isinstance(res, HREXSimulationResult):
        file_client.store(Path(leg_name) / "hrex_transition_matrix.png", res.hrex_plots.transition_matrix_png)
        file_client.store(
            Path(leg_name) / "hrex_swap_acceptance_rates_convergence.png",
            res.hrex_plots.swap_acceptance_rates_convergence_png,
        )
        file_client.store(
            Path(leg_name) / "hrex_replica_state_distribution_heatmap.png",
            res.hrex_plots.replica_state_distribution_heatmap_png,
        )
        if res.water_sampling_diagnostics is not None:
            file_client.store(
                Path(leg_name) / "water_sampling_acceptances.png",
                plot_as_png_fxn(
                    plot_water_proposals_by_state,
                    [state.lamb for state in res.final_result.initial_states],
                    res.water_sampling_diagnostics.cumulative_proposals_by_state,
                ),
            )


def main():
    parser = ArgumentParser(description="Run the RBFE legs for a pair of molecules")
    parser.add_argument("--sdf_path", help="Path to sdf file containing mols", required=True)
    parser.add_argument("--mol_a", help="Name of mol a in sdf_path", required=True)
    parser.add_argument("--mol_b", help="Name of mol b in sdf_path", required=True)
    parser.add_argument("--pdb_path", help="Path to pdb file containing structure")
    parser.add_argument("--n_eq_steps", default=200_000, type=int, help="Number of steps to perform equilibration")
    parser.add_argument("--n_frames", default=2000, type=int, help="Number of frames to generation")
    parser.add_argument("--steps_per_frame", default=400, type=int, help="Steps per frame")
    parser.add_argument(
        "--n_windows", default=DEFAULT_NUM_WINDOWS, type=int, help="Max number of windows from bisection"
    )
    parser.add_argument("--min_overlap", default=0.667, type=float, help="Overlap to target in bisection")
    parser.add_argument(
        "--target_overlap", default=0.667, type=float, help="Overlap to optimize final HREX schedule to"
    )
    parser.add_argument("--seed", default=2025, type=int, help="Seed")
    parser.add_argument("--legs", default=["vacuum", "solvent", "complex"], nargs="+")
    parser.add_argument("--forcefield", default=DEFAULT_FF)
    parser.add_argument(
        "--n_gpus", default=None, type=int, help="Number of GPUs to use, defaults to all GPUs if not provided"
    )
    parser.add_argument(
        "--water_sampling_padding",
        type=float,
        default=0.4,
        help="How much to expand the radius of the sphere used for water sampling (nm). Half of the largest intramolecular distance is used as the starting radius to which the padding is added: dist/2 + padding",
    )
    parser.add_argument(
        "--rest_max_temperature_scale",
        default=3.0,
        type=float,
        help="Maximum scale factor for the effective temperature of REST-softened interactions. Setting to 1.0 effectively disables REST.",
    )
    parser.add_argument(
        "--rest_temperature_scale_interpolation",
        default="exponential",
        type=str,
        help="Functional form to use for temperature scale interpolation in REST",
    )
    parser.add_argument(
        "--output_dir", default=None, help="Directory to output results, else generates a directory based on the time"
    )
    parser.add_argument("--custom_core_path", default=None, help="Path to custom core")
    args = parser.parse_args()

    if "complex" in args.legs:
        assert args.pdb_path is not None, "Must provide PDB to run complex leg"

    mols_by_name = read_sdf_mols_by_name(args.sdf_path)
    np.random.seed(args.seed)

    mol_a = mols_by_name[args.mol_a]
    mol_b = mols_by_name[args.mol_b]

    output_dir = args.output_dir
    if output_dir is None:
        date = datetime.now()
        date_str = date.strftime("%Y_%b_%d_%H_%M")
        output_dir = f"rbfe_{date_str}_{args.mol_a}_{args.mol_b}"
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_client = FileClient(dest_dir)

    ff = Forcefield.load_from_file(args.forcefield)

    mol_radius = get_radius_of_mol_pair(mol_a, mol_b)

    md_params = MDParams(
        n_eq_steps=args.n_eq_steps,
        n_frames=args.n_frames,
        steps_per_frame=args.steps_per_frame,
        seed=args.seed,
        hrex_params=HREXParams(
            optimize_target_overlap=args.target_overlap,
            rest_params=RESTParams(args.rest_max_temperature_scale, args.rest_temperature_scale_interpolation),
        ),
        water_sampling_params=WaterSamplingParams(radius=mol_radius + args.water_sampling_padding),
    )

    core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
    print("WARNING: using a custom core")
    # custom_core that demaps the phenyl ring
    core = np.stack([np.arange(28), np.arange(28)], axis=1)

    # Store top level data
    file_client.store("atom_mapping.svg", plot_atom_mapping_grid(mol_a, mol_b, core).encode("utf-8"))
    with open(file_client.full_path("md_params.pkl"), "wb") as ofs:
        pickle.dump(md_params, ofs)
    with open(file_client.full_path("core.pkl"), "wb") as ofs:
        pickle.dump(core, ofs)
    with open(file_client.full_path("ff.py"), "w") as ofs:
        ofs.write(ff.serialize())
    with Chem.SDWriter(file_client.full_path("mols.sdf")) as writer:
        writer.write(mol_a)
        writer.write(mol_b)

    num_gpus = args.n_gpus
    if num_gpus is None:
        num_gpus = get_gpu_count()

    pool = CUDAPoolClient(num_gpus)
    pool.verify()

    futures = []
    for leg_name in args.legs:
        fut = pool.submit(
            run_leg,
            file_client,
            mol_a,
            mol_b,
            core,
            leg_name,
            ff,
            args.pdb_path,
            md_params,
            args.n_windows,
            args.min_overlap,
        )
        futures.append(fut)
    for fut in futures:
        fut.result()


if __name__ == "__main__":
    main()
