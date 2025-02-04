import argparse
import csv
import pickle
import traceback
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, NamedTuple, Optional

import numpy as np
from openmm import app
from rdkit import Chem

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS, KCAL_TO_KJ
from timemachine.fe import atom_mapping, rbfe
from timemachine.fe.free_energy import MDParams, WaterSamplingParams
from timemachine.fe.utils import get_mol_name, read_sdf
from timemachine.ff import Forcefield
from timemachine.parallel.client import AbstractClient, AbstractFileClient, CUDAPoolClient, FileClient


class Edge(NamedTuple):
    mol_a_name: str
    mol_b_name: str
    metadata: dict[str, Any]


def get_failure_result_path(mol_a_name: str, mol_b_name: str):
    return f"failure_rbfe_result_{mol_a_name}_{mol_b_name}.pkl"


def get_success_result_path(mol_a_name: str, mol_b_name: str):
    return f"success_rbfe_result_{mol_a_name}_{mol_b_name}.pkl"


def run_edge_and_save_results(
    edge: Edge,
    mols: dict[str, Chem.rdchem.Mol],
    forcefield: Forcefield,
    protein: app.PDBFile,
    file_client: AbstractFileClient,
    n_windows: Optional[int],
    md_params: MDParams = rbfe.DEFAULT_MD_PARAMS,
):
    # Ensure that all mol props (e.g. _Name) are included in pickles
    # Without this get_mol_name(mol) will fail on roundtripped mol
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    edge_prefix = f"{edge.mol_a_name}_{edge.mol_b_name}"

    try:
        mol_a = mols[edge.mol_a_name]
        mol_b = mols[edge.mol_b_name]

        all_cores = atom_mapping.get_cores(
            mol_a,
            mol_b,
            **DEFAULT_ATOM_MAPPING_KWARGS,
        )
        core = all_cores[0]

        complex_res, complex_top, _ = rbfe.run_complex(
            mol_a,
            mol_b,
            core,
            forcefield,
            protein,
            md_params,
            n_windows=n_windows,
        )

        if isinstance(complex_res, rbfe.HREXSimulationResult):
            file_client.store(
                f"{edge_prefix}_complex_hrex_transition_matrix.png", complex_res.hrex_plots.transition_matrix_png
            )
            file_client.store(
                f"{edge_prefix}_complex_hrex_swap_acceptance_rates_convergence.png",
                complex_res.hrex_plots.swap_acceptance_rates_convergence_png,
            )
            file_client.store(
                f"{edge_prefix}_complex_hrex_replica_state_distribution_heatmap.png",
                complex_res.hrex_plots.replica_state_distribution_heatmap_png,
            )

        solvent_res, solvent_top, _ = rbfe.run_solvent(
            mol_a,
            mol_b,
            core,
            forcefield,
            protein,
            md_params,
            n_windows=n_windows,
        )
        if isinstance(solvent_res, rbfe.HREXSimulationResult):
            file_client.store(
                f"{edge_prefix}_solvent_hrex_transition_matrix.png", solvent_res.hrex_plots.transition_matrix_png
            )
            file_client.store(
                f"{edge_prefix}_solvent_hrex_swap_acceptance_rates_convergence.png",
                solvent_res.hrex_plots.swap_acceptance_rates_convergence_png,
            )
            file_client.store(
                f"{edge_prefix}_solvent_hrex_replica_state_distribution_heatmap.png",
                solvent_res.hrex_plots.replica_state_distribution_heatmap_png,
            )

    except Exception as err:
        print(
            "failed:",
            " | ".join(
                [
                    f"{edge.mol_a_name} -> {edge.mol_b_name} (kJ/mol)",
                    f"exp_ddg {edge.metadata['exp_ddg']:.2f}" if "exp_ddg" in edge.metadata else "",
                    (
                        f"fep_ddg {edge.metadata['fep_ddg']:.2f} +- {edge.metadata['fep_ddg_err']:.2f}"
                        if "fep_ddg" in edge.metadata and "fep_ddg_err" in edge.metadata
                        else ""
                    ),
                ]
            ),
        )

        path = get_failure_result_path(edge.mol_a_name, edge.mol_b_name)
        tb = traceback.format_exception(None, err, err.__traceback__)
        file_client.store(path, pickle.dumps((edge, err, tb)))

        print(err)
        traceback.print_exc()

        return file_client.full_path(path)

    path = get_success_result_path(edge.mol_a_name, edge.mol_b_name)
    pkl_obj = (mol_a, mol_b, edge.metadata, core, solvent_res, solvent_top, complex_res, complex_top)
    file_client.store(path, pickle.dumps(pkl_obj))

    solvent_ddg = sum(solvent_res.final_result.dGs)
    solvent_ddg_err = np.linalg.norm(solvent_res.final_result.dG_errs)
    complex_ddg = sum(complex_res.final_result.dGs)
    complex_ddg_err = np.linalg.norm(complex_res.final_result.dG_errs)

    tm_ddg = complex_ddg - solvent_ddg
    tm_err = np.linalg.norm([complex_ddg_err, solvent_ddg_err])

    print(
        "finished:",
        " | ".join(
            [
                f"{edge.mol_a_name} -> {edge.mol_b_name} (kJ/mol)",
                f"complex {complex_ddg:.2f} +- {complex_ddg_err:.2f}",
                f"solvent {solvent_ddg:.2f} +- {solvent_ddg_err:.2f}",
                f"tm_pred {tm_ddg:.2f} +- {tm_err:.2f}",
                f"exp_ddg {edge.metadata['exp_ddg']:.2f}" if "exp_ddg" in edge.metadata else "",
                (
                    f"fep_ddg {edge.metadata['fep_ddg']:.2f} +- {edge.metadata['fep_ddg_err']:.2f}"
                    if "fep_ddg" in edge.metadata and "fep_ddg_err" in edge.metadata
                    else ""
                ),
            ]
        ),
    )

    return file_client.full_path(path)


def run_edges_parallel(
    ligands: Sequence[Chem.rdchem.Mol],
    edges: Sequence[Edge],
    ff: Forcefield,
    protein: app.PDBFile,
    n_gpus: int,
    pool_client: Optional[AbstractClient] = None,
    file_client: Optional[AbstractFileClient] = None,
    md_params: MDParams = rbfe.DEFAULT_MD_PARAMS,
    n_windows: Optional[int] = None,
):
    mols = {get_mol_name(mol): mol for mol in ligands}

    pool_client = pool_client or CUDAPoolClient(n_gpus)
    pool_client.verify()

    file_client = file_client or FileClient()

    # Ensure that all mol props (e.g. _Name) are included in pickles
    # Without this get_mol_name(mol) will fail on roundtripped mol
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    jobs = [
        pool_client.submit(
            run_edge_and_save_results,
            edge,
            mols,
            ff,
            protein,
            file_client,
            n_windows,
            md_params,
        )
        for edge in edges
    ]

    # Remove references to completed jobs to allow garbage collection.
    # TODO: The current approach uses O(edges) memory in the worst case (e.g. if the first job gets stuck). Ideally we
    #   should process and remove references to jobs in the order they complete, but this would require an interface
    #   presently not implemented in our custom future classes.
    paths = []
    while jobs:
        job = jobs.pop(0)
        paths.append(job.result())

    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate relative free energy difference between complex and solvent given a results csv file."
    )
    parser.add_argument(
        "--n_frames", type=int, help="number of frames to use for the free energy estimate", required=True
    )
    parser.add_argument("--ligands", type=str, help="SDF file containing the ligands of interest", required=True)
    parser.add_argument("--results_csv", type=str, help="CSV containing the results and edges", required=True)
    parser.add_argument("--forcefield", type=str, help="location of the ligand forcefield", required=True)
    parser.add_argument("--protein", type=str, help="PDB of the protein complex", required=True)
    parser.add_argument("--n_gpus", type=int, help="number of gpus", required=True)
    parser.add_argument("--seed", type=int, help="random seed for the runs", required=True)
    parser.add_argument("--n_eq_steps", type=int, help="number of steps used for equilibration", required=False)
    parser.add_argument("--n_windows", type=int, help="number of lambda windows", required=False)
    parser.add_argument(
        "--water_sampling_interval", type=int, help="how often to run water sampling", required=False, default=None
    )

    return parser.parse_args()


def read_edges_csv(path: str) -> list[Edge]:
    with open(path) as fp:
        reader = csv.reader(fp, delimiter=",")
        next(reader, None)  # skip header
        return [
            Edge(
                mol_a_name,
                mol_b_name,
                {
                    "exp_ddg": float(exp_ddg_kcal) * KCAL_TO_KJ,
                    "fep_ddg": float(fep_ddg_kcal) * KCAL_TO_KJ,
                    "fep_ddg_err": float(fep_ddg_err_kcal) * KCAL_TO_KJ,
                    "ccc_ddg": float(ccc_ddg_kcal) * KCAL_TO_KJ,
                    "ccc_ddg_err": float(ccc_ddg_err_kcal) * KCAL_TO_KJ,
                },
            )
            for (
                mol_a_name,
                mol_b_name,
                exp_ddg_kcal,
                fep_ddg_kcal,
                fep_ddg_err_kcal,
                ccc_ddg_kcal,
                ccc_ddg_err_kcal,
            ) in reader
        ]


if __name__ == "__main__":
    args = parse_args()

    ligands = read_sdf(args.ligands)
    edges = read_edges_csv(args.results_csv)
    forcefield = Forcefield.load_from_file(args.forcefield)
    protein = app.PDBFile(str(args.protein))

    md_params = replace(rbfe.DEFAULT_HREX_PARAMS, n_frames=args.n_frames, n_eq_steps=args.n_eq_steps, seed=args.seed)
    if args.water_sampling_interval is not None and args.water_sampling_interval > 0:
        md_params = replace(md_params, water_sampling_params=WaterSamplingParams(interval=args.water_sampling_interval))

    _ = run_edges_parallel(
        ligands,
        edges,
        forcefield,
        protein,
        args.n_gpus,
        md_params=md_params,
        n_windows=args.n_windows,
    )
