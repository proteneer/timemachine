import argparse
import csv
from dataclasses import replace
from typing import List

from openmm import app

from timemachine.constants import KCAL_TO_KJ
from timemachine.fe import rbfe
from timemachine.fe.free_energy import WaterSamplingParams
from timemachine.fe.utils import read_sdf
from timemachine.ff import Forcefield


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


def read_edges_csv(path: str) -> List[rbfe.Edge]:
    with open(path) as fp:
        reader = csv.reader(fp, delimiter=",")
        next(reader, None)  # skip header
        return [
            rbfe.Edge(
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

    _ = rbfe.run_edges_parallel(
        ligands,
        edges,
        forcefield,
        protein,
        args.n_gpus,
        md_params=md_params,
        n_windows=args.n_windows,
    )
