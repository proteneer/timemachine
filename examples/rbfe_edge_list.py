import argparse
import csv
from typing import List

from simtk.openmm import app

from timemachine.constants import KCAL_TO_KJ
from timemachine.fe import rbfe
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
    parser.add_argument("--results_csv", type=str, help="Results containing all the csv", required=True)
    parser.add_argument("--forcefield", type=str, help="location of the ligand forcefield", required=True)
    parser.add_argument("--protein", type=str, help="PDB of the protein complex", required=True)
    parser.add_argument("--n_gpus", type=int, help="number of gpus", required=True)
    parser.add_argument("--seed", type=int, help="random seed for the runs", required=True)

    return parser.parse_args()


def read_edges_csv(csv_file: str) -> List[rbfe.Edge]:
    with open(args.results_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)  # skip header
        return [
            rbfe.Edge(
                mol_a_name,
                mol_b_name,
                {
                    "exp_ddg_kcal": float(exp_ddg) * KCAL_TO_KJ,
                    "fep_ddg_kcal": float(fep_ddg) * KCAL_TO_KJ,
                    "fep_ddg_err_kcal": float(fep_ddg_err) * KCAL_TO_KJ,
                    "ccc_ddg_kcal": float(ccc_ddg) * KCAL_TO_KJ,
                    "ccc_ddg_err_kcal": float(ccc_ddg_err) * KCAL_TO_KJ,
                },
            )
            for mol_a_name, mol_b_name, exp_ddg, fep_ddg, fep_ddg_err, ccc_ddg, ccc_ddg_err in reader
        ]


if __name__ == "__main__":
    args = parse_args()

    ligands = read_sdf(args.ligands)
    edges = read_edges_csv(args.results_csv)
    forcefield = Forcefield.load_from_file(args.forcefield)
    protein = app.PDBFile(str(args.protein))

    paths = rbfe.run_edges_parallel(
        args.n_frames,
        ligands,
        edges,
        forcefield,
        args.protein,
        args.n_gpus,
        args.seed,
    )
    for path in paths:
        print(path)
