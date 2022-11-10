import argparse

from timemachine.fe import rbfe


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


if __name__ == "__main__":
    args = parse_args()
    rbfe.run_parallel(
        args.n_frames,
        args.ligands,
        args.results_csv,
        args.forcefield,
        args.protein,
        args.n_gpus,
        args.seed,
    )
