# run from a simulation state pickle to help with debugging

import argparse
import pickle

from timemachine.fe import rbfe


def read_from_args():

    parser = argparse.ArgumentParser(
        description="Estimate relative free energy difference between complex and solvent given a results csv file."
    )
    parser.add_argument("--pickle_path", type=str, help="path to pickle file", required=True)
    args = parser.parse_args()

    print("Processing pickle: ", args.pickle_path)

    with open(args.pickle_path, "rb") as fh:
        (meta, solvent_res, solvent_top, complex_res, complex_top) = pickle.load(fh)
        print("running solvent")
        for initial_state in solvent_res.initial_states:
            all_coords, all_boxes = rbfe.sample(initial_state, solvent_res.protocol)
        print("running complex")
        for initial_state in complex_res.initial_states:
            all_coords, all_boxes = rbfe.sample(initial_state, complex_res.protocol)


if __name__ == "__main__":
    read_from_args()
