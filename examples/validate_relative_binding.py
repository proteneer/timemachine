# This script runs the rabfe code in "inference" mode.
# There are multiple stages to this protocol:

# For the complex setup, we proceed as follows:
# 1) Conversion of the ligand parameters into a ff-independent state
# 2) Adding restraints to the ligand to the non-interacting "blocker" molecule
# 3) Compute the free energy of the swapping the ligand with the blocker
# 4) Release the restraints attached to the blocker
# Note that 2) and 4) are done directly via an endpoint correction.

# For the solvent setup, we proceed as follows:
# 1) Conversion of the ligand parameters into a ff-independent state.
# 2) Run an absolute hydration free energy of the ff-independent state.
import os
import pickle
import argparse
import numpy as np

from pathlib import Path

from fe.free_energy_rabfe import (
    construct_absolute_lambda_schedule_complex,
    construct_absolute_lambda_schedule_solvent,
    construct_conversion_lambda_schedule,
    setup_relative_restraints_by_distance,
)
from fe.utils import convert_uM_to_kJ_per_mole
from fe import model_rabfe
from fe.frames import endpoint_frames_only, all_frames

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from parallel.client import CUDAPoolClient, GRPCClient
from parallel.utils import get_gpu_count


from training.dataset import Dataset
from rdkit import Chem

import timemachine
from md import builders, minimizer

ALL_FRAMES = "all"
ENDPOINTS_ONLY = "endpoints"


def cache_wrapper(cache_path: str, fxn: callable, overwrite: bool = False) -> callable:
    """Given a path and a function, will either write the result of the function call
    to the file or read form the file. The output of the function is expected to be picklable.

    Parameters
    ----------
    cache_path: string
        Path to write pickled results to and to read results from


    fxn: callable
        Function whose result to cache. Used for expensive functions whose results
        should be stored between runs.

    overwrite: boolean
        Overwrite the cache if the file already exists. Useful is the cache file was incorrectly
        generated or the implementation of fxn has changed.

    Returns
    -------

    wrapped_function: callable
        A function with the same signature as the original fxn whose results will be stored to
        the cached file.
    """

    def fn(*args, **kwargs):
        if overwrite or not os.path.isfile(cache_path):
            print(f"Caching {fxn.__name__} result to {cache_path}")
            val = fxn(*args, **kwargs)
            with open(cache_path, "wb") as ofs:
                pickle.dump(val, ofs)
        else:
            with open(cache_path, "rb") as ifs:
                val = pickle.load(ifs)
        return val

    return fn


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relatively absolute Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--epochs", default=1, help="Number of Epochs", type=int)

    parser.add_argument(
        "--property_field",
        help="Property field to convert to kcals/mols",
        default=None,
    )

    parser.add_argument(
        "--property_units",
        help="must be either nM or uM",
        required=True,
        choices=["nM", "uM"],
    )

    parser.add_argument("--hosts", nargs="*", default=None, help="Hosts running GRPC worker to use for compute")

    parser.add_argument("--num_gpus", type=int, help="number of gpus", default=get_gpu_count())

    parser.add_argument(
        "--num_complex_conv_windows", type=int, help="number of lambda windows for complex conversion", required=True
    )

    parser.add_argument("--num_complex_windows", type=int, help="number of vacuum lambda windows", required=True)

    parser.add_argument(
        "--num_solvent_conv_windows", type=int, help="number of lambda windows for solvent conversion", required=True
    )

    parser.add_argument("--num_solvent_windows", type=int, help="number of solvent lambda windows", required=True)

    parser.add_argument(
        "--num_complex_equil_steps",
        type=int,
        help="number of equilibration steps for each complex lambda window",
        required=True,
    )

    parser.add_argument(
        "--num_complex_prod_steps",
        type=int,
        help="number of production steps for each complex lambda window",
        required=True,
    )

    parser.add_argument(
        "--num_solvent_equil_steps",
        type=int,
        help="number of equilibration steps for each solvent lambda window",
        required=True,
    )

    parser.add_argument(
        "--num_solvent_prod_steps",
        type=int,
        help="number of production steps for each solvent lambda window",
        required=True,
    )

    parser.add_argument(
        "--num_complex_preequil_steps",
        type=int,
        help="number of pre-equilibration steps for each complex lambda window",
        required=True,
    )

    parser.add_argument(
        "--blocker_name", type=str, help="Name of the ligand the sdf file to be used as a blocker", required=True
    )
    parser.add_argument("--frames_written", choices=[ALL_FRAMES, ENDPOINTS_ONLY], default=ENDPOINTS_ONLY)

    parser.add_argument("--protein_pdb", type=str, help="Path to the target pdb", required=True)

    parser.add_argument("--ligand_sdf", type=str, help="Path to the ligand's sdf", required=True)

    parser.add_argument("--shuffle", action="store_true", help="Shuffle ligand order")

    cmd_args = parser.parse_args()

    print("cmd_args", cmd_args)

    if not cmd_args.hosts:
        num_gpus = cmd_args.num_gpus
        # set up multi-GPU client
        client = CUDAPoolClient(max_workers=num_gpus)
    else:
        # Setup GRPC client
        print("Connecting to GRPC workers...")
        client = GRPCClient(hosts=cmd_args.hosts)
    client.verify()

    path_to_ligand = cmd_args.ligand_sdf
    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)
    root = Path(timemachine.__file__).parent.parent

    with open(root.joinpath("ff/params/smirnoff_1_1_0_ccc.py")) as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)
    mols = [x for x in suppl]

    dataset = Dataset(mols)

    # construct lambda schedules for complex and solvent
    complex_absolute_schedule = construct_absolute_lambda_schedule_complex(cmd_args.num_complex_windows)
    solvent_absolute_schedule = construct_absolute_lambda_schedule_solvent(cmd_args.num_solvent_windows)

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, complex_topology = builders.build_protein_system(
        cmd_args.protein_pdb
    )

    solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

    blocker_mol = None

    for mol in mols:
        if mol.GetProp("_Name") == cmd_args.blocker_name:
            # we should only have one copy.
            assert blocker_mol is None
            blocker_mol = mol

    assert blocker_mol is not None

    frame_filter = None
    if cmd_args.frames_written == ALL_FRAMES:
        frame_filter = all_frames
    elif cmd_args.frames_written == ENDPOINTS_ONLY:
        frame_filter = endpoint_frames_only
    assert frame_filter is not None, f"Unknown frame writing mode: {cmd_args.frames_written}"

    print("Reference Molecule:", blocker_mol.GetProp("_Name"), Chem.MolToSmiles(blocker_mol))

    temperature = 300.0
    pressure = 1.0
    dt = 2.5e-3

    cached_equil_path = "equil_{}_blocker_{}.pkl".format(
        os.path.basename(cmd_args.protein_pdb).split(".")[0], cmd_args.blocker_name
    ).replace(" ", "_")

    # Generate an equilibrated reference structure to use.
    complex_ref_x0, complex_ref_box0 = cache_wrapper(cached_equil_path, minimizer.equilibrate_complex)(
        blocker_mol,
        complex_system,
        complex_coords,
        temperature,
        pressure,
        forcefield,
        complex_box,
        cmd_args.num_complex_preequil_steps,
    )

    rabfe_model = model_rabfe.RABFEModel(
        client,
        forcefield,
        blocker_mol,
        complex_system,
        complex_topology,
        solvent_system,
        solvent_topology,
        temperature,
        pressure,
        dt,
        complex_ref_x0,
        complex_ref_box0,
        solvent_coords,
        solvent_box,
        frame_filter=frame_filter,
    )

    # Setup Complex Windows
    rabfe_model.set_complex_conversion_schedule(construct_conversion_lambda_schedule(cmd_args.num_complex_conv_windows))
    rabfe_model.set_complex_decouple_schedule(construct_absolute_lambda_schedule_complex(cmd_args.num_complex_windows))

    # Setup Solvent Windows
    rabfe_model.set_solvent_conversion_schedule(construct_conversion_lambda_schedule(cmd_args.num_solvent_conv_windows))
    rabfe_model.set_solvent_decouple_schedule(construct_absolute_lambda_schedule_solvent(cmd_args.num_solvent_windows))

    # Setup Complex Steps
    rabfe_model.set_complex_decouple_equil_steps(cmd_args.num_complex_equil_steps)
    rabfe_model.set_complex_decouple_prod_steps(cmd_args.num_complex_prod_steps)
    rabfe_model.set_complex_conversion_equil_steps(cmd_args.num_complex_equil_steps)
    rabfe_model.set_complex_conversion_prod_steps(cmd_args.num_complex_prod_steps)

    rabfe_model.set_solvent_decouple_equil_steps(cmd_args.num_solvent_equil_steps)
    rabfe_model.set_solvent_decouple_prod_steps(cmd_args.num_solvent_prod_steps)
    rabfe_model.set_solvent_conversion_equil_steps(cmd_args.num_solvent_equil_steps)
    rabfe_model.set_solvent_conversion_prod_steps(cmd_args.num_solvent_prod_steps)

    ordered_params = forcefield.get_ordered_params()

    def simulate_pair(epoch: int, blocker: Chem.Mol, mol: Chem.Mol):
        mol_name = mol.GetProp("_Name")

        suffix = f"{mol_name}_{epoch}"

        # generate the core_idxs
        core_idxs = setup_relative_restraints_by_distance(mol, blocker)
        results = rabfe_model.simulate_futures(
            ordered_params,
            mol,
            core_idxs,
            suffix,
        )
        return results

    runs = []
    for epoch in range(cmd_args.epochs):
        if cmd_args.shuffle:
            dataset.shuffle()
        for mol in dataset.data:
            try:
                runs.append((epoch, simulate_pair(epoch, blocker_mol, mol)))
            except Exception as e:
                mol_name = mol.GetProp("_Name")
                print(f"Error simulating Mol: {mol_name}: {e}")
    for i in range(len(runs)):
        # Pop off futures to avoid accumulating memory.
        epoch, rabfe_future = runs.pop(0)
        mol = rabfe_future.mol
        mol_name = mol.GetProp("_Name")

        label_dG = "'N/A'"
        if cmd_args.property_field is not None:
            try:
                concentration = float(mol.GetProp(cmd_args.property_field))

                if cmd_args.property_units == "uM":
                    label_dG = convert_uM_to_kJ_per_mole(concentration)
                elif cmd_args.property_units == "nM":
                    label_dG = convert_uM_to_kJ_per_mole(concentration / 1000)
                else:
                    assert 0, "Unknown property units"
            except Exception as e:
                print(f"Unable to find property {cmd_args.property_field}: {e}")
        print(f"Epoch: {epoch}, Processing Mol: {mol_name}, Label: {label_dG}, Seed {rabfe_future.seed}")
        try:
            dG, dG_err = rabfe_model.predict_from_futures(rabfe_future)
            print(f"Epoch: {epoch}, Mol: {mol_name}, Predicted dG: {dG}, dG Err:" f" {dG_err}, Label: {label_dG}")
        except Exception as e:
            print(f"Error processing Mol: {mol_name}: {e}")
