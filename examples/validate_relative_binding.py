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

import argparse
import datetime
import logging
import os
import pickle

import numpy as np
from rdkit import Chem

from timemachine.fe import model_rabfe
from timemachine.fe.frames import all_frames, endpoint_frames_only, no_frames
from timemachine.fe.free_energy import RABFEResult, get_romol_conf
from timemachine.fe.lambda_schedule import (
    construct_absolute_lambda_schedule_complex,
    construct_absolute_lambda_schedule_solvent,
    construct_conversion_lambda_schedule,
)
from timemachine.fe.model_utils import verify_rabfe_pair
from timemachine.fe.restraints import setup_relative_restraints_by_distance
from timemachine.fe.utils import convert_uM_to_kJ_per_mole
from timemachine.ff import Forcefield
from timemachine.md import builders, minimizer
from timemachine.parallel.client import CUDAPoolClient, GRPCClient
from timemachine.parallel.utils import get_gpu_count
from timemachine.potentials import rmsd
from timemachine.training.dataset import Dataset

ALL_FRAMES = "all"
ENDPOINTS_ONLY = "endpoints"
NO_FRAMES = "none"

NANO_MOLAR = "nM"
MICRO_MOLAR = "uM"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


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
    default_output_path = f"rabfe_{datetime.datetime.utcnow().isoformat(timespec='seconds').replace(':', '_')}.sdf"

    parser = argparse.ArgumentParser(
        description="Relatively absolute Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output_path", default=default_output_path, help="Path to write out SDF")

    parser.add_argument("--epochs", default=1, help="Number of Epochs", type=int)

    parser.add_argument(
        "--property_field",
        help="Property field to convert to kcals/mols",
        default=None,
    )

    parser.add_argument(
        "--property_units",
        help="Units of the label, either uM or nM",
        default=MICRO_MOLAR,
        choices=[NANO_MOLAR, MICRO_MOLAR],
    )

    parser.add_argument("--hosts", nargs="*", default=None, help="Hosts running GRPC worker to use for compute")

    parser.add_argument("--num_gpus", type=int, help="number of gpus", default=get_gpu_count())

    parser.add_argument(
        "--num_complex_conv_windows", type=int, help="number of lambda windows for complex conversion", default=64
    )

    parser.add_argument("--num_complex_windows", type=int, help="number of vacuum lambda windows", default=63)

    parser.add_argument(
        "--num_solvent_conv_windows", type=int, help="number of lambda windows for solvent conversion", default=64
    )

    parser.add_argument("--num_solvent_windows", type=int, help="number of solvent lambda windows", default=64)

    parser.add_argument(
        "--num_complex_equil_steps",
        type=int,
        help="number of equilibration steps for each complex lambda window",
        default=50000,
    )

    parser.add_argument(
        "--num_complex_prod_steps",
        type=int,
        help="number of production steps for each complex lambda window",
        default=800000,
    )

    parser.add_argument(
        "--num_solvent_equil_steps",
        type=int,
        help="number of equilibration steps for each solvent lambda window",
        default=50000,
    )

    parser.add_argument(
        "--num_solvent_prod_steps",
        type=int,
        help="number of production steps for each solvent lambda window",
        default=800000,
    )

    parser.add_argument(
        "--num_host_preequil_steps",
        type=int,
        help="number of pre-equilibration steps for the host systems",
        default=200000,
    )

    parser.add_argument("--seed", default=None, type=int, help="Seed to run simulation with, default is random")

    parser.add_argument(
        "--blocker_name", type=str, help="Name of the ligand the sdf file to be used as a blocker", required=True
    )
    parser.add_argument("--frames_written", choices=[ALL_FRAMES, ENDPOINTS_ONLY, NO_FRAMES], default=ENDPOINTS_ONLY)

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

    seed = cmd_args.seed
    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)
    print("Simulation seed", seed)
    np.random.seed(seed)

    path_to_ligand = cmd_args.ligand_sdf
    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)

    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")
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
    elif cmd_args.frames_written == NO_FRAMES:
        frame_filter = no_frames
    assert frame_filter is not None, f"Unknown frame writing mode: {cmd_args.frames_written}"

    print("Reference Molecule:", blocker_mol.GetProp("_Name"), Chem.MolToSmiles(blocker_mol))

    temperature = 300.0
    pressure = 1.0
    dt = 2.5e-3

    cached_equil_complex_path = "equil_{}_blocker_{}.pkl".format(
        os.path.basename(cmd_args.protein_pdb).split(".")[0], cmd_args.blocker_name
    ).replace(" ", "_")
    cached_equil_solvent_path = "equil_solvent_blocker_{}.pkl".format(cmd_args.blocker_name).replace(" ", "_")

    # Generate an equilibrated reference structure to use.
    complex_ref_x0, complex_ref_box0 = cache_wrapper(cached_equil_complex_path, minimizer.equilibrate_host)(
        blocker_mol,
        complex_system,
        complex_coords,
        temperature,
        pressure,
        forcefield,
        complex_box,
        cmd_args.num_host_preequil_steps,
        seed=seed,
    )

    # Generate an equilibrated solvent box to use.
    solvent_ref_x0, solvent_ref_box0 = cache_wrapper(cached_equil_solvent_path, minimizer.equilibrate_host)(
        blocker_mol,
        solvent_system,
        solvent_coords,
        temperature,
        pressure,
        forcefield,
        solvent_box,
        cmd_args.num_host_preequil_steps,
        seed=seed,
    )

    # force constant for the harmonic core-restraints
    k_core = 30.0

    # complex models
    binding_model_complex_decouple = model_rabfe.RelativeBindingModel(
        client,
        forcefield,
        complex_system,
        complex_absolute_schedule,
        complex_topology,
        temperature,
        pressure,
        dt,
        cmd_args.num_complex_equil_steps,
        cmd_args.num_complex_prod_steps,
        frame_filter=frame_filter,
        k_core=k_core,
    )

    complex_conversion_schedule = construct_conversion_lambda_schedule(cmd_args.num_complex_conv_windows)
    binding_model_complex_conversion = model_rabfe.RelativeConversionModel(
        client,
        forcefield,
        complex_system,
        complex_absolute_schedule,
        complex_topology,
        temperature,
        pressure,
        dt,
        cmd_args.num_complex_equil_steps,
        cmd_args.num_complex_prod_steps,
        frame_filter=frame_filter,
        k_core=k_core,
    )

    # solvent models.
    solvent_conversion_schedule = construct_conversion_lambda_schedule(cmd_args.num_solvent_conv_windows)

    binding_model_solvent_conversion = model_rabfe.AbsoluteConversionModel(
        client,
        forcefield,
        solvent_system,
        solvent_conversion_schedule,
        solvent_topology,
        temperature,
        pressure,
        dt,
        cmd_args.num_solvent_equil_steps,
        cmd_args.num_solvent_prod_steps,
        frame_filter=frame_filter,
    )

    binding_model_solvent_decouple = model_rabfe.AbsoluteStandardHydrationModel(
        client,
        forcefield,
        solvent_system,
        solvent_absolute_schedule,
        solvent_topology,
        temperature,
        pressure,
        dt,
        cmd_args.num_solvent_equil_steps,
        cmd_args.num_solvent_prod_steps,
        frame_filter=frame_filter,
    )

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    def simulate_pair(epoch: int, blocker: Chem.Mol, mol: Chem.Mol):
        verify_rabfe_pair(mol, blocker)
        mol_name = mol.GetProp("_Name")

        # generate the core_idxs
        core_idxs = setup_relative_restraints_by_distance(mol, blocker)
        mol_coords = get_romol_conf(mol)  # original coords

        num_complex_atoms = complex_coords.shape[0]
        num_solvent_atoms = solvent_coords.shape[0]

        # Use core_idxs to generate
        R, t = rmsd.get_optimal_rotation_and_translation(
            x1=complex_ref_x0[num_complex_atoms:][core_idxs[:, 1]],  # reference core atoms
            x2=mol_coords[core_idxs[:, 0]],  # mol core atoms
        )

        aligned_mol_coords = rmsd.apply_rotation_and_translation(mol_coords, R, t)

        ref_coords = complex_ref_x0[num_complex_atoms:]
        complex_host_coords = complex_ref_x0[:num_complex_atoms]
        complex_box0 = complex_ref_box0

        solvent_host_coords = solvent_ref_x0[:num_solvent_atoms]
        solvent_box0 = solvent_ref_box0

        # compute the free energy of swapping an interacting mol with a non-interacting reference mol
        complex_decouple_x0 = minimizer.minimize_host_4d(
            [mol, blocker_mol],
            complex_system,
            complex_host_coords,
            forcefield,
            complex_box0,
            [aligned_mol_coords, ref_coords],
        )
        complex_decouple_x0 = np.concatenate([complex_decouple_x0, aligned_mol_coords, ref_coords])

        # compute the free energy of conversion in complex
        min_solvent_coords = minimizer.minimize_host_4d(
            [mol], solvent_system, solvent_host_coords, forcefield, solvent_box0
        )
        solvent_x0 = np.concatenate([min_solvent_coords, mol_coords])

        suffix = f"{mol_name}_{epoch}"

        seed = np.random.randint(np.iinfo(np.int32).max)

        # Order of these simulations should match the order in which predictions are computed to ensure
        # efficient use of parallelism.
        return {
            "solvent_conversion": binding_model_solvent_conversion.simulate_futures(
                ordered_params,
                mol,
                solvent_x0,
                solvent_box0,
                prefix="solvent_conversion_" + suffix,
                seed=seed,
            ),
            "solvent_decouple": binding_model_solvent_decouple.simulate_futures(
                ordered_params,
                mol,
                solvent_x0,
                solvent_box0,
                prefix="solvent_decouple_" + suffix,
                seed=seed,
            ),
            "complex_conversion": binding_model_complex_conversion.simulate_futures(
                ordered_params,
                mol,
                blocker_mol,
                core_idxs,
                complex_decouple_x0,
                complex_box0,
                prefix="complex_conversion_" + suffix,
                seed=seed,
            ),
            "complex_decouple": binding_model_complex_decouple.simulate_futures(
                ordered_params,
                mol,
                blocker_mol,
                core_idxs,
                complex_decouple_x0,
                complex_box0,
                prefix="complex_decouple_" + suffix,
                seed=seed,
            ),
            "mol": mol,
            "blocker": blocker,
            "epoch": epoch,
            "seed": seed,
        }

    def predict_dG(results: dict) -> RABFEResult:
        dG_solvent_conversion, dG_solvent_conversion_error = binding_model_solvent_conversion.predict_from_futures(
            results["solvent_conversion"][0],
            results["mol"],
            results["solvent_conversion"][1],
            results["solvent_conversion"][2],
        )
        dG_solvent_decouple, dG_solvent_decouple_error = binding_model_solvent_decouple.predict_from_futures(
            results["solvent_decouple"][0],
            results["mol"],
            results["solvent_decouple"][1],
            results["solvent_decouple"][2],
        )
        dG_complex_conversion, dG_complex_conversion_error = binding_model_complex_conversion.predict_from_futures(
            results["complex_conversion"][0],
            results["mol"],
            results["blocker"],
            results["complex_conversion"][1],
            results["complex_conversion"][2],
        )
        dG_complex_decouple, dG_complex_decouple_error = binding_model_complex_decouple.predict_from_futures(
            results["complex_decouple"][0],
            results["mol"],
            results["blocker"],
            results["complex_decouple"][1],
            results["complex_decouple"][2],
        )

        rabfe_result = RABFEResult(
            mol_name=mol_name,
            dG_complex_conversion=dG_complex_conversion,
            dG_complex_conversion_error=dG_complex_conversion_error,
            dG_complex_decouple=dG_complex_decouple,
            dG_complex_decouple_error=dG_complex_decouple_error,
            dG_solvent_conversion=dG_solvent_conversion,
            dG_solvent_conversion_error=dG_solvent_conversion_error,
            dG_solvent_decouple=dG_solvent_decouple,
            dG_solvent_decouple_error=dG_solvent_decouple_error,
        )
        rabfe_result.log()
        return rabfe_result

    runs = []
    for epoch in range(cmd_args.epochs):
        if cmd_args.shuffle:
            dataset.shuffle()
        for mol in dataset.data:
            try:
                runs.append(simulate_pair(epoch, blocker_mol, mol))
            except Exception:
                mol_name = mol.GetProp("_Name")
                logger.exception(f"Error simulating Mol: {mol_name}")
    with Chem.SDWriter(cmd_args.output_path) as writer:
        for i in range(len(runs)):
            # Pop off futures to avoid accumulating memory.
            run = runs.pop(0)
            mol = run["mol"]
            seed = run["seed"]
            epoch = run["epoch"]
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
            print(f"Epoch: {epoch}, Processing Mol: {mol_name}, Label: {label_dG}")
            try:
                result = predict_dG(run)
                print(
                    f"Epoch: {epoch}, Seed: {seed}, Mol: {mol_name}, Predicted dG: {result.dG_bind}, dG Err:"
                    f" {result.dG_bind_err}, Label: {label_dG}"
                )
            except Exception:
                logger.exception(f"Error processing Mol: {mol_name}")
                continue
            try:
                result.apply_to_mol(mol)
                mol.SetProp("Epoch", str(epoch))
                writer.write(mol)
                # Flush so that if the script fails, we still get mols
                writer.flush()
            except Exception:
                logger.exception("Failed to write mol")
