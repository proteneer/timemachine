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

from fe.free_energy_rabfe import construct_absolute_lambda_schedule_complex, construct_absolute_lambda_schedule_solvent, construct_conversion_lambda_schedule, get_romol_conf, setup_relative_restraints_using_smarts
from fe.utils import convert_uM_to_kJ_per_mole
from fe import model_rabfe
from fe.free_energy_rabfe import RABFEResult

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers.serialize import serialize_handlers
from parallel.client import CUDAPoolClient, GRPCClient
from parallel.utils import get_gpu_count

import multiprocessing
from training.dataset import Dataset
from rdkit import Chem

import timemachine
from timemachine.potentials import rmsd
from md import builders, minimizer
from rdkit.Chem import rdFMCS
from fe.atom_mapping import CompareDistNonterminal

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description="Relatively absolute Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--epochs",
        default=1,
        help="Number of Epochs",
        type=int
    )

    parser.add_argument(
        "--property_field",
        help="Property field to convert to kcals/mols",
        required=True
    )

    parser.add_argument(
        "--property_units",
        help="must be either nM or uM",
        required=True
    )

    parser.add_argument(
        "--hosts",
        nargs="*",
        default=None,
        help="Hosts running GRPC worker to use for compute"
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus",
        default=get_gpu_count()
    )

    parser.add_argument(
        "--num_complex_conv_windows",
        type=int,
        help="number of lambda windows for complex conversion",
        required=True
    )

    parser.add_argument(
        "--num_complex_windows",
        type=int,
        help="number of vacuum lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_solvent_conv_windows",
        type=int,
        help="number of lambda windows for solvent conversion",
        required=True
    )

    parser.add_argument(
        "--num_solvent_windows",
        type=int,
        help="number of solvent lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_complex_equil_steps",
        type=int,
        help="number of equilibration steps for each complex lambda window",
        required=True
    )

    parser.add_argument(
        "--num_complex_prod_steps",
        type=int,
        help="number of production steps for each complex lambda window",
        required=True
    )

    parser.add_argument(
        "--num_solvent_equil_steps",
        type=int,
        help="number of equilibration steps for each solvent lambda window",
        required=True
    )

    parser.add_argument(
        "--num_solvent_prod_steps",
        type=int,
        help="number of production steps for each solvent lambda window",
        required=True
    )

    parser.add_argument(
        "--num_complex_preequil_steps",
        type=int,
        help="number of pre-equilibration steps for each complex lambda window",
        required=True
    )

    parser.add_argument(
        "--blocker_name",
        type=str,
        help='Name of the ligand the sdf file to be used as a blocker',
        required=True
    )

    parser.add_argument(
        "--protein_pdb",
        type=str,
        help="Path to the target pdb",
        required=True
    )

    parser.add_argument(
        "--ligand_sdf",
        type=str,
        help="Path to the ligand's sdf",
        required=True
    )

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

    path_to_ligand =  cmd_args.ligand_sdf
    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)
    root = Path(timemachine.__file__).parent.parent

    with open(root.joinpath('ff/params/smirnoff_1_1_0_ccc.py')) as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)
    mols = [x for x in suppl]

    dataset = Dataset(mols)

    # construct lambda schedules for complex and solvent
    complex_absolute_schedule = construct_absolute_lambda_schedule_complex(cmd_args.num_complex_windows)
    solvent_absolute_schedule = construct_absolute_lambda_schedule_solvent(cmd_args.num_solvent_windows)

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, complex_topology = builders.build_protein_system(
        cmd_args.protein_pdb)

    solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

    # pick the largest mol as the blocker
    largest_size = 0
    blocker_mol = None

    for mol in mols:
        if mol.GetProp("_Name") == cmd_args.blocker_name:
            # we should only have one copy.
            assert blocker_mol is None
            blocker_mol = mol

    assert blocker_mol is not None

    print("Reference Molecule:", blocker_mol.GetProp("_Name"), Chem.MolToSmiles(blocker_mol))

    temperature = 300.0
    pressure = 1.0
    dt = 2.5e-3

    # Generate an equilibrated reference structure to use.
    print("Equilibrating reference molecule in the complex.")
    if not os.path.exists("equil.pickle"):
        complex_ref_x0, complex_ref_box0 = minimizer.equilibrate_complex(
            blocker_mol,
            complex_system,
            complex_coords,
            temperature,
            pressure,
            forcefield,
            complex_box,
            cmd_args.num_complex_preequil_steps
        )
        with open("equil.pickle", "wb") as ofs:
            pickle.dump((complex_ref_x0, complex_ref_box0), ofs)
    else:
        print("Loading existing pickle from cache")
        with open("equil.pickle", "rb") as ifs:
            complex_ref_x0, complex_ref_box0 = pickle.load(ifs)
    # complex models.
    complex_conversion_schedule = construct_conversion_lambda_schedule(cmd_args.num_complex_conv_windows)

    binding_model_complex_conversion = model_rabfe.AbsoluteConversionModel(
        client,
        forcefield,
        complex_system,
        complex_conversion_schedule,
        complex_topology,
        temperature,
        pressure,
        dt,
        cmd_args.num_complex_equil_steps,
        cmd_args.num_complex_prod_steps
    )

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
        cmd_args.num_complex_prod_steps
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
        cmd_args.num_solvent_prod_steps
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
        cmd_args.num_solvent_prod_steps
    )

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    mcs_params = rdFMCS.MCSParameters()
    mcs_params.AtomTyper = CompareDistNonterminal()
    mcs_params.BondTyper = rdFMCS.BondCompare.CompareAny

    def pred_fn(params, mol, mol_ref):

        result = rdFMCS.FindMCS(
            [mol, mol_ref],
            mcs_params
        )

        core_smarts = result.smartsString
        
        print("core_smarts", core_smarts)

        # generate the core_idxs
        core_idxs = setup_relative_restraints_using_smarts(mol, mol_ref, core_smarts)
        mol_coords = get_romol_conf(mol) # original coords
        
        num_complex_atoms = complex_coords.shape[0]

        # Use core_idxs to generate
        R, t = rmsd.get_optimal_rotation_and_translation(
            x1=complex_ref_x0[num_complex_atoms:][core_idxs[:, 1]], # reference core atoms
            x2=mol_coords[core_idxs[:, 0]], # mol core atoms
        )

        aligned_mol_coords = rmsd.apply_rotation_and_translation(mol_coords, R, t)

        ref_coords = complex_ref_x0[num_complex_atoms:]
        complex_host_coords = complex_ref_x0[:num_complex_atoms]
        complex_box0 = complex_ref_box0

        mol_name = mol.GetProp("_Name")

        # compute the free energy of conversion in complex
        complex_conversion_x0 = minimizer.minimize_host_4d([mol], complex_system, complex_host_coords, forcefield, complex_box0, [aligned_mol_coords])
        complex_conversion_x0 = np.concatenate([complex_conversion_x0, aligned_mol_coords])
        dG_complex_conversion, dG_complex_conversion_error = binding_model_complex_conversion.predict(
            params,
            mol,
            complex_conversion_x0,
            complex_box0,
            prefix='complex_conversion_'+str(epoch),
            core_idxs=core_idxs[:, 0]
        )

        # compute the free energy of swapping an interacting mol with a non-interacting reference mol
        complex_decouple_x0 = minimizer.minimize_host_4d([mol, mol_ref], complex_system, complex_host_coords, forcefield, complex_box0, [aligned_mol_coords, ref_coords])
        complex_decouple_x0 = np.concatenate([complex_decouple_x0, aligned_mol_coords, ref_coords])
        dG_complex_decouple, dG_complex_decouple_error = binding_model_complex_decouple.predict(
            params,
            mol,
            mol_ref,
            core_idxs,
            complex_decouple_x0,
            complex_box0,
            prefix='complex_decouple_'+mol_name+"_"+str(epoch))

        # solvent
        min_solvent_coords = minimizer.minimize_host_4d([mol], solvent_system, solvent_coords, forcefield, solvent_box)
        solvent_x0 = np.concatenate([min_solvent_coords, mol_coords])
        solvent_box0 = solvent_box
        dG_solvent_conversion, dG_solvent_conversion_error = binding_model_solvent_conversion.predict(
            params,
            mol,
            solvent_x0,
            solvent_box0,
            prefix='solvent_conversion_'+mol_name+"_"+str(epoch)
        )
        dG_solvent_decouple, dG_solvent_decouple_error = binding_model_solvent_decouple.predict(
            params,
            mol,
            solvent_x0,
            solvent_box0,
            prefix='solvent_decouple_'+mol_name+"_"+str(epoch),
        )

        rabfe_result = RABFEResult(
            mol_name=mol_name,
            dG_complex_conversion=dG_complex_conversion,
            dG_complex_decouple=dG_complex_decouple,
            dG_solvent_conversion=dG_solvent_conversion,
            dG_solvent_decouple=dG_solvent_decouple,
        )
        rabfe_result.log()

        dG_err = np.sqrt(dG_complex_conversion_error**2 + dG_complex_decouple_error**2 + dG_solvent_conversion_error**2 + dG_solvent_decouple_error**2)

        return rabfe_result.dG_bind, dG_err


    for epoch in range(cmd_args.epochs):
        epoch_params = serialize_handlers(ordered_handles)
        # dataset.shuffle()
        for mol in dataset.data:
            concentration = float(mol.GetProp(cmd_args.property_field))

            if cmd_args.property_units == 'uM':
                label_dG = convert_uM_to_kJ_per_mole(concentration)
            elif cmd_args.property_units == 'nM':
                label_dG = convert_uM_to_kJ_per_mole(concentration/1000)
            else:
                assert 0, "Unknown property units"

            print("processing mol", mol.GetProp("_Name"), "with binding dG", label_dG, "SMILES", Chem.MolToSmiles(mol))

            pred_dG = pred_fn(ordered_params, mol, blocker_mol)
            print("epoch", epoch, "mol", mol.GetProp("_Name"), "pred", pred_dG, "label", label_dG)
