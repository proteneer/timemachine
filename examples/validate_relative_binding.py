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
import numpy as np
from jax import numpy as jnp

from fe.free_energy_rabfe import construct_absolute_lambda_schedule, construct_conversion_lambda_schedule, get_romol_conf, setup_relative_restraints
from fe.utils import convert_uIC50_to_kJ_per_mole
# from fe import model_abfe, model_rabfe, model_conversion
from fe import model_rabfe
from md import builders

from testsystems.relative import hif2a_ligand_pair

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers.serialize import serialize_handlers
from ff.handlers.nonbonded import AM1CCCHandler, LennardJonesHandler
from parallel.client import CUDAPoolClient, GRPCClient
from parallel.utils import get_gpu_count

from typing import Union, Optional, Iterable, Any, Tuple, Dict

from optimize.step import truncated_step

import multiprocessing
from training.dataset import Dataset
from rdkit import Chem

from timemachine.potentials import rmsd
from md import builders, minimizer

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description="Relatively absolute Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    path_to_ligand = 'tests/data/ligands_40.sdf'
    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)

    with open('ff/params/smirnoff_1_1_0_ccc.py') as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)
    mols = [x for x in suppl]

    dataset = Dataset(mols)

    # construct lambda schedules for complex and solvent
    complex_absolute_schedule = construct_absolute_lambda_schedule(cmd_args.num_complex_windows)
    solvent_absolute_schedule = construct_absolute_lambda_schedule(cmd_args.num_solvent_windows)

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, complex_topology = builders.build_protein_system(
        'tests/data/hif2a_nowater_min.pdb')

    solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

    # pick the largest mol as the blocker
    largest_size = 0
    ref_mol = None
    for mol in mols:
        if mol.GetNumAtoms() > largest_size:
            largest_size = mol.GetNumAtoms()
            ref_mol = mol

    print("Reference Molecule:", ref_mol.GetProp("_Name"), Chem.MolToSmiles(ref_mol))

    temperature = 300.0
    pressure = 1.0
    dt = 2.5e-3

    # Generate an equilibrated reference structure to use.
    print("Equilibrating reference molecule in the complex.")
    complex_ref_x0, complex_ref_box0 = minimizer.equilibrate_complex(
        ref_mol,
        complex_system,
        complex_coords,
        temperature,
        pressure,
        forcefield,
        complex_box,
        cmd_args.num_complex_preequil_steps
    )

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

    def pred_fn(params, mol, mol_ref):

        # generate the core_idxs
        core_idxs = setup_relative_restraints(mol, mol_ref)
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

        # compute the free energy of conversion in complex
        complex_conversion_x0 = minimizer.minimize_host_4d([mol], complex_system, complex_host_coords, forcefield, complex_box0, [aligned_mol_coords])
        complex_conversion_x0 = np.concatenate([complex_conversion_x0, aligned_mol_coords])
        dG_complex_conversion, dG_complex_conversion_error = binding_model_complex_conversion.predict(
            params,
            mol,
            complex_conversion_x0,
            complex_box0,
            prefix='complex_conversion_'+str(epoch))

        print("dG_complex_conversion", dG_complex_conversion)

        return 0.0, 0.0

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
            prefix='complex_decouple_'+str(epoch))

        # effective free energy of removing from complex
        dG_complex = dG_complex_conversion + dG_complex_decouple

        # solvent
        min_solvent_coords = minimizer.minimize_host_4d([mol], solvent_system, solvent_coords, forcefield, solvent_box)
        solvent_x0 = np.concatenate([min_solvent_coords, mol_coords])
        solvent_box0 = solvent_box
        dG_solvent_conversion, dG_solvent_conversion_error = binding_model_solvent_conversion.predict(
            params,
            mol,
            solvent_x0,
            solvent_box0,
            prefix='solvent_conversion_'+str(epoch)
        )
        dG_solvent_decouple, dG_solvent_decouple_error = binding_model_solvent_decouple.predict(
            params,
            mol,
            solvent_x0,
            solvent_box0,
            prefix='solvent_decouple_'+str(epoch),
        )

        # effective free energy of removing from solvent
        dG_solvent = dG_solvent_conversion + dG_solvent_decouple

        dG_err = np.sqrt(dG_complex_conversion_error**2 + dG_complex_decouple_error**2 + dG_solvent_conversion_error**2 + dG_solvent_decouple_error**2)

        # the final value we seek is the free energy of moving
        # from the solvent into the complex.

        return dG_solvent - dG_complex, dG_err


    for epoch in range(10):
        epoch_params = serialize_handlers(ordered_handles)
        # dataset.shuffle()
        for mol in dataset.data:

            if mol.GetProp("_Name") != '254':
                continue

            label_dG = convert_uIC50_to_kJ_per_mole(float(mol.GetProp("IC50[uM](SPA)")))

            print("processing mol", mol.GetProp("_Name"), "with binding dG", label_dG, "SMILES", Chem.MolToSmiles(mol))

            pred_dG = pred_fn(ordered_params, mol, ref_mol)
            print("epoch", epoch, "mol", mol.GetProp("_Name"), "pred", pred_dG, "label", label_dG)
