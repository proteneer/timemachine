# This script validates the endpoint correction protocol on
# relative hydration free energies by comparing two absolute differences
# against the relative difference.

import argparse
import numpy as np

from fe.free_energy_rabfe import (
    construct_absolute_lambda_schedule_solvent,
    construct_relative_lambda_schedule,
    setup_relative_restraints_by_distance,
    get_romol_conf,
)
from fe import model_rabfe
from timemachine.md import builders, minimizer

from testsystems.relative import hif2a_ligand_pair

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from parallel.client import CUDAPoolClient, GRPCClient
from parallel.utils import get_gpu_count

import multiprocessing
from timemachine.training.dataset import Dataset
from rdkit import Chem

if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Absolute Hydration Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--hosts", nargs="*", default=None, help="Hosts running GRPC worker to use for compute")

    parser.add_argument("--num_gpus", type=int, help="number of gpus", default=get_gpu_count())

    parser.add_argument("--num_windows", type=int, help="number of solvent lambda windows", required=True)

    parser.add_argument(
        "--num_equil_steps",
        type=int,
        help="number of equilibration steps for each solvent lambda window",
        required=True,
    )

    parser.add_argument(
        "--num_prod_steps", type=int, help="number of production steps for each solvent lambda window", required=True
    )

    cmd_args = parser.parse_args()

    if not cmd_args.hosts:
        num_gpus = cmd_args.num_gpus
        # set up multi-GPU client
        client = CUDAPoolClient(max_workers=num_gpus)
    else:
        # Setup GRPC client
        print("Connecting to GRPC workers...")
        client = GRPCClient(hosts=cmd_args.hosts)
    client.verify()

    path_to_ligand = "tests/data/ligands_40.sdf"
    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)

    with open("ff/params/smirnoff_1_1_0_ccc.py") as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)
    mols = [x for x in suppl]

    dataset = Dataset(mols)

    absolute_solvent_schedule = construct_absolute_lambda_schedule_solvent(cmd_args.num_windows)
    relative_solvent_schedule = construct_relative_lambda_schedule(cmd_args.num_windows - 1)
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

    model_relative = model_rabfe.RelativeHydrationModel(
        client,
        forcefield,
        solvent_system,
        relative_solvent_schedule,
        solvent_topology,
        temperature,
        pressure,
        dt,
        cmd_args.num_equil_steps,
        cmd_args.num_prod_steps,
    )

    model_absolute = model_rabfe.AbsoluteHydrationModel(
        client,
        forcefield,
        solvent_system,
        absolute_solvent_schedule,
        solvent_topology,
        temperature,
        pressure,
        dt,
        cmd_args.num_equil_steps,
        cmd_args.num_prod_steps,
    )

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    M = len(dataset.data)

    # generate initial coordinates
    def minimize_absolute(mol):
        hc = minimizer.minimize_host_4d([mol], solvent_system, solvent_coords, forcefield, solvent_box)
        return np.concatenate([hc, get_romol_conf(mol)])

    def minimize_relative(mol_a, mol_b):
        hc = minimizer.minimize_host_4d([mol_a, mol_b], solvent_system, solvent_coords, forcefield, solvent_box)
        return np.concatenate([hc, get_romol_conf(mol_a), get_romol_conf(mol_b)])

    for epoch in range(100):

        for i in range(M):

            for j in range(i + 1, M):

                mol_a = dataset.data[i]
                mol_b = dataset.data[j]

                # relative calculation
                core_idxs = setup_relative_restraints_by_distance(mol_a, mol_b)

                xab = minimize_relative(mol_a, mol_b)
                ddG_ab, ddG_ab_err = model_relative.predict(
                    ordered_params,
                    mol_a,
                    mol_b,
                    core_idxs,
                    x0=xab,
                    box0=solvent_box,
                    prefix="epoch_"
                    + str(epoch)
                    + "_solvent_relative_"
                    + mol_a.GetProp("_Name")
                    + "_"
                    + mol_b.GetProp("_Name"),
                )

                # absolute calculation
                xa = minimize_absolute(mol_a)
                xb = minimize_absolute(mol_b)

                dG_a, dG_a_err = model_absolute.predict(
                    ordered_params, mol_a, x0=xa, box0=solvent_box, prefix="solvent_absolute_" + mol_a.GetProp("_Name")
                )
                dG_b, dG_b_err = model_absolute.predict(
                    ordered_params, mol_b, x0=xb, box0=solvent_box, prefix="solvent_absolute_" + mol_b.GetProp("_Name")
                )
                dG_ab_err = np.sqrt(dG_a_err ** 2 + dG_b_err ** 2)

                print(
                    f"mol_i {i} {mol_a.GetProp('_Name')} mol_j {j} {mol_b.GetProp('_Name')} ddG_ab {ddG_ab:.3f} +- {ddG_ab_err:.3f} dG_a-dG_b {dG_a-dG_b:.3f} +- {dG_ab_err:.3f}"
                )
