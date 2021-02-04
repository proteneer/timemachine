# This script computes the relative binding free energy of a single edge.


import os
import argparse
import numpy as np
import jax

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem

from fe import topology
from fe import model
from md import builders

import functools

from ff import Forcefield
from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers import nonbonded
from parallel.client import CUDAPoolClient


import multiprocessing

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18


def wrap_method(args, fxn):
    # TODO: is there a more functools-y approach to make
    #   a function accept tuple instead of positional arguments?
    return fxn(*args)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relative Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus",
        required=True
    )

    parser.add_argument(
        "--num_complex_windows",
        type=int,
        help="number of vacuum lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_solvent_windows",
        type=int,
        help="number of solvent lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_equil_steps",
        type=int,
        help="number of equilibration steps for each lambda window",
        required=True
    )

    parser.add_argument(
        "--num_prod_steps",
        type=int,
        help="number of production steps for each lambda window",
        required=True
    )


    cmd_args = parser.parse_args()
    multiprocessing.set_start_method('spawn')  # CUDA runtime is not forkable

    client = CUDAPoolClient(max_workers=cmd_args.num_gpus)

    suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    # (ytz): these are *binding* free energies, i.e. values that are less than zero.
    label_dG_a = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp("IC50[uM](SPA)")))
    label_dG_b = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp("IC50[uM](SPA)")))

    print("binding dG_a", label_dG_a)
    print("binding dG_b", label_dG_b)

    label_ddG = label_dG_b - label_dG_a # complex - solvent

    core = np.array([[ 0,  0],
       [ 2,  2],
       [ 1,  1],
       [ 6,  6],
       [ 5,  5],
       [ 4,  4],
       [ 3,  3],
       [15, 16],
       [16, 17],
       [17, 18],
       [18, 19],
       [19, 20],
       [20, 21],
       [32, 30],
       [26, 25],
       [27, 26],
       [ 7,  7],
       [ 8,  8],
       [ 9,  9],
       [10, 10],
       [29, 11],
       [11, 12],
       [12, 13],
       [14, 15],
       [31, 29],
       [13, 14],
       [23, 24],
       [30, 28],
       [28, 27],
       [21, 22]]
    )

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    ff = Forcefield(ff_handlers)

    lambda_schedules = []

    for num_host_windows in [cmd_args.num_complex_windows, cmd_args.num_solvent_windows]:
        A = int(.35*num_host_windows)
        B = int(.30*num_host_windows)
        C = num_host_windows - A - B

        # Emprically, we see the largest variance in std <du/dl> near the endpoints in the nonbonded
        # terms. Bonded terms are roughly linear. So we add more lambda windows at the endpoint to
        # help improve convergence.
        lambda_schedule = np.concatenate([
            np.linspace(0.0,  0.25, A, endpoint=False),
            np.linspace(0.25, 0.75, B, endpoint=False),
            np.linspace(0.75, 1.0,  C, endpoint=True)
        ])

        assert len(lambda_schedule) == num_host_windows

        lambda_schedules.append(lambda_schedule)

    complex_schedule, solvent_schedule = lambda_schedules

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')
    complex_box += np.eye(3)*0.1 # BFGS this later

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    solvent_box += np.eye(3)*0.1 # BFGS this later

    # client = None

    binding_model = model.RBFEModel(
        client,
        ff,
        complex_system,
        complex_coords,
        complex_box,
        complex_schedule,
        solvent_system,
        solvent_coords,
        solvent_box,
        solvent_schedule,
        cmd_args.num_equil_steps,
        cmd_args.num_prod_steps
    )

    vg_fn = jax.value_and_grad(binding_model.loss, argnums=0)

    ordered_params = ff.get_ordered_params()
    ordered_handles = ff.get_ordered_handles()

    gradient_clip_thresholds = {
        nonbonded.AM1CCCHandler: 0.05,
        nonbonded.LennardJonesHandler: np.array([0.001,0])
    }

    for epoch in range(100):

        epoch_params = serialize_handlers(ff_handlers)
        loss, loss_grad = vg_fn(ordered_params, mol_a, mol_b, core, label_ddG)

        print("epoch", epoch, "loss", loss)

        for loss_grad, handle in zip(loss_grad, ordered_handles):
            assert handle.params.shape == loss_grad.shape

            if type(handle) in gradient_clip_thresholds:
                bounds = gradient_clip_thresholds[type(handle)]
                loss_grad = np.clip(loss_grad, -bounds, bounds)
                print("adjust handle", handle, "by", loss_grad)
                handle.params -= loss_grad

                # useful for debugging to dump out the grads
                # for smirks, dp in zip(handle.smirks, loss_grad):
                    # if np.any(dp) > 0:
                        # print(smirks, dp)

        # write ff parameters after each epoch
        with open("checkpoint_"+str(epoch)+".py", 'w') as fh:
            fh.write(epoch_params)
