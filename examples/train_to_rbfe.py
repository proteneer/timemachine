# This script computes the relative binding free energy of a single edge.

import os
import argparse
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem

from fe import topology
from md import builders
from md import minimizer

import functools

from ff import Forcefield
from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers import nonbonded

import multiprocessing

from fe import free_energy


def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593 * np.log(amount_in_uM * 1e-6) * 4.18


def wrap_method(args, fn):
    gpu_idx = args[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    return fn(*args[1:])


def run_epoch(ff, mol_a, mol_b, core):
    # build the protein system.
    complex_system, complex_coords, _, _, complex_box = builders.build_protein_system(
        'tests/data/hif2a_nowater_min.pdb')
    complex_box += np.eye(3) * 0.1  # BFGS this later

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    solvent_box += np.eye(3) * 0.1  # BFGS this later

    combined_handle_and_grads = {}
    stage_dGs = []

    for stage, host_system, host_coords, host_box, num_host_windows in [
        ("complex", complex_system, complex_coords, complex_box, cmd_args.num_complex_windows),
        ("solvent", solvent_system, solvent_coords, solvent_box, cmd_args.num_solvent_windows)]:

        A = int(.35 * num_host_windows)
        B = int(.30 * num_host_windows)
        C = num_host_windows - A - B

        # Emprically, we see the largest variance in std <du/dl> near the endpoints in the nonbonded
        # terms. Bonded terms are roughly linear. So we add more lambda windows at the endpoint to
        # help improve convergence.
        lambda_schedule = np.concatenate([
            np.linspace(0.0, 0.25, A, endpoint=False),
            np.linspace(0.25, 0.75, B, endpoint=False),
            np.linspace(0.75, 1.0, C, endpoint=True)
        ])

        assert len(lambda_schedule) == num_host_windows

        print("Minimizing the host structure to remove clashes.")
        minimized_host_coords = minimizer.minimize_host_4d(mol_a, host_system, host_coords, ff, host_box)

        rfe = free_energy.RelativeFreeEnergy(mol_a, mol_b, core, ff)

        # solvent leg
        host_args = []
        for lambda_idx, lamb in enumerate(lambda_schedule):
            gpu_idx = lambda_idx % cmd_args.num_gpus
            host_args.append((gpu_idx, lamb, host_system, minimized_host_coords, host_box, cmd_args.num_equil_steps,
                              cmd_args.num_prod_steps))

        results = pool.map(functools.partial(wrap_method, fn=rfe.host_edge), host_args, chunksize=1)

        ghs = []

        for lamb, (bonded_du_dl, nonbonded_du_dl, grads_and_handles) in zip(lambda_schedule, results):
            ghs.append(grads_and_handles)
            print("final", stage, "lambda", lamb, "bonded:", bonded_du_dl[0], bonded_du_dl[1], "nonbonded:",
                  nonbonded_du_dl[0], nonbonded_du_dl[1])

        dG_host = np.trapz([x[0][0] + x[1][0] for x in results], lambda_schedule)
        stage_dGs.append(dG_host)

        # use gradient information from the endpoints
        for (grad_lhs, handle_type_lhs), (grad_rhs, handle_type_rhs) in zip(ghs[0], ghs[-1]):
            assert handle_type_lhs == handle_type_rhs  # ffs are forked so the return handler isn't same object as that of ff
            grad = grad_rhs - grad_lhs
            # complex - solvent
            if handle_type_lhs not in combined_handle_and_grads:
                combined_handle_and_grads[handle_type_lhs] = grad
            else:
                combined_handle_and_grads[handle_type_lhs] -= grad

        print(stage, "pred_dG:", dG_host)

    pred = stage_dGs[0] - stage_dGs[1]

    loss = np.abs(pred - label)

    print("loss", loss, "pred", pred, "label", label)

    dl_dpred = np.sign(pred - label)

    # (ytz): these should be made configurable later on.
    gradient_clip_thresholds = {
        nonbonded.AM1CCCHandler: 0.05,
        nonbonded.LennardJonesHandler: np.array([0.001, 0])
    }

    # update gradients in place.
    # for handle_type, grad in combined_handle_and_grads.items():

    for handle_type, grad in combined_handle_and_grads.items():
        if handle_type in gradient_clip_thresholds:
            bounds = gradient_clip_thresholds[handle_type]
            dl_dp = dl_dpred * grad  # chain rule
            # lots of room to improve here.
            dl_dp = np.clip(dl_dp, -bounds, bounds)  # clip gradients so they're well behaved

            if handle_type == nonbonded.AM1CCCHandler:
                # sanity check as we have other charge methods that exist
                assert handle_type == type(ff.q_handle)
                ff.q_handle.params -= dl_dp

                # useful for debugging to dump out the grads
                # for smirks, dp in zip(ff.q_handle.smirks, dl_dp):
                # if np.any(dp) > 0:
                # print(smirks, dp)

            elif handle_type == nonbonded.LennardJonesHandler:
                # sanity check again, even though we don't have other lj methods currently
                assert handle_type == type(ff.lj_handle)
                ff.lj_handle.params -= dl_dp


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
    pool = multiprocessing.Pool(cmd_args.num_gpus)

    suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    # (ytz): these are *binding* free energies, i.e. values that are less than zero.
    label_dG_a = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp("IC50[uM](SPA)")))
    label_dG_b = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp("IC50[uM](SPA)")))

    print("binding dG_a", label_dG_a)
    print("binding dG_b", label_dG_b)

    label = label_dG_b - label_dG_a  # complex - solvent

    core = np.array([[0, 0],
                     [2, 2],
                     [1, 1],
                     [6, 6],
                     [5, 5],
                     [4, 4],
                     [3, 3],
                     [15, 16],
                     [16, 17],
                     [17, 18],
                     [18, 19],
                     [19, 20],
                     [20, 21],
                     [32, 30],
                     [26, 25],
                     [27, 26],
                     [7, 7],
                     [8, 8],
                     [9, 9],
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
    forcefield = Forcefield(ff_handlers)

    for epoch in range(100):
        run_epoch(forcefield, mol_a, mol_b, core)

        epoch_params = serialize_handlers(ff_handlers)

        # write ff parameters after each epoch
        with open("checkpoint_" + str(epoch) + ".py", 'w') as fh:
            fh.write(epoch_params)
