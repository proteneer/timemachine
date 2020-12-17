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
from ff.handlers.deserialize import deserialize_handlers

import multiprocessing

from fe import free_energy


def wrap_method(args, fn):
    gpu_idx = args[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    return fn(*args[1:])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relative Hydration Free Energy Consistency Testing",
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
        help="number of equilibration lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_prod_steps",
        type=int,
        help="number of production lambda windows",
        required=True
    )


    cmd_args = parser.parse_args()

    multiprocessing.set_start_method('spawn') # CUDA runtime is not forkable
    pool = multiprocessing.Pool(cmd_args.num_gpus)

    suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

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

    # the water system first.
    complex_system, complex_coords, _, _, complex_box = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')
    complex_box += np.eye(3)*0.1 # BFGS this later

    # the water system first.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    solvent_box += np.eye(3)*0.1 # BFGS this later


    for label, host_system, host_coords, host_box, num_host_windows in [
        ("complex", complex_system, complex_coords, complex_box, cmd_args.num_complex_windows),
        ("solvent", solvent_system, solvent_coords, solvent_box, cmd_args.num_solvent_windows)]:

        A = int(.35*num_host_windows)
        B = int(.30*num_host_windows)
        C = num_host_windows - A - B

        lambda_schedule = np.concatenate([
            np.linspace(0.0,  0.25, A, endpoint=False),
            np.linspace(0.25, 0.75, B, endpoint=False),
            np.linspace(0.75, 1.0,  C, endpoint=True)
        ])

        assert len(lambda_schedule) == num_host_windows

        print("Minimizing the host structure to remove clashes.")
        minimized_host_coords = minimizer.minimize_host_4d(mol_a, host_system, host_coords, ff, host_box)

        rfe = free_energy.RelativeFreeEnergy(mol_a, mol_b, core, ff)

        # solvent leg
        host_args = []
        for lambda_idx, lamb in enumerate(lambda_schedule):
            gpu_idx = lambda_idx % cmd_args.num_gpus
            host_args.append((gpu_idx, lamb, host_system, minimized_host_coords, host_box, cmd_args.num_equil_steps, cmd_args.num_prod_steps))
        
        results = pool.map(functools.partial(wrap_method, fn=rfe.host_edge), host_args, chunksize=1)

        for lamb, (bonded_du_dl, nonbonded_du_dl) in zip(lambda_schedule, results):
            print("final", label, "lambda", lamb, "bonded:", bonded_du_dl[0], bonded_du_dl[1], "nonbonded:", nonbonded_du_dl[0], nonbonded_du_dl[1])

        dG_host = np.trapz([x[0][0]+x[1][0] for x in results], lambda_schedule)
        print("dG:", dG_host)

