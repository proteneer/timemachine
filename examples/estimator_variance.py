# Adapted from https://github.com/proteneer/timemachine/blob/8f4c6d009ff27e070c19ff16901082197c54494d/examples/rbfe_single.py

# without computing gradient (just computing dG_estimate), with default lambda protocol,
# what is the bias and variance of dG_estimate as a function of computational effort?

# computational effort will have two main tunable dials:
# * number of lambda windows
# * number of MD steps per lambda window

# (there are others, such as the amount of effort spent on initial minimization
# and equilibration)

# later, we can extend this script to look at
# * variance of the resulting gradient estimator
# * using an optimized lambda(t) protocol
# * using a different estimator than the one from TI


from functools import partial

# parallelization across multiple GPUs
from parallel.client import CUDAPoolClient

import os

import numpy as np
from rdkit import Chem

from fe import free_energy, topology
from fe.free_energy import construct_lambda_schedule
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from md import builders
from md import minimizer

from time import time

from pathlib import Path

root = Path(__file__).parent.parent
path_to_ligand = str(root.joinpath('tests/data/ligands_40.sdf'))
path_to_protein = str(root.joinpath('tests/data/hif2a_nowater_min.pdb'))
path_to_ff = str(root.joinpath('ff/params/smirnoff_1_1_0_ccc.py'))


def wrap_method(args, fxn):
    return fxn(*args)


from collections import namedtuple

RelativeTransformation = namedtuple("RelativeTransformation", ["ff", "mol_a", "mol_b", "core"])


def estimate_dG(transformation: RelativeTransformation,
                num_lambdas: int,
                num_steps_per_lambda: int,
                num_equil_steps: int,
                ):
    # build the protein system.
    complex_system, complex_coords, _, _, complex_box = builders.build_protein_system(
        path_to_protein)
    complex_box += np.eye(3) * 0.1  # BFGS this later

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    solvent_box += np.eye(3) * 0.1  # BFGS this later

    stage_dGs = []

    ff = transformation.ff
    mol_a, mol_b = transformation.mol_a, transformation.mol_b
    core = transformation.core

    # TODO: measure performance of complex and solvent separately

    lambda_schedule = construct_lambda_schedule(num_lambdas)

    for stage, host_system, host_coords, host_box in [
        ("complex", complex_system, complex_coords, complex_box),
        ("solvent", solvent_system, solvent_coords, solvent_box)]:

        print("Minimizing the host structure to remove clashes.")
        minimized_host_coords = minimizer.minimize_host_4d(mol_a, host_system, host_coords, ff, host_box)

        single_topology = topology.SingleTopology(mol_a, mol_b, core, ff)
        rfe = free_energy.RelativeFreeEnergy(single_topology)

        # solvent leg
        host_args = []
        for lambda_idx, lamb in enumerate(lambda_schedule):
            gpu_idx = lambda_idx % num_gpus
            host_args.append((gpu_idx, lamb, host_system, minimized_host_coords,
                              host_box, num_equil_steps, num_steps_per_lambda))

        # one GPU job per lambda window
        print('submitting tasks to client!')
        do_work = partial(wrap_method, fxn=rfe.host_edge)
        futures = []
        for lambda_idx, lamb in enumerate(lambda_schedule):
            arg = (lamb, host_system, minimized_host_coords, host_box, num_equil_steps, num_prod_steps)
            futures.append(client.submit(do_work, arg))

        results = []
        for fut in futures:
            results.append(fut.result())

        def _mean_du_dlambda(result):
            """summarize result of rfe.host_edge into mean du/dl

            TODO: refactor where this analysis step occurs
            """
            bonded_du_dl, nonbonded_du_dl, _ = result
            return np.mean(bonded_du_dl + nonbonded_du_dl)

        dG_host = np.trapz([_mean_du_dlambda(x) for x in results], lambda_schedule)
        stage_dGs.append(dG_host)

    pred = stage_dGs[0] - stage_dGs[1]
    return pred


if __name__ == "__main__":

    # command line argument parser generated the following variables:
    # num_gpus, num_complex_windows, num_solvent_windows, num_equil_steps,
    # num_prod_steps ...
    num_gpus = 10

    # TODO: can I just get all of the du/dlambda time-series information
    #   so that we can guess the variance of dG_estimate without having to run
    #   multiple times from scratch?
    num_repeats = 5

    client = CUDAPoolClient(max_workers=num_gpus)

    # TODO: move this test system constructor into a test fixture sort of thing

    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)
    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

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
    with open(path_to_ff) as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)

    transformation = RelativeTransformation(forcefield, mol_a, mol_b, core)

    arguments = dict(
        num_lambdas=60,
        num_steps_per_lambda=1000,
        num_equil_steps=1000,
    )

    num_steps_grid = [10,20,30,40,50,100,250,500,750,1000,2500,5000,10000,50000]

    results = dict()
    times = dict()

    for num_steps in num_steps_grid:
        arguments['num_steps_per_lambda'] = num_steps
        arguments['num_equil_steps'] = num_steps

        results[str(num_steps)] = np.zeros(num_repeats)
        times[str(num_steps)] = np.zeros(num_repeats)

        for i in range(num_repeats):
            print(f'starting trial #{i + 1}...')
            t0 = time()
            dG = estimate_dG(transformation, **arguments)
            results[str(num_steps)][i] = dG
            t1 = time()
            elapsed = t1 - t0
            times[str(num_steps)][i] = elapsed
            print(f'\tdG_estimate = {dG:.3f}\n\t(wall time: {elapsed:.3f} s)')

        np.savez('estimator_variance_results.npz', **results)
        np.savez('estimator_variance_timings.npz', **times)

    # TODO: also loop over num_lambdas from 2 up to 1000 or something...

    # TODO: cache initial minimized structure and other setup, if this is all a
    #   deterministic function of lambda
