# relative hydration free energy with a single topology protocol.
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

from multiprocessing import Pool

from fe import relative


def wrap_method(args, fn):
    gpu_idx = args[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    return fn(*args[1:])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relative Hydration Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus"
    )

    parser.add_argument(
        "--num_vacuum_windows",
        type=int,
        help="number of vacuum lambda windows"
    )

    parser.add_argument(
        "--num_solvent_windows",
        type=int,
        help="number of solvent lambda windows"
    )

    parser.add_argument(
        "--num_equil_steps",
        type=int,
        help="number of equilibration lambda windows"
    )

    parser.add_argument(
        "--num_prod_steps",
        type=int,
        help="number of production lambda windows"
    )

    cmd_args = parser.parse_args()

    p = Pool(cmd_args.num_gpus)

    suppl = Chem.SDMolSupplier('tests/data/benzene_flourinated.sdf', removeHs=False)
    all_mols = [x for x in suppl]

    mol_a = all_mols[0]
    mol_b = all_mols[1]

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    ff = Forcefield(ff_handlers)

    core = np.stack([
        np.arange(mol_a.GetNumAtoms()),
        np.arange(mol_b.GetNumAtoms())
    ], axis=1)

    rfe = relative.RelativeFreeEnergy(mol_a, mol_b, core, ff)

    vacuum_lambda_schedule = np.linspace(0.0, 1.0, cmd_args.num_vacuum_windows)
    solvent_lambda_schedule = np.linspace(0.0, 1.0, cmd_args.num_solvent_windows)

    # vacuum leg
    vacuum_args = []
    for lambda_idx, lamb in enumerate(vacuum_lambda_schedule):
        gpu_idx = lambda_idx % cmd_args.num_gpus
        vacuum_args.append((gpu_idx, lamb, cmd_args.num_equil_steps, cmd_args.num_prod_steps))

        # functools.partial(wrap_method, fn=rfe.vacuum_edge)((gpu_idx, lamb, 10000, 20000))

    results = p.map(functools.partial(wrap_method, fn=rfe.vacuum_edge), vacuum_args)

    for lamb, (bonded_du_dl, nonbonded_du_dl) in zip(vacuum_lambda_schedule, results):
        print("final vacuum lambda", lamb, "bonded", bonded_du_dl, nonbonded_du_dl)

    dG_vacuum = np.trapz([x[0]+x[1] for x in results], vacuum_lambda_schedule)
    print("dG vacuum:", dG_vacuum)

    # solvent leg
    # build the water system first.
    solvent_system, solvent_coords, solvent_box, omm_topology = builders.build_water_system(4.0)
    solvent_box += np.eye(3)*0.1

    minimized_solvent_coords = minimizer.minimize_host_4d(mol_a, solvent_system, solvent_coords, ff, solvent_box)

    solvent_args = []
    for lambda_idx, lamb in enumerate(solvent_lambda_schedule):
        gpu_idx = lambda_idx % cmd_args.num_gpus
        solvent_args.append((gpu_idx, lamb, solvent_system, minimized_solvent_coords, solvent_box, cmd_args.num_equil_steps, cmd_args.num_prod_steps))
    
    results = p.map(functools.partial(wrap_method, fn=rfe.host_edge), solvent_args)

    for lamb, (bonded_du_dl, nonbonded_du_dl) in zip(solvent_lambda_schedule, results):
        print("final solvent lambda", lamb, "bonded", bonded_du_dl, nonbonded_du_dl)

    dG_solvent = np.trapz([x[0]+x[1] for x in results], vacuum_lambda_schedule)
    print("dG solvent:", dG_solvent)