# test protocols for setting up relative binding free energy calculations.


import argparse
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.lib import potentials, custom_ops
from timemachine.lib import LangevinIntegrator

from ff.handlers import openmm_deserializer
from ff.handlers.deserialize import deserialize_handlers

from fe import pdb_writer
from fe import rbfe
from md import Recipe
from md import builders

from multiprocessing import Pool

def get_romol_conf(mol):
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf = guest_conf/10 # from angstroms to nm
    return np.array(guest_conf, dtype=np.float64)

def run(args):

    lamb, intg, bound_potentials, masses, x0, box, gpu_idx, stage = args
    # print("running on", gpu_idx)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    u_impls = []
    for bp in bound_potentials:
        u_impls.append(bp.bound_impl(precision=np.float32))

    # important that we reseed here.
    intg.seed = np.random.randint(np.iinfo(np.int32).max)
    intg_impl = intg.impl()

    v0 = np.zeros_like(x0)

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg_impl,
        u_impls
    )

    # secondary minimization needed only for stage 1
    if stage == 1:
        for l in np.linspace(0.35, lamb, 500):
            ctxt.step(l)

    # equilibration
    for step in range(20000):
    # for step in range(1000):
        ctxt.step(lamb)


    # print(ctxt.get_x_t())

    du_dl_obs = custom_ops.AvgPartialUPartialLambda(u_impls, 10)
    ctxt.add_observable(du_dl_obs)

    # add observable for <du/dl>
    for step in range(50000):
    # for step in range(5000):
        ctxt.step(lamb)

    print(lamb, du_dl_obs.avg_du_dl())

    assert np.any(np.abs(ctxt.get_x_t()) > 100) == False
    assert np.any(np.isnan(ctxt.get_x_t())) == False
    assert np.any(np.isinf(ctxt.get_x_t())) == False

    return du_dl_obs.avg_du_dl()

def minimize(args):

    bound_potentials, masses, x0, box = args

    u_impls = []
    for bp in bound_potentials:
        u_impls.append(bp.bound_impl(precision=np.float32))

    seed = np.random.randint(np.iinfo(np.int32).max)

    intg = LangevinIntegrator(
        300.0,
        1.5e-3,
        1.0,
        masses,
        seed
    ).impl()

    v0 = np.zeros_like(x0)

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        u_impls
    )

    steps = 500

    lambda_schedule = np.linspace(0.35, 0.0, 500)
    for lamb in lambda_schedule:
        ctxt.step(lamb)

    return ctxt.get_x_t()

def minimize_setup(r_host, r_ligand):
    r_combined = r_host.combine(r_ligand)

    print(len(r_combined.masses))

    # assert 0
    host_atom_idxs = np.arange(len(r_host.masses))
    ligand_atom_idxs = np.arange(len(r_ligand.masses)) + len(r_host.masses)
    rbfe.set_nonbonded_lambda_idxs(r_combined, ligand_atom_idxs, 0, 1)

    # for bp in r_combined.bound_potentials:
        # if isinstance(bp, potentials.Nonbonded):
            # print(len(bp.get_lambda_offset_idxs()))
            # print(len(bp.get_lambda_plane_idxs()))
            # assert 0

    return r_combined

def main(args, stage):

    # benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")) # a
    # phenol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1")) # b
                                            #01234567890
    benzene = Chem.AddHs(Chem.MolFromSmiles("C1=CC=C2C=CC=CC2=C1")) # a
    phenol =  Chem.AddHs(Chem.MolFromSmiles("C1=CC=C2C=CC=CC2=C1")) # b

    AllChem.EmbedMolecule(benzene)
    AllChem.EmbedMolecule(phenol)

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    r_benzene = Recipe.from_rdkit(benzene, ff_handlers)
    r_phenol = Recipe.from_rdkit(phenol, ff_handlers)

    r_combined = r_benzene.combine(r_phenol)
    core_pairs = np.array([
        [0,0],
        [1,1],
        [2,2],
        [3,3],
        [4,4],
        [5,5],
        [6,6],
        [7,7],
        [8,8],
        [9,9],
        # [10,10]
    ], dtype=np.int32)
    core_pairs[:, 1] += benzene.GetNumAtoms()

    a_idxs = np.arange(benzene.GetNumAtoms())
    b_idxs = np.arange(phenol.GetNumAtoms()) + benzene.GetNumAtoms()


    core_k = 20.0

    if stage == 0:
        centroid_k = 200.0
        rbfe.stage_0(r_combined, b_idxs, core_pairs, centroid_k, core_k)
        # lambda_schedule = np.linspace(0.0, 1.0, 2)
        # lambda_schedule = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        lambda_schedule = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    elif stage == 1:
        rbfe.stage_1(r_combined, a_idxs, b_idxs, core_pairs, core_k)
        lambda_schedule = np.linspace(0.0, 1.2, 60)
    else:
        assert 0



    system, host_coords, box, topology = builders.build_water_system(4.0)

    r_host = Recipe.from_openmm(system)
    r_final = r_host.combine(r_combined)

    # minimize coordinates of host + ligand A
    ha_coords = np.concatenate([
        host_coords,
        get_romol_conf(benzene)
    ])


    pool = Pool(args.num_gpus)

    # we need to run this in a subprocess since the cuda runtime
    # must not be initialized in the master thread due to lack of
    # fork safety
    r_minimize = minimize_setup(r_host, r_benzene)
    ha_coords = pool.map(minimize, [(r_minimize.bound_potentials, r_minimize.masses, ha_coords, box)], chunksize=1)
    # this is a list
    ha_coords = ha_coords[0]
    pool.close()

    pool = Pool(args.num_gpus)

    x0 = np.concatenate([
        ha_coords,
        get_romol_conf(phenol)
    ])

    masses = np.concatenate([r_host.masses, r_benzene.masses, r_phenol.masses])

    seed = np.random.randint(np.iinfo(np.int32).max)

    intg = LangevinIntegrator(
        300.0,
        1.5e-3,
        1.0,
        masses,
        seed
    )

    # production run at various values of lambda
    for epoch in range(10):
        avg_du_dls = []

        run_args = []
        for lamb_idx, lamb in enumerate(lambda_schedule):
            run_args.append((
                lamb,
                intg,
                r_final.bound_potentials,
                r_final.masses,
                x0,
                box,
                lamb_idx % args.num_gpus,
                stage))

        avg_du_dls = pool.map(run, run_args, chunksize=1)

        print("stage", stage, "epoch", epoch, "dG", np.trapz(avg_du_dls, lambda_schedule))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RBFE testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus"
    )

    args = parser.parse_args()

    main(args, 0)