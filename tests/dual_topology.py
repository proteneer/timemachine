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


def get_heavy_atom_idxs(mol):
    idxs = []
    for a_idx, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() > 1:
            idxs.append(a_idx)
    return np.array(idxs, dtype=np.int32)


def get_romol_conf(mol):
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf = guest_conf/10 # from angstroms to nm
    return np.array(guest_conf, dtype=np.float64)


def run(args):

    lamb, intg, bound_potentials, masses, x0, box, gpu_idx, stage = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    u_impls = []
    for bp in bound_potentials:
        u_impls.append(bp.bound_impl(precision=np.float32))

    # important that we reseed here.

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
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

    frames = [x0]

    # secondary minimization needed only for stage 1
    if stage == 1:
        for l in np.linspace(0.35, lamb, 500):
            ctxt.step(l)

    # equilibration
    for step in range(10000):
        ctxt.step(lamb)

    assert np.any(np.abs(ctxt.get_x_t()) > 100) == False
    assert np.any(np.isnan(ctxt.get_x_t())) == False
    assert np.any(np.isinf(ctxt.get_x_t())) == False

    # du_dl_obs = custom_ops.AvgPartialUPartialLambda(u_impls, 10)
    du_dl_obs = custom_ops.AvgPartialUPartialLambda(u_impls, 5)
    ctxt.add_observable(du_dl_obs)

    for step in range(90000):
        # if step % 500 == 0:
            # frames.append(ctxt.get_x_t())
        ctxt.step(lamb)

    assert np.any(np.abs(ctxt.get_x_t()) > 100) == False
    assert np.any(np.isnan(ctxt.get_x_t())) == False
    assert np.any(np.isinf(ctxt.get_x_t())) == False

    print("lamb", lamb, "du_dl", du_dl_obs.avg_du_dl())

    return du_dl_obs.avg_du_dl(), frames

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
    host_atom_idxs = np.arange(len(r_host.masses))
    ligand_atom_idxs = np.arange(len(r_ligand.masses)) + len(r_host.masses)
    rbfe.set_nonbonded_lambda_idxs(r_combined, ligand_atom_idxs, 0, 1)

    return r_combined

def main(args, stage):

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    mols = [x for x in suppl]
    ligand_a = mols[0]
    ligand_b = mols[1]

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    r_ligand_a = Recipe.from_rdkit(ligand_a, ff_handlers)
    r_ligand_b = Recipe.from_rdkit(ligand_b, ff_handlers)

    r_combined = r_ligand_a.combine(r_ligand_b)
   
    a_idxs = get_heavy_atom_idxs(ligand_a)
    b_idxs = get_heavy_atom_idxs(ligand_b)
    b_idxs += ligand_a.GetNumAtoms()

    a_full_idxs = np.arange(0, ligand_a.GetNumAtoms())
    offset_idxs = np.arange(0, ligand_b.GetNumAtoms()) + ligand_a.GetNumAtoms()

    shape_k = 200.0
    centroid_k = 50.0

    if stage == 0:
        rbfe.stage_0(r_combined, a_idxs, b_idxs, offset_idxs, centroid_k, shape_k)
        lambda_schedule = np.linspace(0.0, 1.0, 20)
        # lambda_schedule = np.array([1.0, 1.0, 1.0])
    elif stage == 1:
        rbfe.stage_1(r_combined, a_idxs, b_idxs, a_full_idxs, offset_idxs, centroid_k, shape_k)
        lambda_schedule = np.linspace(0.0, 1.2, 60)
        # lambda_schedule = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    else:
        assert 0

    system, host_coords, box, topology = builders.build_water_system(4.0)

    r_host = Recipe.from_openmm(system)
    r_final = r_host.combine(r_combined)

    # minimize coordinates of host + ligand A
    ha_coords = np.concatenate([
        host_coords,
        get_romol_conf(ligand_a)
    ])

    pool = Pool(args.num_gpus)

    # we need to run this in a subprocess since the cuda runtime
    # must not be initialized in the master thread due to lack of
    # fork safety
    r_minimize = minimize_setup(r_host, r_ligand_a)
    ha_coords = pool.map(minimize, [(r_minimize.bound_potentials, r_minimize.masses, ha_coords, box)], chunksize=1)
    # this is a list of size 1
    ha_coords = ha_coords[0]
    pool.close()

    pool = Pool(args.num_gpus)

    x0 = np.concatenate([
        ha_coords,
        get_romol_conf(ligand_b)
    ])

    masses = np.concatenate([r_host.masses, r_ligand_a.masses, r_ligand_b.masses])

    seed = np.random.randint(np.iinfo(np.int32).max)

    intg = LangevinIntegrator(
        300.0,
        1.5e-3,
        1.0,
        masses,
        seed
    )

    # production run at various values of lambda
    for epoch in range(100):
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

        results = pool.map(run, run_args, chunksize=1)

        avg_du_dls = []
        writer = pdb_writer.PDBWriter([topology, ligand_a, ligand_b], "frames.pdb")
        for dl, frames in results:
            avg_du_dls.append(dl)
            for frame in frames:
                writer.write_frame(frame*10)
            writer.close()

        for lamb, dl in zip(lambda_schedule, avg_du_dls):
            print("lamb", lamb, "du_dl", dl)


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

    main(args, 1)