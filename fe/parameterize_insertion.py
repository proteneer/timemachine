import copy
import argparse
import time
import numpy as np
from io import StringIO
import itertools
import os
import sys
import pickle

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)

from scipy.stats import special_ortho_group
import jax
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from simtk.openmm import app
from simtk.openmm.app import PDBFile

from timemachine.lib import custom_ops, ops
from fe.utils import to_md_units, write
from fe import math_utils

from multiprocessing import Process, Pipe
from matplotlib import pyplot as plt

from jax.experimental import optimizers

from rdkit.Chem import rdMolAlign

from fe import simulation
from fe import loss, bar
from fe.pdb_writer import PDBWriter

from ff import forcefield
from ff import system
from ff import openmm_converter

def com(conf):
    return np.sum(conf, axis=0)/conf.shape[0]

def recenter(conf, true_com, scale_factor=1):
    mol_com = np.sum(conf, axis=0)/conf.shape[0]
    centered = conf - mol_com  # centered to origin
    return true_com + centered/scale_factor 

from hilbertcurve.hilbertcurve import HilbertCurve

def hilbert_sort(conf):
    hc = HilbertCurve(16, 3)
    int_confs = (conf*1000).astype(np.int64)+10000
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists)
    return perm

def get_masses(m):
    masses = []
    for a in m.GetAtoms():
        masses.append(a.GetMass())
    return masses

import jax.numpy as jnp

exp_grad = jax.grad(bar.EXP)

def exp_grad_filter(all_du_dls_raw, T, schedule, true_dG):
    """
    Compute the derivative of exp weighted free energy with respect
    to the input work values
    """
    all_du_dls = []
    for du_dl in all_du_dls_raw:
        if du_dl is not None:
            all_du_dls.append(du_dl)
        else:
            all_du_dls.append(np.zeros_like(schedule))
    all_du_dls = jnp.array(all_du_dls)

    bkwd = all_du_dls[:, T:]
    bkwd_sched = schedule[T:]

    dG_bkwd = math_utils.trapz(bkwd, bkwd_sched) # integral from 0 to inf
    dG_bkwd = -dG_bkwd

    kT = 2.479
    dG_bkwd /= kT

    grads = exp_grad(dG_bkwd)

    return grads

def error_fn(all_du_dls_raw, T, schedule, true_dG):

    print("schedule", schedule)
    print("T", T)

    all_du_dls = []
    for conf_idx, du_dl in enumerate(all_du_dls_raw):
        if du_dl is not None:
            all_du_dls.append(du_dl)
            print("conf_idx", conf_idx, "dG", math_utils.trapz(du_dl[T:], schedule[T:]))
        else:
            print("conf_idx", conf_idx, "dG None")

    all_du_dls = jnp.array(all_du_dls)

    fwd = all_du_dls
    fwd_sched = schedule

    dG_fwd = math_utils.trapz(fwd, fwd_sched) # integral from inf to 0
    # this is in kJ/mol, inputs to BAR needs to be in 1/kT.
    kT = 2.479
    dG_fwd /= kT

    pred_dG = bar.EXP(dG_fwd)
    pred_dG *= kT

    print("dG_fwd", pred_dG, "true_dG", true_dG)
    return jnp.abs(pred_dG - true_dG)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--precision', type=str, required=True)    
    parser.add_argument('--complex_pdb', type=str, required=True)
    parser.add_argument('--ligand_sdf', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--num_conformers', type=int, required=True)
    parser.add_argument('--forcefield', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    assert os.path.isdir(args.out_dir)


    if args.precision == 'single':
        precision = np.float32
    elif args.precision == 'double':
        precision = np.float64
    else:
        raise Exception("precision must be either single or double")

    suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)

    all_guest_mols = []
    for guest_mol in suppl:
        all_guest_mols.append(guest_mol)
    
    np.random.seed(args.seed)

    perm = np.arange(len(all_guest_mols))
    np.random.shuffle(perm)


    print(perm[0])
    guest_mol = all_guest_mols[perm[0]]

    num_gpus = args.num_gpus
    all_du_dls = []

    cutoff = 1.25

    # insertion only
    # Ts = [
    #     1000, # apo minimization
    #     7000, # insertion
    # ]

    # lambda_offset = 0

    # complete_T = np.sum(Ts)

    # complete_lambda = np.concatenate([
    #     np.linspace(cutoff, 0.35, Ts[0]),
    #     np.linspace(0.35,   0.0,  Ts[1]),
    # ])

    # complete_cas = np.concatenate([
    #     np.linspace(0.00, 0.20, Ts[0]),
    #     np.linspace(0.20, 0.95, Ts[1]),
    # ])

    # complete_dts = np.concatenate([
    #     np.linspace(0.001, 0.001, Ts[0]),
    #     np.linspace(0.001, 0.001, Ts[1]),
    # ])


    # insertion deletion

    # Ts = [
    #     2000, # insertion
    #     2000, # holo-minimization
    #     4000, # deletion
    # ]

    # lambda_offset = Ts[0]

    # complete_T = np.sum(Ts)

    # cutoff = 1.25

    # complete_lambda = np.concatenate([
    #     np.linspace(cutoff, 0.0, Ts[0]),
    #     np.linspace( 0.0, 0.0, Ts[1]),
    #     np.linspace( 0.0, cutoff, Ts[2])
    # ])

    # complete_cas = np.concatenate([
    #     np.linspace(0.00, 0.20, Ts[0]),
    #     np.linspace(0.20, 0.80, Ts[1]),
    #     np.linspace(0.80, 0.95, Ts[2])
    # ])

    # complete_dts = np.concatenate([
    #     np.linspace(0.001, 0.001, Ts[0]),
    #     np.linspace(0.001, 0.001, Ts[1]),
    #     np.linspace(0.001, 0.001, Ts[2])
    # ])

    # equilibration

    # Ts = [
    #     5000, # minimization (with thermal noise lol)
    #     20000, # equilibration
    #     10000, # decoupling
    # ]

    # lambda_offset = Ts[0]+Ts[1]

    # complete_lambda = np.concatenate([
    #     np.linspace(cutoff, 0.0, Ts[0]),
    #     np.linspace( 0.0, 0.0, Ts[1]),
    #     np.linspace( 0.0, cutoff, Ts[2]),
    # ])

    # complete_cas = np.concatenate([
    #     np.linspace(0.00,  0.913, Ts[0]),
    #     np.linspace(0.913,  0.913, Ts[1]),
    #     np.linspace(0.913,  0.913, Ts[2]),
    # ])

    # complete_dts = np.concatenate([
    #     np.linspace(1e-9, 1e-3, Ts[0]),
    #     np.linspace(1e-3, 1e-3, Ts[1]),
    #     np.linspace(1e-3, 1e-3, Ts[2]),
    # ])

    # fully decouple equilibration

    Ts = [
        5000, # minimization (with thermal noise lol)
        50000, # equilibration
    ]

    lambda_offset = Ts[0]+Ts[1]

    complete_lambda = np.concatenate([
        np.linspace(cutoff, cutoff, Ts[0]),
        np.linspace(cutoff, cutoff, Ts[1])
    ])

    complete_cas = np.concatenate([
        np.linspace(0.00,  0.913, Ts[0]),
        np.linspace(0.913,  0.913, Ts[1])
    ])

    complete_dts = np.concatenate([
        np.linspace(1e-9, 1e-3, Ts[0]),
        np.linspace(1e-3, 1e-3, Ts[1])
    ])



    print("dt", complete_dts[3400])

    init_conf = guest_mol.GetConformer(0)
    init_conf = np.array(init_conf.GetPositions(), dtype=np.float64)
    init_conf = init_conf/10 # convert to md_units
    conf_com = com(init_conf)
    init_mol = Chem.Mol(guest_mol)
    num_conformers = args.num_conformers

    np.random.seed(args.seed)

    num_epochs = 1000

    open_ff = forcefield.Forcefield(args.forcefield)
    nrg_fns = open_ff.parameterize(guest_mol, cutoff=cutoff)

    all_du_dls = []

    # generate a set of gas phase conformers using the RDKit
    orig_pos = np.array(guest_mol.GetConformer(0).GetPositions())
    # print("NC", guest_mol.GetNumConformers())
    # guest_mol.RemoveAllConformers()
    AllChem.EmbedMultipleConfs(guest_mol, num_conformers, clearConfs=False, randomSeed=args.seed)
    for conf_idx in range(num_conformers):
        conformer = guest_mol.GetConformer(conf_idx)
        if conf_idx == 0:
            for atom_idx, pos in enumerate(orig_pos):
                conformer.SetAtomPosition(atom_idx, (float(pos[0]), float(pos[1]), float(pos[2])))

        else:
            guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
            guest_conf = guest_conf/10 # convert to md_units
            rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
            guest_conf = np.matmul(guest_conf, rot_matrix)*10

            for atom_idx, pos in enumerate(guest_conf):
                conformer.SetAtomPosition(atom_idx, (float(pos[0]), float(pos[1]), float(pos[2])))


    # coreIds = (np.array([2,3,4,5,6,7,8,9,10]) - 1).tolist()
    # rdMolAlign.AlignMolConformers(guest_mol, atomIds=coreIds)
    rdMolAlign.AlignMolConformers(guest_mol)

    pickle.dump(guest_mol, open(os.path.join(args.out_dir, "mol.pkl"), "wb"))

    lr = 5e-4
    # opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_init, opt_update, get_params = optimizers.sgd(lr)

    host_pdb_file = args.complex_pdb
    host_pdb = app.PDBFile(host_pdb_file)
    host_conf = []
    for x,y,z in host_pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)
    host_name = "complex"

    # set up the system
    amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    host_system = amber_ff.createSystem(host_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)

    host_system = openmm_converter.deserialize_system(host_system, cutoff=cutoff)
    num_host_atoms = len(host_system.masses)

    print("num_host_atoms", num_host_atoms)


    guest_masses = get_masses(guest_mol)
    guest_system = system.System(nrg_fns, open_ff.params, open_ff.param_groups, guest_masses)

    combined_system = host_system.merge(guest_system)

    cbs = -0.001/np.array(combined_system.masses)
    ccs = 0.07035977/np.sqrt(np.array(combined_system.masses))
    # ccs = 0/np.sqrt(np.array(combined_system.masses))
    # ccs = 0.0/np.array(combined_system.masses)
    # cbs[num_host_atoms:] = 0 # rigidify the ligand
    # cbs = -0.001/np.ones_like(combined_system.masses)

    # cbs = -.5*np.ones_like(np.array(combined_system.masses))*0.0001

    lambda_idxs = np.zeros(len(combined_system.masses), dtype=np.int32)
    lambda_idxs[num_host_atoms:] = 1

    sim = simulation.Simulation(
        combined_system,
        complete_dts,
        complete_cas,
        cbs,
        ccs,
        complete_lambda,
        lambda_idxs,
        precision,
        args.seed
    )

    initial_params = sim.system.params

    all_args = []

    for conf_idx in range(num_conformers):

        conformer = guest_mol.GetConformer(conf_idx)
        guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
        guest_conf = guest_conf/10 # convert to md_units
        guest_conf = recenter(guest_conf, conf_com)
        x0 = np.concatenate([host_conf, guest_conf])       # combined geometry

        combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), init_mol)
        combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
        out_file = os.path.join(args.out_dir, "iconf_"+str(conf_idx)+".pdb")
        writer = PDBWriter(combined_pdb_str, out_file)
        # writer = None
        v0 = np.zeros_like(x0)
        # writer can be None if we don't care about vis
        all_args.append((x0, v0, conf_idx % num_gpus, writer))


    batch_size = args.num_gpus

    plt.xlabel("lambda")
    plt.ylabel("du_dl")

    axes = plt.gca()
    axes.set_xlim([0, cutoff])
    axes.set_ylim([-3000, 2000])



    for idx in range(0, len(all_args), batch_size):
        batch_args = all_args[idx:idx+batch_size]

        processes = []
        child_conns = []
        parent_conns = []


        for b_args in batch_args:

            parent_conn, child_conn = Pipe()
            parent_conns.append(parent_conn)   

            b_args = b_args + (child_conn,)
            p = Process(target=sim.run_forward_and_backward, args=b_args)
            p.daemon = True
            processes.append(p)
            p.start()

        for b_idx, pc in enumerate(parent_conns):
            du_dls = pc.recv()
            all_du_dls.append(du_dls)
            pc.send(None)
      
            if du_dls is not None:
                plt.plot(complete_lambda[lambda_offset:], du_dls[lambda_offset:])
                plt.savefig(os.path.join(args.out_dir, "du_dls"))
                work = np.trapz(du_dls[lambda_offset:], complete_lambda[lambda_offset:])
                # if work < 0:
                print("conf_idx", idx+b_idx, "work", work)

   

        # true_dG = -26.61024 # -6.36 * 4.184 (for insertion)

        # error = error_fn(all_du_dls, deletion_offset, complete_lambda, true_dG)

        for p in processes:
            p.join()
