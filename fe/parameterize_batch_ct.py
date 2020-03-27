import copy
import argparse
import time
import numpy as np
from io import StringIO
import itertools
import os
import sys

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

from fe import simulation
from fe import loss
from fe import dataset
from fe.pdb_writer import PDBWriter

from ff import forcefield
from ff import system
from ff import openmm_converter

from fe import bar

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

    all_du_dls = []
    for conf_idx, du_dl in enumerate(all_du_dls_raw):
        if du_dl is not None:
            all_du_dls.append(du_dl)
            print("conf_idx", conf_idx, "dG", math_utils.trapz(du_dl[T:], schedule[T:]))
        else:
            print("conf_idx", conf_idx, "dG None")

    all_du_dls = jnp.array(all_du_dls)
    bkwd = all_du_dls[:, T:]
    bkwd_sched = schedule[T:]
    dG_bkwd = math_utils.trapz(bkwd, bkwd_sched) # integral from 0 to inf
    dG_bkwd = -dG_bkwd
    kT = 2.479
    dG_bkwd /= kT
    pred_dG = bar.EXP(dG_bkwd)
    pred_dG *= kT
    pred_dG = -pred_dG
    pred_dG = pred_dG / 8.0
    print("pred_dG", pred_dG, "true_dG", true_dG)
    return jnp.abs(pred_dG - true_dG)

def error_fn_2(all_du_dls_raw, T, schedule, true_dG):

    all_du_dls = []
    for conf_idx, du_dl in enumerate(all_du_dls_raw):
        if du_dl is not None:
            all_du_dls.append(du_dl)

    all_du_dls = jnp.array(all_du_dls)
    bkwd = all_du_dls[:, T:]
    bkwd_sched = schedule[T:]
    dG_bkwd = math_utils.trapz(bkwd, bkwd_sched) # integral from 0 to inf
    dG_bkwd = -dG_bkwd
    kT = 2.479
    dG_bkwd /= kT
    pred_dG = bar.EXP(dG_bkwd)
    pred_dG *= kT
    pred_dG = -pred_dG
    pred_dG = pred_dG / 8.0
    return pred_dG

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--precision', type=str, required=True)    
    parser.add_argument('--complex_pdb', type=str, required=True)
    parser.add_argument('--ligand_sdf', type=str, required=True, nargs="*")
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--num_conformers', type=int, required=True)
    parser.add_argument('--forcefield', type=str, required=True)
    args = parser.parse_args()

    assert os.path.isdir(args.out_dir)

    if args.precision == 'single':
        precision = np.float32
    elif args.precision == 'double':
        precision = np.float64
    else:
        raise Exception("precision must be either single or double")

    num_gpus = args.num_gpus
    num_conformers = args.num_conformers

    all_du_dls = []

    insertion_T = 5000
    part_one_T = 500
    part_two_T = insertion_T - part_one_T
    insertion_lambda_part_one = np.linspace(1.0, 0.3, part_one_T) # insertion
    insertion_lambda_part_two = np.linspace(0.3, 0.0, part_two_T) # insertion
    insertion_lambda = np.concatenate([insertion_lambda_part_one, insertion_lambda_part_two])
    insertion_cas = np.ones(insertion_T, dtype=np.float64)*0.9
    insertion_dts = np.ones(insertion_T) * 0.001

    relaxation_T = 2000
    relaxation_lambda = np.zeros(relaxation_T) # relaxation
    relaxation_cas = np.ones(relaxation_T, dtype=np.float64)*0.9
    relaxation_dts = np.linspace(0.001, 0.01, relaxation_T).astype(np.float64)

    deletion_T = 5000
    deletion_lambda = np.linspace(0.0, 5.0, deletion_T).astype(np.float64)
    deletion_cas = np.ones(deletion_T, dtype=np.float64)*0.9
    deletion_dts = np.ones(deletion_T)*0.0015

    deletion_offset = insertion_T + relaxation_T # when we start doing deletion

    complete_T = insertion_T + relaxation_T + deletion_T
    complete_lambda = np.concatenate([insertion_lambda, relaxation_lambda, deletion_lambda])
    complete_cas = np.concatenate([insertion_cas, relaxation_cas, deletion_cas])
    complete_dts = np.concatenate([insertion_dts, relaxation_dts, deletion_dts])

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

    cutoff = 1.25

    host_system = openmm_converter.deserialize_system(host_system, cutoff=cutoff)
    num_host_atoms = len(host_system.masses)

    print("num_host_atoms", num_host_atoms)

    for ligand_sdf_file in args.ligand_sdf:
       for guest_mol in Chem.SDMolSupplier(ligand_sdf_file, removeHs=False):
           break

    open_ff = forcefield.Forcefield(args.forcefield)
    nrg_fns = open_ff.parameterize(guest_mol, cutoff=cutoff)
    guest_masses = get_masses(guest_mol)
    guest_system = system.System(nrg_fns, open_ff.params, open_ff.param_groups, guest_masses)

    off_groups_path = os.path.join(args.out_dir, "original_off_param_groups.txt")
    np.savetxt(off_groups_path, open_ff.param_groups)

    combined_system = host_system.merge(guest_system)
    cbs = -0.0001/np.array(combined_system.masses)
    lambda_idxs = np.zeros(len(combined_system.masses), dtype=np.int32)
    lambda_idxs[num_host_atoms:] = 1

    sim = simulation.Simulation(
        combined_system,
        complete_dts,
        complete_cas,
        cbs,
        complete_lambda,
        lambda_idxs,
        precision
    )

    initial_params = sim.system.params

    lr = 5e-4
    opt_init, opt_update, get_params = optimizers.adam(lr)

    opt_state = opt_init(initial_params)

    num_epochs = 100
    #vary_allowed_groups = [(7, 0.01)]*10 + [(14, 0.01)]*20 + [(12, 1e-2)]*20 + [(13, 1e-2)]*20 + [(7, 0.01)]*30
    ligand_list = []     

    for ligand_sdf_file in args.ligand_sdf:
        for rd_mol in Chem.SDMolSupplier(ligand_sdf_file, removeHs=False):
            init_mol = Chem.Mol(rd_mol)
            dG_value = rd_mol.GetProp("_dG")
            
            ### generate 512 conformers and prioritize them based on work values
            pool_conformers = 88
            np.random.seed(2020)
            init_conf = rd_mol.GetConformer(0)
            init_conf = np.array(init_conf.GetPositions(), dtype=np.float64)
            init_conf = init_conf/10 # convert to md_units
            conf_com = com(init_conf)
            rd_mol.RemoveAllConformers()
            AllChem.EmbedMultipleConfs(rd_mol, pool_conformers, randomSeed=2020)

            for conf_idx in range(pool_conformers):
                conformer = rd_mol.GetConformer(conf_idx)
                guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
                guest_conf = guest_conf/10 # convert to md_units
                rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
                guest_conf = np.matmul(guest_conf, rot_matrix)*10

                for atom_idx, pos in enumerate(guest_conf):
                    conformer.SetAtomPosition(atom_idx, (float(pos[0]), float(pos[1]), float(pos[2])))

            open_ff = forcefield.Forcefield(args.forcefield)
            nrg_fns = open_ff.parameterize(rd_mol, cutoff=cutoff)
            guest_masses = get_masses(rd_mol)
            guest_system = system.System(nrg_fns, open_ff.params, open_ff.param_groups, guest_masses)

            combined_system = host_system.merge(guest_system)
            cbs = -0.0001/np.array(combined_system.masses)
            lambda_idxs = np.zeros(len(combined_system.masses), dtype=np.int32)
            lambda_idxs[num_host_atoms:] = 1

            sim = simulation.Simulation(
                combined_system,
                complete_dts,
                complete_cas,
                cbs,
                complete_lambda,
                lambda_idxs,
                precision
            )

            all_args = []

            child_conns = []
            parent_conns = []
            for conf_idx in range(pool_conformers):

                conformer = rd_mol.GetConformer(conf_idx)
                guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
                guest_conf = guest_conf/10 # convert to md_units
                guest_conf = recenter(guest_conf, conf_com)
                x0 = np.concatenate([host_conf, guest_conf])       # combined geometry
                combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), init_mol)
                combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                #out_file = os.path.join(args.out_dir, "fwd_conf_idx"+str(conf_idx)+"_insertion_ligand_dG_"+str(dG_value)+".pdb")
                #writer = PDBWriter(combined_pdb_str, out_file)
                v0 = np.zeros_like(x0)

                parent_conn, child_conn = Pipe()
                parent_conns.append(parent_conn)
                # writer can be None if we don't care about vis
                all_args.append([x0, v0, conf_idx % num_gpus, None, child_conn])

            processes = []

            for arg in all_args:
                p = Process(target=sim.run_forward_and_backward, args=arg)
                p.daemon = True
                processes.append(p)
                p.start()

            all_du_dls = []
            for pc in parent_conns:
                du_dls = pc.recv()
                all_du_dls.append(du_dls)
            
            all_du_dls = np.array(all_du_dls)
            dG_bkwd = []
            for du_dls in all_du_dls:
                if du_dls is None:
                    dG_bkwd.append(-9999.9)
                else:
                    bkwd = du_dls[deletion_offset:]
                    dG_bkwd.append(math_utils.trapz(bkwd, deletion_lambda))
            dG_bkwd = np.asarray(dG_bkwd)
            np.savetxt(os.path.join(args.out_dir, "bkwd_ligand_%s_.txt"%dG_value), dG_bkwd)
            pool_conformer_idxs = np.argsort(dG_bkwd)[::-1][:num_conformers]
            ligand_list.append((rd_mol, float(dG_value), pool_conformer_idxs, dG_bkwd))

            for pc in parent_conns:
                pc.send(None)
                pc.close()

            for p in processes:
                p.join()

    print("ligand_list", ligand_list)
    ligand_dataset = dataset.Dataset(ligand_list)
    train, test = ligand_dataset.split(0.7)

    loss_file_path = os.path.join(args.out_dir, "loss.txt")
    if os.path.exists(loss_file_path):
        os.remove(loss_file_path)

    for epoch in range(num_epochs):        
        for idx, dataset in enumerate([train, test]):
            if idx == 1:
                inference = True
                job_name = 'test'
            else:
                inference = False
                job_name = 'train'
                dataset.shuffle()

            for id, data in enumerate(dataset.iterbatches(1)):
                print("id, data ", id, data)
                # sample from the rdkit DG distribution (this can be changed later to another distribution later on)
                guest_mol = copy.deepcopy(data[0][0])
                init_mol = Chem.Mol(guest_mol)
                true_dG = data[0][1] # in kJ/mol and positive sign 
                pool_conformer_idxs = data[0][2]
                pool_bkwd_dG = data[0][3]

                open_ff = forcefield.Forcefield(args.forcefield)
                nrg_fns = open_ff.parameterize(guest_mol, cutoff=cutoff)
                guest_masses = get_masses(guest_mol)
                guest_system = system.System(nrg_fns, open_ff.params, open_ff.param_groups, guest_masses)

                combined_system = host_system.merge(guest_system)
                cbs = -1*np.ones_like(np.array(combined_system.masses))*0.0001
                lambda_idxs = np.zeros(len(combined_system.masses), dtype=np.int32)
                lambda_idxs[num_host_atoms:] = -1

                sim = simulation.Simulation(
                    combined_system,
                    complete_dts,
                    complete_cas,
                    cbs,
                    complete_lambda,
                    lambda_idxs,
                    precision
                )

                epoch_train_params = np.asarray(get_params(opt_state))
                
                if not inference:
                    epoch_train_ff_params = copy.deepcopy(open_ff)
                    epoch_train_ff_params.params = epoch_train_params[len(host_system.params):]
                    fname = "epoch_"+str(epoch)+"_ligand_"+str(id)+"_params"
                    fpath = os.path.join(args.out_dir, fname)
                    epoch_train_ff_params.save(fpath)

                sim.system.params = epoch_train_params

                all_args = []

                child_conns = []
                parent_conns = []

                for conf_idx in pool_conformer_idxs:

                    conformer = guest_mol.GetConformer(int(conf_idx))
                    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
                    guest_conf = guest_conf/10 # convert to md_units
                    guest_conf = recenter(guest_conf, conf_com)
                    select_path = os.path.join(args.out_dir, "preselected_conf_"+str(conf_idx)+"_lig_"+str(true_dG))
                    np.save(select_path, guest_conf)
                    x0 = np.concatenate([host_conf, guest_conf])       # combined geometry
                    combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), init_mol)
                    combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))

                    out_file = os.path.join(args.out_dir, "epoch_"+str(epoch)+"_inde_lig_"+
                        str(true_dG)+"_conf_"+str(conf_idx)+"_bkwd_"+str(pool_bkwd_dG[conf_idx])+".pdb")

                    writer = PDBWriter(combined_pdb_str, out_file)

                    v0 = np.zeros_like(x0)

                    parent_conn, child_conn = Pipe()
                    parent_conns.append(parent_conn)
                    # writer can be None if we don't care about vis
                    all_args.append([x0, v0, conf_idx % num_gpus, writer, child_conn])

                processes = []

                for arg in all_args:
                    p = Process(target=sim.run_forward_and_backward, args=arg)
                    p.daemon = True
                    processes.append(p)
                    p.start()

                all_du_dls = []
                for pc in parent_conns:
                    du_dls = pc.recv()
                    #if du_dls is not None:
                    all_du_dls.append(du_dls)
                    #else:
                    #    all_du_dls.append(np.ones_like(complete_lambda)*1e-10)

                #all_du_dls = np.array(all_du_dls)
                
                if inference:
                    # test mode
                    error = error_fn(all_du_dls, deletion_offset, complete_lambda, true_dG)
                    with open(loss_file_path, "a+") as f:
                        f.write("---EPOCH %d---Test---LOSS--- %f\n " % (epoch, error))
                        f.write("---True---dG--- %f\n" % true_dG)
                        f.write("---Predicted---dG--- %f\n" % error_fn_2(all_du_dls, deletion_offset, complete_lambda, true_dG))

                    for pc in parent_conns:
                        pc.send(None)
                        pc.close()

                else:
                    # training mode, so we need to compute the derivatives
                    print("all_du_dls ",all_du_dls)
                    loss_grad_fn = jax.grad(error_fn, argnums=(0,))
                    error = error_fn(all_du_dls, deletion_offset, complete_lambda, true_dG)
                    work_grads = exp_grad_filter(all_du_dls, deletion_offset, complete_lambda, true_dG)

                    with open(loss_file_path, "a+") as f:
                        f.write("---EPOCH %d---Train---LOSS--- %f\n " % (epoch, error))
                        f.write("---True---dG--- %f\n" % true_dG)
                        f.write("---Predicted---dG--- %f\n" % error_fn_2(all_du_dls, deletion_offset, complete_lambda, true_dG))

                    error_grad = loss_grad_fn(all_du_dls, deletion_offset, complete_lambda, true_dG)
                    all_du_dl_adjoints = error_grad[0]

                    work_grad_cutoff = 1e-2

                    assert len(parent_conns) == len(all_du_dl_adjoints)
                    assert len(all_du_dl_adjoints) == len(work_grads)
                    assert len(work_grads) ==  len(all_du_dls)
                    
                    picked_conformers = []
                    unstable_conformers = []

                    for conf_idx, (pc, du_dl_adjoints, wg, du_dls) in enumerate(zip(parent_conns, all_du_dl_adjoints, work_grads, all_du_dls)):
                        if du_dls is None:
                            unstable_conformers.append(conf_idx)
                        if wg < work_grad_cutoff or du_dls is None:
                            pc.send(None)
                        else:
                            picked_conformers.append(conf_idx)
                            pc.send(du_dl_adjoints)

                    print("picked conformers:", picked_conformers)
                    print("unstable conformers:", unstable_conformers)

                    # receive everything at once
                    all_dl_dps = []
                    for conf_idx, (pc, wg, du_dls) in enumerate(zip(parent_conns, work_grads, all_du_dls)):
                        if wg > work_grad_cutoff and du_dls is not None:
                            dl_dp = pc.recv()
                            all_dl_dps.append(dl_dp)

                    # terminate all the processes
                    for p_idx, p in enumerate(processes):
                        p.join()

                    all_dl_dps = np.array(all_dl_dps)
                    all_dl_dps = np.sum(all_dl_dps, axis=0)

                    allowed_groups = {
                       14: 0.01}
                       #12: 0.01,
                       #13: 0.01}
                    #allowed_groups = {vary_allowed_groups[epoch][0]: vary_allowed_groups[epoch][1]}

                    filtered_grad = []
                    for g_idx, (g, gp) in enumerate(zip(all_dl_dps, sim.system.param_groups)):
                        if gp in allowed_groups:
                            pf = allowed_groups[gp]
                            filtered_grad.append(g*pf)
                            if g != 0:
                                print("derivs", g_idx, '\t group', gp, '\t', g, '\t adjusted to', g*pf, '\t old val', sim.system.params[g_idx])
                        else:
                            filtered_grad.append(0)

                    filtered_grad = np.array(filtered_grad)
                    opt_state = opt_update(epoch, filtered_grad, opt_state)

                    for pc in parent_conns:
                        pc.close()

                # terminate all the processes
                for p in processes:
                    p.join()
