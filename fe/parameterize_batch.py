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

class PDBWriter():

    def __init__(self, pdb_str, out_filepath):
        self.pdb_str = pdb_str
        self.out_filepath = out_filepath
        self.outfile = None
        self.n_frames = 25

    def write_header(self):
        """
        Confusingly this initializes writer as well because 
        """
        outfile = open(self.out_filepath, 'w')
        self.outfile = outfile
        cpdb = app.PDBFile(self.pdb_str)
        PDBFile.writeHeader(cpdb.topology, self.outfile)
        self.topology = cpdb.topology
        self.frame_idx = 0

    def write(self, x):
        if self.outfile is None:
            raise ValueError("remember to call write_header first")
        self.frame_idx += 1
        PDBFile.writeModel(self.topology, x, self.outfile, self.frame_idx)

    def close(self):
        PDBFile.writeFooter(self.topology, self.outfile)
        self.outfile.flush()

def get_masses(m):
    masses = []
    for a in m.GetAtoms():
        masses.append(a.GetMass())
    return masses

import jax.numpy as jnp

def error_fn(all_du_dls, T, schedule, true_dG):
    fwd = all_du_dls[:, :T//2]
    fwd_sched = schedule[:T//2]
    bkwd = all_du_dls[:, T//2:]
    bkwd_sched = schedule[T//2:]
    dG_fwd = math_utils.trapz(fwd, fwd_sched) # integral from inf to 0
    dG_bkwd = math_utils.trapz(bkwd, bkwd_sched) # integral from 0 to inf
    # dG_fwd and dG_bkwd have the same sign, so we need to flip dG_bkwd so the
    # direction of integral is the same (requirement for pymbar.BAR)
    dG_bkwd = -dG_bkwd # this is needed for BAR to be correct

    # this is in kJ/mol, inputs to BAR needs to be in 1/kT.
    kT = 2.479
    # kT = 1
    dG_fwd /= kT
    dG_bkwd /= kT

    pred_dG = loss.mybar(jnp.stack([dG_fwd, dG_bkwd]))
    pred_dG *= kT

    # print(dG_fwd, dG_bkwd)
    print("pred_dG", pred_dG)
    return jnp.abs(pred_dG - true_dG)

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
    all_du_dls = []

    start = 1e3
    end = 1.0
    NT = 500
    base = np.exp(np.log(end/start)/NT)
    exps = np.arange(NT)
    part_one = np.power(base, exps)*start
    part_two = np.linspace(1.0, 0.3, 1000)
    part_three = np.linspace(0.3, 0.0, 4000)

    forward_schedule = np.concatenate([part_one, part_two, part_three])
    backward_schedule = forward_schedule[::-1]
    lambda_schedule = np.concatenate([forward_schedule, backward_schedule])

    T = lambda_schedule.shape[0]
    assert T % 2 == 0
    dt = 0.0015
    step_sizes = np.ones(T)*dt

    assert T % 2 == 0
    cas = np.ones(T)*0.93

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

    host_system = openmm_converter.deserialize_system(host_system)
    num_host_atoms = len(host_system.masses)

    print("num_host_atoms", num_host_atoms)

    for ligand_sdf_file in args.ligand_sdf:
        for guest_mol in Chem.SDMolSupplier(ligand_sdf_file, removeHs=False):
            break

    open_ff = forcefield.Forcefield(args.forcefield)
    nrg_fns = open_ff.parameterize(guest_mol)
    guest_masses = get_masses(guest_mol)
    guest_system = system.System(nrg_fns, open_ff.params, open_ff.param_groups, guest_masses)

    combined_system = host_system.merge(guest_system)
    cbs = -1*np.ones_like(np.array(combined_system.masses))*0.0001
    lambda_idxs = np.zeros(len(combined_system.masses), dtype=np.int32)
    lambda_idxs[num_host_atoms:] = -1

    sim = simulation.Simulation(
        combined_system,
        step_sizes,
        cas,
        cbs,
        lambda_schedule,
        lambda_idxs,
        precision
    )

    initial_params = sim.system.params

    lr = 5e-4
    # opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_init, opt_update, get_params = optimizers.sgd(lr)

    opt_state = opt_init(initial_params)

    num_epochs = 100

    # torsion, charge, gb radii, gb scaling factor, torsion
    vary_allowed_groups = [(7, 0.5)]*10 + [(14, 0.5)]*20 + [(12, 1e-2)]*20 + [(13, 1e-2)]*20 + [(7, 0.5)]*30

    assert len(vary_allowed_groups) == num_epochs

    ligand_list = []     

    for ligand_sdf_file in args.ligand_sdf:
        for rd_mol in Chem.SDMolSupplier(ligand_sdf_file):
            dG_value = rd_mol.GetProp("_dG")
            ligand_list.append((rd_mol, float(dG_value)))

    ligand_dataset = dataset.Dataset(ligand_list)
    train, test = ligand_dataset.split(0.7)

    if os.path.exists(args.out_dir+"/loss.txt"):
        os.remove(args.out_dir+"/loss.txt")

    for epoch in range(num_epochs):
        # sample from the rdkit DG distribution (this can be changed later to another distribution later on)
        train.shuffle()
        
        for data in train.iterbatches(1):

            guest_mol = copy.deepcopy(data[0][0])
            true_dG = data[0][1] # in kJ/mol and positive sign 
            
            print(guest_mol, true_dG)

            init_conf = guest_mol.GetConformer(0)
            init_conf = np.array(init_conf.GetPositions(), dtype=np.float64)
            init_conf = init_conf/10 # convert to md_units
            conf_com = com(init_conf)

            init_mol = Chem.Mol(guest_mol)

            num_conformers = args.num_conformers

            # generate a set of gas phase conformers using the RDKit
            guest_mol.RemoveAllConformers()
            AllChem.EmbedMultipleConfs(guest_mol, num_conformers, randomSeed=2020)
            np.random.seed(2020)
            for conf_idx in range(num_conformers):
                conformer = guest_mol.GetConformer(conf_idx)
                guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
                guest_conf = guest_conf/10 # convert to md_units
                rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
                guest_conf = np.matmul(guest_conf, rot_matrix)*10

                for atom_idx, pos in enumerate(guest_conf):
                    conformer.SetAtomPosition(atom_idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    
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

            host_system = openmm_converter.deserialize_system(host_system)
            num_host_atoms = len(host_system.masses)

            # open_ff = forcefield.Forcefield("ff/smirnoff_1.1.0.py")
            open_ff = forcefield.Forcefield(args.forcefield)
            nrg_fns = open_ff.parameterize(guest_mol)
            guest_masses = get_masses(guest_mol)
            guest_system = system.System(nrg_fns, open_ff.params, open_ff.param_groups, guest_masses)

            combined_system = host_system.merge(guest_system)
            cbs = -1*np.ones_like(np.array(combined_system.masses))*0.0001
            lambda_idxs = np.zeros(len(combined_system.masses), dtype=np.int32)
            lambda_idxs[num_host_atoms:] = -1

            sim = simulation.Simulation(
                combined_system,
                step_sizes,
                cas,
                cbs,
                lambda_schedule,
                lambda_idxs,
                precision
            )

            epoch_train_params = get_params(opt_state)

            epoch_train_ff_params = copy.deepcopy(open_ff)
            epoch_train_ff_params.params = epoch_train_params[len(host_system.params):]
            fname = "epoch_"+str(epoch)+"_dG_"+str(true_dG)+"_params"
            fpath = os.path.join(args.out_dir, fname)
            epoch_train_ff_params.save(fpath)

            sim.system.params = epoch_train_params

            all_args = []

            child_conns = []
            parent_conns = []
            for conf_idx in range(num_conformers):

                conformer = guest_mol.GetConformer(conf_idx)
                guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
                guest_conf = guest_conf/10 # convert to md_units
                guest_conf = recenter(guest_conf, conf_com)
                x0 = np.concatenate([host_conf, guest_conf])       # combined geometry

                combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), init_mol)
                combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                out_file = os.path.join(args.out_dir, "epoch_"+str(epoch)+"_insertion_deletion_"+host_name+"_conf_"+str(conf_idx)+".pdb")
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
                all_du_dls.append(du_dls)

            all_du_dls = np.array(all_du_dls)
            loss_grad_fn = jax.grad(error_fn, argnums=(0,)) 

            error = error_fn(all_du_dls, T, lambda_schedule, true_dG)

            print("---EPOCH", epoch, "---Train---LOSS", error)
            with open(args.out_dir+"/loss.txt", "a+") as f:
                f.write("---EPOCH %d----Train---LOSS---- %f\n\n " % (epoch, error))

            error_grad = loss_grad_fn(all_du_dls, T, lambda_schedule, true_dG)
            all_du_dl_adjoints = error_grad[0]

            # send everything at once
            for pc, du_dl_adjoints in zip(parent_conns, all_du_dl_adjoints):
                pc.send(du_dl_adjoints)

            # receive everything at once
            all_dl_dps = []
            for pc in parent_conns:
                dl_dp = pc.recv()
                all_dl_dps.append(dl_dp)

            # terminate all the processes
            for p in processes:
                p.join()

            all_dl_dps = np.array(all_dl_dps)
            all_dl_dps = np.sum(all_dl_dps, axis=0)

            allowed_groups = {vary_allowed_groups[epoch][0]: vary_allowed_groups[epoch][1]}

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

        for data in test.iterbatches(1):

            guest_mol = copy.deepcopy(data[0][0])
            true_dG = data[0][1] # in kJ/mol and positive sign 

            init_conf = guest_mol.GetConformer(0)
            init_conf = np.array(init_conf.GetPositions(), dtype=np.float64)
            init_conf = init_conf/10 # convert to md_units
            conf_com = com(init_conf)

            init_mol = Chem.Mol(guest_mol)

            num_conformers = args.num_conformers

            # generate a set of gas phase conformers using the RDKit
            guest_mol.RemoveAllConformers()
            AllChem.EmbedMultipleConfs(guest_mol, num_conformers, randomSeed=2020)
            np.random.seed(2020)
            for conf_idx in range(num_conformers):
                conformer = guest_mol.GetConformer(conf_idx)
                guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
                guest_conf = guest_conf/10 # convert to md_units
                rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
                guest_conf = np.matmul(guest_conf, rot_matrix)*10

                for atom_idx, pos in enumerate(guest_conf):
                    conformer.SetAtomPosition(atom_idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    
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

            host_system = openmm_converter.deserialize_system(host_system)
            num_host_atoms = len(host_system.masses)

            # open_ff = forcefield.Forcefield("ff/smirnoff_1.1.0.py")
            open_ff = forcefield.Forcefield(args.forcefield)
            nrg_fns = open_ff.parameterize(guest_mol)
            guest_masses = get_masses(guest_mol)
            guest_system = system.System(nrg_fns, open_ff.params, open_ff.param_groups, guest_masses)

            combined_system = host_system.merge(guest_system)
            cbs = -1*np.ones_like(np.array(combined_system.masses))*0.0001
            lambda_idxs = np.zeros(len(combined_system.masses), dtype=np.int32)
            lambda_idxs[num_host_atoms:] = -1

            sim = simulation.Simulation(
                combined_system,
                step_sizes,
                cas,
                cbs,
                lambda_schedule,
                lambda_idxs,
                precision
            )

            epoch_train_params = get_params(opt_state)

            sim.system.params = epoch_train_params

            all_args = []

            child_conns = []
            parent_conns = []
            for conf_idx in range(num_conformers):

                conformer = guest_mol.GetConformer(conf_idx)
                guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
                guest_conf = guest_conf/10 # convert to md_units
                guest_conf = recenter(guest_conf, conf_com)
                x0 = np.concatenate([host_conf, guest_conf])       # combined geometry

                combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), init_mol)
                combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                out_file = os.path.join(args.out_dir, "epoch_"+str(epoch)+"_insertion_deletion_"+host_name+"_conf_"+str(conf_idx)+".pdb")
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
                all_du_dls.append(du_dls)

            all_du_dls = np.array(all_du_dls)
            loss_grad_fn = jax.grad(error_fn, argnums=(0,)) 

            error = error_fn(all_du_dls, T, lambda_schedule, true_dG)

            print("---EPOCH", epoch, "---Test---LOSS", error)
            with open(args.out_dir+"/loss.txt", "a+") as f:
                f.write("---EPOCH %d----Test---LOSS---- %f\n\n " % (epoch, error))






