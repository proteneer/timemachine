import argparse
import time
import numpy as np
from io import StringIO
import itertools
import gnuplotlib as gp
import os
import sys
import random

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)
from jax.experimental import optimizers

from scipy.stats import special_ortho_group
import jax
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from simtk.openmm import app
from simtk.openmm.app import forcefield as ff
from simtk.openmm.app import PDBFile

from timemachine.lib import custom_ops, ops
from fe.utils import to_md_units, write
from fe import math_utils
from fe import simulation, loss, dataset
from system import serialize, forcefield

import multiprocessing
from matplotlib import pyplot as plt

from openforcefield.typing.engines import smirnoff

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
    dG_fwd /= kT
    dG_bkwd /= kT
    pred_dG = loss.mybar(jnp.stack([dG_fwd, dG_bkwd]))
    pred_dG *= kT

    return jnp.abs(pred_dG - true_dG)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--precision', type=str, required=True)    
    parser.add_argument('--complex_pdb', type=str, required=True)

    """The ligands in ligand_sdf_file should be listed in the same order as the true_dG_file"""
    parser.add_argument('--ligand_sdf_file', type=str, nargs='*')
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--jobs_per_gpu', type=int, required=True)
    parser.add_argument('--num_conformers', type=int, required=True)

    """The dG values are in kJ/mol and positive sign for both insertion and deletion (1kcal/mol=4.184kJ/mol)"""
    parser.add_argument('--true_dG_file', type=str, nargs='*')

    args = parser.parse_args()

    assert os.path.isdir(args.frames_dir)

    if args.precision == 'single':
        precision = np.float32
    elif args.precision == 'double':
        precision = np.float64
    else:
        raise Exception("precision must be either single or double")

    num_gpus = args.num_gpus
    num_workers = args.num_gpus*args.jobs_per_gpu

    print('Creating multiprocessing pool with',args.num_gpus, 'gpus and', args.jobs_per_gpu, 'jobs per gpu')
    pool = multiprocessing.Pool(num_workers)

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
    dt = 0.001
    step_sizes = np.ones(T)*dt
    assert T % 2 == 0
    cas = np.ones(T)*0.93
    epoch = 0
    lr = 5e-4

    ligand_list = []
    for ligand_sdf_file, ligand_dG_file in zip(args.ligand_sdf_file, args.true_dG_file):
        ligand_sdf_file_readlines = open(ligand_sdf_file, 'r').readlines()
        ligand_dG_file_readlines = open(ligand_dG_file, 'r').readlines()

        for file_name, deltaG in zip(ligand_sdf_file_readlines, ligand_dG_file_readlines):
            
            for rd_mol in Chem.SDMolSupplier(file_name.rstrip('\n'), removeHs=False):
                break

            true_dG_value = float(deltaG.split()[0])
            ligand_list.append((rd_mol, true_dG_value))

    guest_mol = ligand_list[0][0]
        
    off = smirnoff.ForceField("test_forcefields/smirnoff99Frosst.offxml")
    _, (guest_params, _), _ = forcefield.parameterize(guest_mol, off, dimension=4)
    
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(guest_params)

    np.savetxt("original_off_params.txt", guest_params)
   
    host_pdb_file = args.complex_pdb
    host_pdb = app.PDBFile(host_pdb_file)
    host_conf = []
    for x,y,z in host_pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)
    host_name = "complex"  

    num_epochs = 100
    # torsion, charge, gb radii, gb scaling factor, torsion
    vary_allowed_groups = [4]*10 + [17]*20 + [18]*20 + [19]*20 + [4]*30

    assert len(vary_allowed_groups) == num_epochs

    ligand_dataset = dataset.Dataset(ligand_list)
    train, test = ligand_dataset.split(0.7)

    for epoch in range(num_epochs):
        # sample from the rdkit DG distribution (this can be changed later to another distribution later on)
        # epoch_params = get_params(opt_state)
        # guest_params = epoch_params

        train.shuffle()
        print(train.data)

###### Training ######
        for data in train.iterbatches(1):

            guest_mol = data[0][0]
            true_dG = data[0][1]

            current_off_params = get_params(opt_state)

            all_du_dls = []

            sim = simulation.Simulation(
                guest_mol,
                host_pdb,
                'insertion',
                step_sizes,
                cas,
                lambda_schedule,
                current_off_params
                )

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

            all_args = []
            for conf_idx in range(num_conformers):

                conformer = guest_mol.GetConformer(conf_idx)
                guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
                guest_conf = guest_conf/10 # convert to md_units
                guest_conf = recenter(guest_conf, conf_com)

                x0 = np.concatenate([host_conf, guest_conf])       # combined geometry

                combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), init_mol)
                combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                out_file = os.path.join(args.frames_dir, "epoch_"+str(epoch)+"_insertion_deletion_"+host_name+"_conf_"+str(conf_idx)+".pdb")
                writer = PDBWriter(combined_pdb_str, out_file)

                # set this to None if we don't care about visualization
                all_args.append([x0, writer, conf_idx % num_gpus, precision, None])

            all_du_dls = pool.map(sim.run_forward_multi, all_args)
            all_du_dls = np.array(all_du_dls)

            loss_grad_fn = jax.grad(error_fn, argnums=(0,))
          
            for du_dls in all_du_dls:
                fwd = du_dls[:T//2]
                bkwd = du_dls[T//2:]

            error = error_fn(all_du_dls, T, lambda_schedule, true_dG)

            print("---EPOCH", epoch, "---- LOSS of one ligand in training ----", error)

            error_grad = loss_grad_fn(all_du_dls, T, lambda_schedule, true_dG)
            all_du_dl_adjoints = error_grad[0]

            # set the adjoints for reverse mode
            for arg_idx, arg in enumerate(all_args):
                arg[-1] = all_du_dl_adjoints[arg_idx]

            dl_dps = pool.map(sim.run_forward_multi, all_args)
            dl_dps = np.array(dl_dps)
            dl_dps = np.sum(dl_dps, axis=0)

            allowed_groups = {vary_allowed_groups[epoch]: 1e-3}        

            filtered_grad = []
            for g_idx, (g, gp) in enumerate(zip(dl_dps, sim.guest_param_groups)):
                if gp in allowed_groups:
                    pf = allowed_groups[gp]
                    filtered_grad.append(g*pf)
                    #if g != 0:
                    #    print("derivs", g_idx, '\t', g, '\t adjusted to', g*pf, '\t old val', current_off_params[g_idx])
                else:
                    filtered_grad.append(0)

            filtered_grad = np.array(filtered_grad)
            opt_state = opt_update(epoch, filtered_grad, opt_state)

######## Testing ##########
        for data in test.iterbatches(1):

            guest_mol = data[0][0]
            true_dG = data[0][1]

            current_off_params = get_params(opt_state)

            all_du_dls = []

            sim = simulation.Simulation(
                guest_mol,
                host_pdb,
                'insertion',
                step_sizes,
                cas,
                lambda_schedule,
                current_off_params
                )

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

            all_args = []
            for conf_idx in range(num_conformers):

                conformer = guest_mol.GetConformer(conf_idx)
                guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
                guest_conf = guest_conf/10 # convert to md_units
                guest_conf = recenter(guest_conf, conf_com)

                x0 = np.concatenate([host_conf, guest_conf])       # combined geometry

                combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), init_mol)
                combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                out_file = os.path.join(args.frames_dir, "epoch_"+str(epoch)+"_insertion_deletion_"+host_name+"_conf_"+str(conf_idx)+".pdb")
                writer = PDBWriter(combined_pdb_str, out_file)

                # set this to None if we don't care about visualization
                all_args.append([x0, writer, conf_idx % num_gpus, precision, None])

            all_du_dls = pool.map(sim.run_forward_multi, all_args)
            all_du_dls = np.array(all_du_dls)

            loss_grad_fn = jax.grad(error_fn, argnums=(0,))
          
            for du_dls in all_du_dls:
                fwd = du_dls[:T//2]
                bkwd = du_dls[T//2:]

            error = error_fn(all_du_dls, T, lambda_schedule, true_dG)

            print("---EPOCH", epoch, "---- LOSS of one ligand in test ----", error)

    pool.close()


