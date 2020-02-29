import argparse
import time
import numpy as np
from io import StringIO
import itertools
import gnuplotlib as gp
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
from simtk.openmm.app import forcefield as ff
from simtk.openmm.app import PDBFile

from timemachine.lib import custom_ops, ops
from fe.utils import to_md_units, write
from fe import math_utils

import multiprocessing
from matplotlib import pyplot as plt

from jax.experimental import optimizers

from fe import simulation
from fe import loss


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
    dG_fwd = math_utils.trapz(fwd, fwd_sched)
    dG_bkwd = math_utils.trapz(bkwd, bkwd_sched)
    pred_dG = loss.mybar(jnp.stack([dG_fwd, dG_bkwd]))

    return jnp.abs(pred_dG - true_dG)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_pdb', type=str)

    parser.add_argument('--precision', type=str, required=True)    
    parser.add_argument('--complex_pdb', type=str, required=True)
    parser.add_argument('--ligand_sdf', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--jobs_per_gpu', type=int, required=True)
    parser.add_argument('--num_conformers', type=int, required=True)
    args = parser.parse_args()


    if args.precision == 'single':
        precision = np.float32
    elif args.precision == 'double':
        precision = np.float64
    else:
        raise Exception("precision must be either single or double")

    suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)
    for guest_mol in suppl:
        break

    T = 10000
    dt = 0.0015
    step_sizes = np.ones(T)*dt
    # cas = np.ones(T)*0.99
    cas = np.ones(T)*0.92

    num_gpus = args.num_gpus
    num_workers = args.num_gpus*args.jobs_per_gpu

    print('Creating multiprocessing pool with',args.num_gpus, 'gpus and', args.jobs_per_gpu, 'jobs per gpu')
    pool = multiprocessing.Pool(num_workers)

    all_du_dls = []

    assert T % 2 == 0

    # insertion deletion lambda_schedule
    # forward_schedule = np.linspace(0.00001, 0.99999, num=T//2)
    forward_schedule = np.linspace(100, 1, num=T//2)
    backward_schedule = np.linspace(1, 100, num=T//2)
    lambda_schedule = np.concatenate([forward_schedule, backward_schedule])

    epoch = 0

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

    sim = simulation.Simulation(
        guest_mol,
        host_pdb,
        'insertion',
        step_sizes,
        cas,
        lambda_schedule
    )

    initial_params = sim.combined_params

    opt_state = opt_init(initial_params)

    num_epochs = 100
    for epoch in range(num_epochs):
        # sample from the rdkit DG distribution (this can be changed later to another distribution later on)

        epoch_params = get_params(opt_state)
        sim.combined_params = epoch_params

        all_args = []
        for conf_idx in range(num_conformers):

            conformer = guest_mol.GetConformer(conf_idx)
            guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
            guest_conf = guest_conf/10 # convert to md_units
            guest_conf = recenter(guest_conf, conf_com)

            x0 = np.concatenate([host_conf, guest_conf])       # combined geometry

            combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), init_mol)
            combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
            out_file = os.path.join("frames", "epoch_"+str(epoch)+"_insertion_deletion_"+host_name+"_conf_"+str(conf_idx)+".pdb")
            writer = PDBWriter(combined_pdb_str, out_file)

            # set this to None if we don't care about visualization
            all_args.append([x0, writer, conf_idx % num_gpus, precision, None])

        all_du_dls = pool.map(sim.run_forward_multi, all_args)
        all_du_dls = np.array(all_du_dls)

        loss_grad_fn = jax.grad(error_fn, argnums=(0,))
      
        error = error_fn(all_du_dls, T, lambda_schedule, 100)

        print("---EPOCH", epoch, "---- LOSS", error)

        error_grad = loss_grad_fn(all_du_dls, T, lambda_schedule, 100)
        all_du_dl_adjoints = error_grad[0]

        for arg, adjoint in zip(all_args, all_du_dl_adjoints):
            arg[-1] = adjoint

        dl_dps = pool.map(sim.run_forward_multi, all_args)

        dl_dps = np.array(dl_dps)
        dl_dps = np.sum(dl_dps, axis=0)

        allowed_groups = {
            17: 0.5, # charge
            # 18: 1e-2, # atomic radii
            19: 1e-2 # scale factor
        }

        filtered_grad = []
        for g_idx, (g, gp) in enumerate(zip(dl_dps, sim.combined_param_groups)):
            if gp in allowed_groups:
                pf = allowed_groups[gp]
                filtered_grad.append(g*pf)
                if g != 0:
                    print("derivs", g_idx, '\t', g, '\t adjusted to', g*pf)
            else:
                filtered_grad.append(0)


        filtered_grad = np.array(filtered_grad)
        opt_state = opt_update(epoch, filtered_grad, opt_state)
            # print(g_idx, '\t', g, '\t', gp)







    # print("du_dl", grad)

    plt.subplot(2, 1, 1)

    insertion_work_vals = []
    deletion_work_vals = []
    for r_idx, r in enumerate(results):

        insertion_du_dls = r[:T//2]
        deletion_du_dls = r[T//2:]

        plt.plot(insertion_du_dls, label='insertion_'+str(r_idx))
        plt.plot(deletion_du_dls, label='deletion_'+str(r_idx))

        insertion_work = np.trapz(insertion_du_dls, forward_schedule)
        deletion_work = np.trapz(deletion_du_dls, backward_schedule)

        insertion_work_vals.append(insertion_work)
        deletion_work_vals.append(deletion_work)
    plt.title('work integrals')
    plt.ylabel('du_dl')
    plt.xlabel('lambda')
    plt.legend()
    # plt.show()
    plt.subplot(2, 1, 2)

    plt.title('insertion/deletion work dist.')
    plt.hist(insertion_work_vals, label='insertion')
    plt.hist(deletion_work_vals, label='deletion')
    plt.legend()
    plt.show()

        # all_du_dls.append(results)

    all_du_dls = np.array(all_du_dls)

    error = loss.EXP_loss(*all_du_dls, lambda_schedule, -20)

    print("Error", error)

    assert 0
