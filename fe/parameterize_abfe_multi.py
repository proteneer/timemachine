import copy
import argparse
import time
import numpy as np
from io import StringIO
import itertools
import os
import sys

from timemachine.integrator import langevin_coefficients

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
from timemachine.potentials import jax_utils
from fe.utils import to_md_units, write
from fe import math_utils

from multiprocessing import Process, Pipe
from matplotlib import pyplot as plt

from jax.experimental import optimizers

from fe import simulation_v2
from fe import loss, bar
from fe.pdb_writer import PDBWriter

from ff import forcefield
from ff import system
from ff import openmm_converter
import jax.numpy as jnp

from rdkit.Chem import rdFMCS


# multistage protocol for RBFE.


def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18


def loss_fn(all_du_dls, true_ddG, lambda_schedule):
    """
    Loss function. Currently set to L1.

    Parameters:
    -----------
    all_du_dls: shape [L, F, T] np.array
        Where L is the number of lambda windows, F is the number of forces, and T is the total number of equilibrated steps

    true_ddG: scalar
        True ddG of the edge.

    Returns:
    --------
    scalar
        Loss

    """
    assert all_du_dls.ndim == 3

    total_du_dls = jnp.sum(all_du_dls, axis=1) # shape [L, T]
    mean_du_dls = jnp.mean(total_du_dls, axis=1) # shape [L]
    pred_ddG = math_utils.trapz(mean_du_dls, lambda_schedule)
    return jnp.abs(pred_ddG - true_ddG)

loss_fn_grad = jax.grad(loss_fn, argnums=(0,))

def setup_lambda_idxs(nrg_fns):
    es_param_idxs, lj_param_idxs, exc_idxs, es_exc_param_idxs, lj_exc_param_idxs, cutoff = nrg_fns['Nonbonded']
    gb_args = nrg_fns['GBSA']
    gb_charges, gb_radii, gb_scales = gb_args[:3]
    n_a = len(es_param_idxs)

    lambda_plane_idxs = np.zeros(n_a, dtype=np.int32)
    lambda_offset_idxs = np.ones(n_a, dtype=np.int32)

    nrg_fns['Nonbonded'] = es_param_idxs, lj_param_idxs, exc_idxs, es_exc_param_idxs, lj_exc_param_idxs, lambda_plane_idxs, lambda_offset_idxs, cutoff
    nrg_fns['GBSA'] = (gb_charges, gb_radii, gb_scales, lambda_plane_idxs, lambda_offset_idxs, *gb_args[3:])

# def find_pocket_neighbors(conf, n_host, cutoff=0.5):
#     """
#     Find all protein atoms that we within cutoff of a ligand atom.
#     """
#     ri = np.expand_dims(conf, axis=0)
#     rj = np.expand_dims(conf, axis=1)
#     dij = jax_utils.distance(ri, rj)
#     all_nbs = []
#     for l_idx, dists in enumerate(dij[n_host:]):
#         nbs = np.argwhere(dists[:n_host] < cutoff)
#         all_nbs.extend(nbs.reshape(-1).tolist())

#     return list(set(all_nbs))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Relative Binding Free Energy Script')
    parser.add_argument('--out_dir', type=str, required=True, help='Location of all output files')
    parser.add_argument('--precision', type=str, required=True, help='Either single or double precision. Double is 8x slower.')
    parser.add_argument('--protein_pdb', type=str, required=True, help='Prepared protein PDB file. This should not have any waters.')
    parser.add_argument('--ligand_sdf', type=str, required=True, help='The ligand sdf used along with posed 3D coordinates. Only the first two ligands are used.')
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of gpus available.')
    parser.add_argument('--forcefield', type=str, required=True, help='Small molecule forcefield to be loaded.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed used for all the random number generators.')
    parser.add_argument('--cutoff', type=float, required=True, help='Nonbonded cutoff. Please set this to 1.0 for now.')
    parser.add_argument('--lamb', type=float, required=False, help='Which lambda window we run at.')
    parser.add_argument('--n_frames', type=int, required=True, help='Number of PDB frames to write. If 0 then writing is skipped entirely.')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps we run')
    parser.add_argument('--a_idx', type=int, required=True, help='A index')

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
    for guest_idx, guest_mol in enumerate(suppl):
        all_guest_mols.append(guest_mol)

    mol_a = all_guest_mols[args.a_idx]

    mol_a_dG = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp("IC50[uM](SPA)")))
    # mol_b_dG = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp("IC50[uM](SPA)")))

    a_name = mol_a.GetProp("_Name")
    # b_name = mol_b.GetProp("_Name")

    print("Ligand A Name:", a_name)
 

    open_ff = forcefield.Forcefield(args.forcefield)
    all_nrg_fns = []

    # a_system = open_ff.parameterize(mol_a, cutoff=args.cutoff, am1=True, zero_charges=True)
    a_system = open_ff.parameterize(mol_a, cutoff=args.cutoff, am1=True, zero_charges=False)

    setup_lambda_idxs(a_system.nrg_fns)

    host_pdb_file = args.protein_pdb
    host_pdb = app.PDBFile(host_pdb_file)
    amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    host_system = amber_ff.createSystem(
        host_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    lambda_plane_idxs = np.zeros(mol_a.GetNumAtoms())

    host_conf = []
    for x,y,z in host_pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)

    conformer = mol_a.GetConformer(0)
    mol_a_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    mol_a_conf = mol_a_conf/10 # convert to md_units

    x0 = np.concatenate([host_conf, mol_a_conf]) # combined geometry

    # pocket_atoms = find_pocket_neighbors(x0, host_conf.shape[0], cutoff=args.pocket_cutoff)
    host_system = openmm_converter.deserialize_system(host_system, cutoff=args.cutoff, pocket_atoms=None)
    # host_system = openmm_converter.deserialize_system(host_system, cutoff=args.cutoff, pocket_atoms=None)
    combined_system = host_system.merge(a_system)

    lr = 1e-3
    opt_init, opt_update, get_params = optimizers.sgd(lr)

    opt_state = opt_init(combined_system.params)

    for epoch in range(100):

        stage_ddGs = []

        print("Starting epoch -----"+str(epoch)+'-----')

        epoch_params = get_params(opt_state)

        temperature = 300
        dt = 1.5e-3
        friction = 40

        masses = np.array(combined_system.masses)
        ca, cbs, ccs = langevin_coefficients(
            temperature,
            dt,
            friction,
            masses
        )

        cbs *= -1

        print("Integrator coefficients:")
        print("ca", ca)
        print("cbs", cbs)
        print("ccs", ccs)
     

        # complete_T = 200000
        complete_T = args.steps
        # complete_T = 40000
        equil_T = 2000
        du_dl_cutoff = 20000

        assert complete_T > equil_T
        assert complete_T > du_dl_cutoff


        # ti_lambdas = np.linspace(0, 1, args.num_windows)
        # ti_lambdas = np.ones(args.num_windows)*0.2
        # ti_lambdas = np.array([0.07, 0.13, 0.20, 0.27, 0.33, 0.33, 0.33])
        # ti_lambdas = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # ti_lambdas = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        # ti_lambdas = np.array([0.00, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.8, 1.2, 1.5, 2.0, 4.0, 10.0])
        #                        0     1     2    3     4      5     6      7      8     9   10   11   12   13   14  15
        # ti_lambdas = np.array([0.0, 0.025, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.5])
        # ti_lambdas = np.array([0.0, 0.1, 0.2, 0.25])
        # ti_lambdas = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        # ti_lambdas = np.ones(8)*args.lamb
        # ti_lambdas = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50])

        # ti_lambdas = np.array([0.00, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
        # ti_lambdas = np.array([0.0, 0.2, 0.5])
        #
        if args.lamb:
            ti_lambdas = np.ones(args.num_gpus)*args.lamb
        else:
            ti_lambdas = np.concatenate([
                np.linspace(0.0, 0.5, 24, endpoint=False),
                np.linspace(0.5, 1.0, 8)
            ])
        
        all_processes = []
        all_pcs = []

        for lambda_idx, lamb in enumerate(ti_lambdas):

            complete_lambda = np.zeros(complete_T) + lamb
            complete_cas = np.ones(complete_T)*ca
            complete_dts = np.concatenate([
                np.linspace(0, dt, equil_T),
                np.ones(complete_T-equil_T)*dt
            ])

            sim = simulation_v2.Simulation(
                combined_system,
                complete_dts,
                complete_cas,
                cbs,
                ccs,
                complete_lambda,
                precision
            )

            intg_seed = np.random.randint(np.iinfo(np.int32).max)
            # intg_seed = 2020

            combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), mol_a)
            combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
            out_file = os.path.join(args.out_dir, str(epoch)+"_rbfe_"+str(lambda_idx)+".pdb")
            writer = PDBWriter(combined_pdb_str, out_file, args.n_frames)

            # zero-out
            # if args.n_frames is 0:
                # writer = None
            # writer = None

            v0 = np.zeros_like(x0)

            parent_conn, child_conn = Pipe()

            input_args = (x0, v0, epoch_params, intg_seed, writer, child_conn, lambda_idx % args.num_gpus, du_dl_cutoff)
            p = Process(target=sim.run_forward_and_backward, args=input_args)

            # sim.run_forward_and_backward(*input_args)

            all_pcs.append(parent_conn)
            all_processes.append(p)

        sum_du_dls = [] # [L, T]
        all_du_dls = [] # [L, F, T] num lambda windows, num forces, num steps

        all_energies = []

        # run inference loop to generate all_du_dls
        for b_idx in range(0, len(all_processes), args.num_gpus):
            for p in all_processes[b_idx:b_idx+args.num_gpus]:
                p.start()

            batch_du_dls = []
            for pc_idx, pc in enumerate(all_pcs[b_idx:b_idx+args.num_gpus]):

                lamb_idx = b_idx+pc_idx
                lamb = ti_lambdas[b_idx+pc_idx]

                offset = equil_T # TBD use this
                full_du_dls, full_energies = pc.recv() # (F, T), (T)
                pc.send(None)
                assert full_du_dls is not None
                total_du_dls = np.sum(full_du_dls, axis=0)

                plt.plot(total_du_dls, label="{:.2f}".format(lamb))
                plt.ylabel("du_dl")
                plt.xlabel("timestep")
                plt.legend()
                fpath = os.path.join(args.out_dir, str(epoch)+"_lambda_du_dls_"+str(lamb_idx))
                plt.savefig(fpath)
                plt.clf()

                plt.plot(full_energies, label="{:.2f}".format(lamb))
                plt.ylabel("U")
                plt.xlabel("timestep")
                plt.legend()

                fpath = os.path.join(args.out_dir, str(epoch)+"_lambda_energies_"+str(lamb_idx))
                plt.savefig(fpath)
                plt.clf()

                sum_du_dls.append(total_du_dls)
                all_du_dls.append(full_du_dls)



        # compute loss and derivatives w.r.t. adjoints
        all_du_dls = np.array(all_du_dls)
        sum_du_dls = np.array(sum_du_dls)

        safe_T = du_dl_cutoff

        # loss = loss_fn(all_du_dls[:, :, safe_T:], true_ddG, ti_lambdas)

        stage_ddG = np.trapz(np.mean(sum_du_dls[:, safe_T:], axis=1), ti_lambdas)
        print("pred_ddG", stage_ddG)

        plt.clf()
        plt.violinplot(sum_du_dls[:, safe_T:].tolist(), positions=ti_lambdas)
        plt.ylabel("du_dlambda")
        plt.savefig(os.path.join(args.out_dir, str(epoch)+"_violin_du_dls"))
        plt.clf()

        plt.boxplot(sum_du_dls[:, safe_T:].tolist(), positions=ti_lambdas)
        plt.ylabel("du_dlambda")
        plt.savefig(os.path.join(args.out_dir, str(epoch)+"_boxplot_du_dls"))
        plt.clf()
