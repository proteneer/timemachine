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
from fe.utils import to_md_units, write
from fe import math_utils

from multiprocessing import Process, Pipe
from matplotlib import pyplot as plt

from jax.experimental import optimizers

from fe import simulation
from fe import loss, bar
from fe import atom_mapping
from fe.pdb_writer import PDBWriter

from ff import forcefield
from ff import system
from ff import openmm_converter
import jax.numpy as jnp

from rdkit.Chem import rdFMCS


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
    parser.add_argument('--num_windows', type=int, required=True, help='Number of lambda windows to be linearly spaced.')
    parser.add_argument('--n_frames', type=int, required=True, help='Number of PDB frames to write. If 0 then writing is skipped entirely.')
    parser.add_argument('--train', type=bool, required=True, help='Whether or not we train.')
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

    all_guest_mols = [all_guest_mols[0], all_guest_mols[2]]

    mol_a, mol_b = all_guest_mols

    mol_a_dG = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp("IC50[uM](SPA)")))
    mol_b_dG = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp("IC50[uM](SPA)")))

    a_name = mol_a.GetProp("_Name")
    b_name = mol_b.GetProp("_Name")

    print("Ligand A Name:", a_name)
    print("Ligand B Name:", b_name)

    print("LHS End State B (complex) A (solvent)")
    print("RHS End State A (complex) B (solvent)")

    a_to_b_map = atom_mapping.mcs_map(*all_guest_mols)

    print("Atom Mapping:", a_to_b_map)

    svg_a, svg_b = atom_mapping.draw_mapping(mol_a, mol_b, a_to_b_map)


    with open(os.path.join(args.out_dir, 'atom_mapping_A_'+str(a_name)+'.svg'), 'w') as fh:
        fh.write(svg_a)

    with open(os.path.join(args.out_dir, 'atom_mapping_B_'+str(b_name)+'.svg'), 'w') as fh:
        fh.write(svg_b)

    open_ff = forcefield.Forcefield(args.forcefield)
    all_nrg_fns = []

    # combined_masses = np.concatenate([a_masses, b_masses])

    print(len(open_ff.params))

    a_system = open_ff.parameterize(mol_a, cutoff=args.cutoff, am1=True)

    print(len(open_ff.params))

    b_system = open_ff.parameterize(mol_b, cutoff=args.cutoff, am1=True)

    print(len(open_ff.params))


    print(len(a_system.params))
    print(len(b_system.params))

    lhs_system, rhs_system = a_system.mix(b_system, a_to_b_map)

    host_pdb_file = args.protein_pdb
    host_pdb = app.PDBFile(host_pdb_file)
    amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    host_system = amber_ff.createSystem(host_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)

    host_system = openmm_converter.deserialize_system(host_system, cutoff=args.cutoff)
    lhs_combined_system = host_system.merge(lhs_system)
    rhs_combined_system = host_system.merge(rhs_system)

    # lr = 1e-2
    # opt_init, opt_update, get_params = optimizers.adam(lr)

    lr = 1e-3
    opt_init, opt_update, get_params = optimizers.sgd(lr)

    np.testing.assert_equal(lhs_combined_system.params, rhs_combined_system.params)

    opt_state = opt_init(lhs_combined_system.params)

    for epoch in range(100):

        print("Starting epoch -----"+str(epoch)+'-----')

        epoch_params = get_params(opt_state)

        temperature = 300
        dt = 1.5e-3
        friction = 40

        masses = np.array(lhs_combined_system.masses)
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
     
        complete_T = 12000
        equil_T = 2000

        ti_lambdas = np.linspace(0, 1, args.num_windows)
        # ti_lambdas = np.ones(args.num_windows)*0.1
        # ti_lambdas = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        # ti_lambdas = np.array([0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07])
        # all_du_dls = []

        all_processes = []
        all_pcs = []

        for lambda_idx, lamb in enumerate(ti_lambdas):

            complete_lambda = np.zeros(complete_T) + lamb
            complete_cas = np.ones(complete_T)*ca
            complete_dts = np.concatenate([
                np.linspace(0, dt, equil_T),
                np.ones(complete_T-equil_T)*dt
            ])

            sim = simulation.Simulation(
                lhs_combined_system,
                rhs_combined_system,
                complete_dts,
                complete_cas,
                cbs,
                ccs,
                complete_lambda,
                precision
            )

            intg_seed = np.random.randint(np.iinfo(np.int32).max)
            # intg_seed = 2020

            combined_ligand = Chem.CombineMols(mol_a, mol_b)
            combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), combined_ligand)
            combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
            out_file = os.path.join(args.out_dir, str(epoch)+"_rbfe_"+str(lambda_idx)+".pdb")
            writer = PDBWriter(combined_pdb_str, out_file, args.n_frames)

            # zero-out
            # if args.n_frames is 0:
                # writer = None
            # writer = None

            host_conf = []
            for x,y,z in host_pdb.positions:
                host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
            host_conf = np.array(host_conf)

            conformer = mol_a.GetConformer(0)
            mol_a_conf = np.array(conformer.GetPositions(), dtype=np.float64)
            mol_a_conf = mol_a_conf/10 # convert to md_units

            conformer = mol_b.GetConformer(0)
            mol_b_conf = np.array(conformer.GetPositions(), dtype=np.float64)
            mol_b_conf = mol_b_conf/10 # convert to md_units

            x0 = np.concatenate([host_conf, mol_a_conf, mol_b_conf]) # combined geometry
            v0 = np.zeros_like(x0)

            parent_conn, child_conn = Pipe()

            input_args = (x0, v0, epoch_params, intg_seed, writer, child_conn, lambda_idx % args.num_gpus)
            p = Process(target=sim.run_forward_and_backward, args=input_args)

            all_pcs.append(parent_conn)
            all_processes.append(p)

        sum_du_dls = [] # [L, T]
        all_du_dls = [] # [L, F, T] num lambda windows, num forces, num steps

        # run inference loop to generate all_du_dls
        for b_idx in range(0, len(all_processes), args.num_gpus):
            for p in all_processes[b_idx:b_idx+args.num_gpus]:
                p.start()

            batch_du_dls = []
            for pc_idx, pc in enumerate(all_pcs[b_idx:b_idx+args.num_gpus]):

                lamb_idx = b_idx+pc_idx
                lamb = ti_lambdas[b_idx+pc_idx]

                offset = equil_T # TBD use this
                full_du_dls = pc.recv() # F, T
                pc.send(None)
                assert full_du_dls is not None
                total_du_dls = np.sum(full_du_dls, axis=0)

                plt.plot(total_du_dls, label="{:.2f}".format(lamb))
                plt.ylabel("du_dl")
                plt.xlabel("timestep")
                plt.legend()

                fpath = os.path.join(args.out_dir, str(epoch)+"_lambda_du_dls_"+str(lamb_idx))
                plt.savefig(fpath)
                plt.close()

                sum_du_dls.append(total_du_dls)
                all_du_dls.append(full_du_dls)

        # compute loss and derivatives w.r.t. adjoints
        true_ddG = mol_a_dG - mol_b_dG
        all_du_dls = np.array(all_du_dls)

        loss = loss_fn(all_du_dls, true_ddG, ti_lambdas)
        print("loss", loss, "pred_ddG", np.trapz(np.mean(sum_du_dls, axis=1), ti_lambdas), "true_ddG", true_ddG)

        plt.close()
        plt.violinplot(sum_du_dls, positions=ti_lambdas)
        plt.ylabel("du_dlambda")
        plt.savefig(os.path.join(args.out_dir, str(epoch)+"_violin_du_dls"))
        plt.close()

        plt.boxplot(sum_du_dls, positions=ti_lambdas)
        plt.ylabel("du_dlambda")
        plt.savefig(os.path.join(args.out_dir, str(epoch)+"_boxplot_du_dls"))
        plt.close()

        # if args.train is True:

        #     all_du_dl_adjoints = loss_fn_grad(all_du_dls, true_ddG, ti_lambdas)[0]

        #     # compute the adjoints
        #     all_dl_dps = []
        #     for b_idx in range(0, len(all_processes), args.num_gpus):

        #         # kick off batch
        #         for pc_idx, pc in enumerate(all_pcs[b_idx:b_idx+args.num_gpus]):
        #             lamb_idx = b_idx+pc_idx
        #             du_dl_adjoints = all_du_dl_adjoints[lamb_idx]
        #             pc.send(du_dl_adjoints)


        #         # sync the batch
        #         for pc_idx, pc in enumerate(all_pcs[b_idx:b_idx+args.num_gpus]):
        #             all_dl_dps.append(pc.recv())

        #     for p in all_processes:
        #         p.join()

        #     dl_dps = np.sum(all_dl_dps, axis=0)

        #     allowed_groups = {
        #         # 7: 0.5,
        #         14: 0.5, # small_molecule charge
        #         12: 1e-2, # GB atomic radii
        #         13: 1e-2 # GB scale factor
        #     }

        #     filtered_grad = []
        #     for g_idx, (g, gp) in enumerate(zip(dl_dps, lhs_combined_system.param_groups)):
        #         if gp in allowed_groups:
        #             pf = allowed_groups[gp]
        #             filtered_grad.append(g*pf)
        #             if g != 0:
        #                 print("derivs", g_idx, '\t group', gp, '\t', g, '\t adjusted to', g*pf, '\t old val', lhs_combined_system.params[g_idx])
        #         else:
        #             filtered_grad.append(0)


        #     print("Epoch", epoch, "pred_ddG LHS->RHS (B to A)", np.trapz(np.mean(sum_du_dls, axis=-1), ti_lambdas), "true ddG", true_ddG, "loss", loss)

        #     filtered_grad = np.array(filtered_grad)
        #     opt_state = opt_update(epoch, filtered_grad, opt_state)
