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
from fe.pdb_writer import PDBWriter

from ff import forcefield
from ff import system
from ff import openmm_converter
import jax.numpy as jnp

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18

def get_masses(m):
    masses = []
    for a in m.GetAtoms():
        masses.append(a.GetMass())
    return masses

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--precision', type=str, required=True)    
    parser.add_argument('--complex_pdb', type=str, required=True)
    parser.add_argument('--ligand_sdf', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
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
    for guest_idx, guest_mol in enumerate(suppl):
        all_guest_mols.append(guest_mol)

    for _, _ in enumerate(all_guest_mols):

        guest_idx = 0
        guest_mol = all_guest_mols[0]

        name = guest_mol.GetProp("_Name")
        true_dG = convert_uIC50_to_kJ_per_mole(float(guest_mol.GetProp("IC50[uM](SPA)")))

        print("====== Processing",str(guest_idx),name,Chem.MolToSmiles(guest_mol, isomericSmiles=True),"========")

        num_gpus = args.num_gpus

        host_pdb_file = args.complex_pdb
        host_pdb = app.PDBFile(host_pdb_file)
        host_conf = []
        for x,y,z in host_pdb.positions:
            host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
        host_conf = np.array(host_conf)
        host_name = "complex"

        conformer = guest_mol.GetConformer(0)
        guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
        guest_conf = guest_conf/10 # convert to md_units
        x0 = np.concatenate([host_conf, guest_conf]) # combined geometry
        # thermostat will bring velocities to the correct temperature
        v0 = np.zeros_like(x0)

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

        open_ff = forcefield.Forcefield(args.forcefield)
        nrg_fns = open_ff.parameterize(guest_mol, cutoff=cutoff, am1=True)
        guest_masses = get_masses(guest_mol)
        guest_system = system.System(nrg_fns, open_ff.params, open_ff.param_groups, guest_masses)

        combined_system = host_system.merge(guest_system)


        # lr = 5e-4
        # opt_init, opt_update, get_params = optimizers.adam(lr)
        # opt_init, opt_update, get_params = optimizers.sgd(lr)

        # opt_state = opt_init(initial_params)

        # num_epochs = 100

        # for epoch in range(num_epochs):

        temperature = 300
        # dt = 1.5e-3
        dt = 1.5e-3
        friction = 50

        masses = np.array(combined_system.masses)
        # masses = np.where(masses < 2.0, masses*8, masses)

        ca, cbs, ccs = langevin_coefficients(
            temperature,
            dt,
            friction,
            masses
        )

        cbs *= -1

        # print("Integrator coefficients:")
        # print("ca", ca)
        # print("cbs", cbs)
        # print("ccs", ccs)



        lambda_schedule = [0.05, 0.1, 0.15, 0.2, 0.225, 0.25, 0.28, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5, 0.7, 0.9, 1.0]
        # lambda_schedule = [0.1, 0.4, 1.0]

        # lambda_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # lambda_schedule = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        # lambda_schedule = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        # lambda_schedule = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
        # lambda_schedule = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        # lambda_schedule = [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25]

        lambda_schedule = np.array(lambda_schedule)

        simulations = []

        for lamb in lambda_schedule:

            # insertion only
            Ts = [
                1000, # insertion/minimization
                14000, # equilibriation
            ]

            offset = 3000

            complete_T = np.sum(Ts)

            complete_lambda = np.concatenate([
                np.linspace(cutoff, lamb, Ts[0]),
                np.linspace(lamb,   lamb, Ts[1]),
            ])

            complete_cas = np.concatenate([
                np.linspace(0.5, ca, Ts[0]),
                np.linspace(ca, ca, Ts[1]),
            ])

            complete_dts = np.concatenate([
                np.linspace(dt, dt, Ts[0]),
                np.linspace(dt, dt, Ts[1]),
            ])



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
                precision
            )

            simulations.append(sim)

        all_args = []
        child_conns = []
        parent_conns = []
        processes = []

        epoch = 0

        for sim_idx, sim in enumerate(simulations):

            combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), guest_mol)
            combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
            out_file = os.path.join(args.out_dir, "eq_"+str(sim_idx)+".pdb")
            writer = PDBWriter(combined_pdb_str, out_file)

            parent_conn, child_conn = Pipe()
            parent_conns.append(parent_conn)

            intg_seed = np.random.randint(np.iinfo(np.int32).max)

            arg = (x0, v0, intg_seed, writer, child_conn, sim_idx % args.num_gpus)

            p = Process(target=sim.run_forward_and_backward, args=arg)
            p.daemon = True
            processes.append(p)


        assert len(processes) == len(parent_conns)
        # start in batches of num_gpus

        def subsample(du_dls):
            us = []
            for u_idx, u in enumerate(du_dls):
                if u_idx % 100 == 0:
                    us.append(u)
            return np.array(us)

        mean_du_dls = []
        all_du_dls = []
        std_du_dls = []
        for b_idx in range(0, len(processes), args.num_gpus):
            for p in processes[b_idx:b_idx+args.num_gpus]:
                p.start()

            batch_du_dls = []
            for pc_idx, pc in enumerate(parent_conns[b_idx:b_idx+args.num_gpus]):

                lamb = lambda_schedule[b_idx+pc_idx]

                du_dls = pc.recv()[offset:]
                # du_dls = subsample(du_dls)

                pc.send(None)

                mean_du_dls.append(np.mean(du_dls))
                std_du_dls.append(np.std(du_dls))
                print("lamb", lamb, "mean/std", np.mean(du_dls), np.std(du_dls))

                assert du_dls is not None
                plt.plot(du_dls, label=str(lamb))

                plt.ylabel("du_dl")
                plt.xlabel("timestep")
                plt.legend()
                plt.savefig(os.path.join(args.out_dir, "epoch"+str(epoch)+"_du_dls"))

                all_du_dls.append(du_dls)

            for p in processes[b_idx:b_idx+args.num_gpus]:
                p.join()




        plt.close()


        plt.violinplot(all_du_dls, positions=lambda_schedule)
        plt.ylabel("du_dlambda")
        plt.savefig(os.path.join(args.out_dir, "violin_du_dls"))
        plt.close()


        plt.boxplot(all_du_dls, positions=lambda_schedule)
        plt.ylabel("du_dlambda")
        plt.savefig(os.path.join(args.out_dir, "boxplot_du_dls"))
        plt.close()



        mean_du_dls = np.concatenate([[0], mean_du_dls, [0]])
        lambda_schedule = np.concatenate([[0], lambda_schedule, [1.25]])


        print(name, "pred_dG", np.trapz(mean_du_dls, lambda_schedule), "true_dG", true_dG)

