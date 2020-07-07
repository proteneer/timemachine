import matplotlib
matplotlib.use('Agg')

import copy
import argparse
import time
import datetime
import numpy as np
from io import StringIO
import itertools
import os
import sys

from fe import runner

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)

from scipy.stats import special_ortho_group
import jax
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


from simtk.openmm.app import PDBFile


# from fe.utils import to_md_units, write
from fe import math_utils, setup_system

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

    parser = argparse.ArgumentParser(description='Absolute Binding Free Energy Script')
    parser.add_argument('--out_dir', type=str, required=True, help='Location of all output files')
    parser.add_argument('--precision', type=str, required=True, help='Either single or double precision. Double is 8x slower.')
    parser.add_argument('--protein_pdb', type=str, required=True, help='Prepared protein PDB file. This should not have any waters.')
    parser.add_argument('--ligand_sdf', type=str, required=True, help='The ligand sdf used along with posed 3D coordinates. Only the first two ligands are used.')
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of gpus available.')
    parser.add_argument('--forcefield', type=str, required=True, help='Small molecule forcefield to be loaded.')
    parser.add_argument('--lamb', type=float, required=False, help='Which lambda window we run at.')
    parser.add_argument('--n_frames', type=int, required=True, help='Number of PDB frames to write. If 0 then writing is skipped entirely.')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps we run')
    parser.add_argument('--a_idx', type=int, required=True, help='A index')
    parser.add_argument('--restr_force', type=float, required=True, help='Strength of the each restraint term, in kJ/mol.')
    parser.add_argument('--restr_alpha', type=float, required=True, help='Width of the well.')
    parser.add_argument('--restr_count', type=int, required=True, help='Number of host atoms we restrain each core atom to.')

    args = parser.parse_args()


    print(r"""launching the timemachine.""")

    print("Launch Time:", datetime.datetime.now())
    print("Arguments:", " ".join(sys.argv))

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
    num_guest_atoms = mol_a.GetNumAtoms()
    mol_a_dG = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp("IC50[uM](SPA)")))

    a_name = mol_a.GetProp("_Name")

    # process the host first

    host_pdb_file = args.protein_pdb
    host_pdb = PDBFile(host_pdb_file)
    # amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    # host_system = amber_ff.createSystem(
    #     host_pdb.topology,
    #     nonbondedMethod=app.NoCutoff,
    #     constraints=None,
    #     rigidWater=False
    # )


    core_smarts = '[#6]1:[#6]:[#6]:[#6](:[#6](:[#6]:1-[#8]-[#6](:[#6]-[#1]):[#6])-[#1])-[#1]'

    print("Using core smarts:", core_smarts)
    core_query = Chem.MolFromSmarts(core_smarts)
    core_atoms = mol_a.GetSubstructMatch(core_query)

    # stage = 0
    # lamb = 0.25


    stage_dGs = []

    epoch = 0
    epoch_dir = os.path.join(args.out_dir, "epoch_"+str(epoch))

    for stage in [0,1,2]:

        print("---Starting stage", stage, '---')
        stage_dir = os.path.join(epoch_dir, "stage_"+str(stage))

        if not os.path.exists(stage_dir):
            os.makedirs(stage_dir)

        x0, combined_masses, final_gradients = setup_system.create_system(
            mol_a,
            host_pdb,
            args.forcefield,
            stage,
            core_atoms,
            args.restr_force,
            args.restr_alpha,
            args.restr_count
        )

        if stage == 0:
            # we need to goto a larger lambda for the morse potential to decay to zero.
            ti_lambdas = np.linspace(7.0, 0.0, 32)
        elif stage == 1:
            # lambda spans from [0, inf], is close enough to zero over [0, 1.2] cutoff
            ti_lambdas = np.concatenate([
                np.linspace(0.0, 0.5, 24, endpoint=False),
                np.linspace(0.5, 1.2, 8)
            ])
        elif stage == 2:
            # we need to goto a larger lambda for the morse potential to decay to zero.
            ti_lambdas = np.linspace(0.0, 7.0, 32)
        else:
            raise Exception("Unknown stage.")


        all_processes = []
        all_pipes = []
        for lambda_idx, lamb in enumerate(ti_lambdas):

            intg = setup_system.Integrator(
                steps=args.steps,
                dt=1.5e-3,
                temperature=300.0,
                friction=40.0,
                masses=combined_masses,
                lamb=lamb,
                seed=np.random.randint(np.iinfo(np.int32).max)
            )

            system = setup_system.System(
                x0,
                np.zeros_like(x0),
                final_gradients,
                intg
            )

            gpu_idx = lambda_idx % args.num_gpus
            parent_conn, child_conn = Pipe()
            # runner.simulate(system, precision, gpu_idx, child_conn)
            # assert 0

            p = Process(target=runner.simulate, args=(system, precision, gpu_idx, child_conn))
            all_processes.append(p)
            all_pipes.append(parent_conn)

        stage_du_dls = []
        for b_idx in range(0, len(all_processes), args.num_gpus):

            for p in all_processes[b_idx:b_idx+args.num_gpus]:
                p.start()

            batch_du_dls = []

            for pc_idx, pc in enumerate(all_pipes[b_idx:b_idx+args.num_gpus]):

                lamb_idx = b_idx+pc_idx
                lamb = ti_lambdas[b_idx+pc_idx]

                full_du_dls, full_energies = pc.recv() # (F, T), (T)

                # pc.send(None)
                assert full_du_dls is not None

                np.save(os.path.join(stage_dir, "lambda_"+str(lamb_idx)+"_full_du_dls"), full_du_dls)
                total_du_dls = np.sum(full_du_dls, axis=0)

                plt.plot(total_du_dls, label="{:.2f}".format(lamb))
                plt.ylabel("du_dl")
                plt.xlabel("timestep")
                plt.legend()
                fpath = os.path.join(stage_dir, "lambda_du_dls_"+str(lamb_idx))
                plt.savefig(fpath)
                plt.clf()

                plt.plot(full_energies, label="{:.2f}".format(lamb))
                plt.ylabel("U")
                plt.xlabel("timestep")
                plt.legend()

                fpath = os.path.join(stage_dir, "lambda_energies_"+str(lamb_idx))
                plt.savefig(fpath)
                plt.clf()

                du_dl_cutoff = 4000
                equil_du_dls = full_du_dls[:, du_dl_cutoff:]

                for f, du_dls in zip(final_gradients, equil_du_dls):
                    fname = f[0]
                    print("lambda:", "{:.3f}".format(lamb), "\t median {:8.2f}".format(np.median(du_dls)), "\t mean", "{:8.2f}".format(np.mean(du_dls)), "+-", "{:7.2f}".format(np.std(du_dls)), "\t <-", fname)

                total_equil_du_dls = np.sum(equil_du_dls, axis=0) # [1, T]
                print("lambda:", "{:.3f}".format(lamb), "\t mean", "{:8.2f}".format(np.mean(total_equil_du_dls)), "+-", "{:7.2f}".format(np.std(total_equil_du_dls)), "\t <- Total")

                stage_du_dls.append(total_equil_du_dls)

        stage_dG = np.trapz(np.mean(stage_du_dls, axis=1), ti_lambdas)
        print("stage", stage, "pred_dG", stage_dG)
        stage_dGs.append(stage_dG)

        plt.boxplot(stage_du_dls, positions=ti_lambdas)
        plt.ylabel(r"$\frac{du}{d\lambda}$")
        plt.savefig(os.path.join(stage_dir, "boxplot_du_dls"))
        plt.clf()

        combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), mol_a)
        combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
        out_file = os.path.join("debug.pdb")
        writer = PDBWriter(combined_pdb_str, out_file, args.n_frames)

            # assert 0

            # writer.write_header()
            # xs = all_coords
            # for frame_idx, x in enumerate(xs):
            #     if frame_idx > 11500 and frame_idx < 11550:
            #         # break
            #     # interval = max(1, xs.shape[0]//writer.n_frames)
            #     # if frame_idx % interval == 0:
            #         # if check_coords(x):
            #         writer.write(x*10)
            #         # else:
            #             # print("failed to write on frame", frame_idx)
            #             # break
            # writer.close()

            # assert 0
            # print(system)


    print("epoch", epoch, "stage_dGs", stage_dGs, "final dG", stage_dGs[0]+stage_dGs[1]+stage_dGs[2])
    