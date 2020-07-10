import matplotlib
matplotlib.use('Agg')
import pickle
import copy
import argparse
import time
import datetime
import numpy as np
from io import StringIO
import itertools
import os
import sys

from ff.handlers.deserialize import deserialize

from multiprocessing import Process, Pipe
from matplotlib import pyplot as plt
from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from simtk.openmm.app import PDBFile
from fe import runner
from fe import math_utils, setup_system
from fe import loss, bar
from fe.pdb_writer import PDBWriter
from ff.handlers import bonded, nonbonded


import grpc

from training import service_pb2
from training import service_pb2_grpc

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18


# def loss_fn(all_du_dls, true_ddG, lambda_schedule):
#     """
#     Loss function. Currently set to L1.

#     Parameters:
#     -----------
#     all_du_dls: shape [L, F, T] np.array
#         Where L is the number of lambda windows, F is the number of forces, and T is the total number of equilibrated steps

#     true_ddG: scalar
#         True ddG of the edge.

#     Returns:
#     --------
#     scalar
#         Loss

#     """
#     assert all_du_dls.ndim == 3

#     total_du_dls = jnp.sum(all_du_dls, axis=1) # shape [L, T]
#     mean_du_dls = jnp.mean(total_du_dls, axis=1) # shape [L]
#     pred_ddG = math_utils.trapz(mean_du_dls, lambda_schedule)
#     return jnp.abs(pred_ddG - true_ddG)

def loss_fn(all_du_dls, lambda_schedules, expected_dG, du_dl_cutoff):
    """
    Parameters
    ----------
    all_du_dls: list of full_stage_du_dls (usually 3 stages)
    """

    stage_dGs = []
    for stage_du_dls, ti_lambdas in zip(all_du_dls, lambda_schedules):
        du_dls = []
        for lamb_full_du_dls in stage_du_dls:
            du_dls.append(jnp.mean(jnp.sum(lamb_full_du_dls[:, du_dl_cutoff:], axis=0)))
        du_dls = jnp.concatenate([du_dls])
        dG = math_utils.trapz(du_dls, ti_lambdas)
        stage_dGs.append(dG)

    pred_dG = jnp.sum(stage_dGs)

    return jnp.abs(pred_dG - expected_dG)

loss_fn_grad = jax.grad(loss_fn, argnums=(0,))
# F = 6
# T = 3000

# test_input = [
#     [np.random.randn(F, T)+5, np.random.randn(F, T), np.random.randn(F, T), np.random.randn(F, T)], # stage 0
#     [np.random.randn(F, T), np.random.randn(F, T), np.random.randn(F, T)], # stage 1
#     [np.random.randn(F, T), np.random.randn(F, T)] # stage 2
# ]

# lamba_schedules = [
#     np.array([0.0, 0.3, 5.0, 6.0]),
#     np.array([2.0, 3.0, 4.0]),
#     np.array([1.0, 2.0])
# ]


#     # pred_du_dl = jnp.mean(jnp.sum(all_du_dls))

# # def loss_fn(full_du_dl, expected_du_dl):
    
# #     pred_du_dl = jnp.mean(jnp.sum(full_du_dl, axis=0))
# #     print("PREDICTED", pred_du_dl, "EXPECTED", expected_du_dl)
# #     # return (expected_du_dl - pred_du_dl)**2
# #     return jnp.abs(expected_du_dl - pred_du_dl)



# # loss_fn(test_input, lamba_schedules, 50.0, 100)
# # stage_adjoint_du_dls = loss_fn_grad(test_input, lamba_schedules, 50.0, 100)[0]

# for stage_idx, add in enumerate(stage_adjoint_du_dls):
#     print("stage", stage_idx)
#     for x_idx, x in enumerate(add):
#         print(x_idx, x)
#     # print(stage_idx, add)



# assert 0


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

    print("Launch Time:", datetime.datetime.now())
    print("Arguments:", " ".join(sys.argv))

    assert os.path.isdir(args.out_dir)

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

    core_smarts = '[#6]1:[#6]:[#6]:[#6](:[#6](:[#6]:1-[#8]-[#6](:[#6]-[#1]):[#6])-[#1])-[#1]'

    print("Using core smarts:", core_smarts)
    core_query = Chem.MolFromSmarts(core_smarts)
    core_atoms = mol_a.GetSubstructMatch(core_query)

    stage_dGs = []

    ff_raw = open(args.forcefield, "r").read()
    ff_handlers = deserialize(ff_raw)

    ports = [
        50000,
        50001,
        50002,
        50003,
        50004,
        50005
    ]

    stubs = []

    for port in ports:

        channel = grpc.insecure_channel('localhost:'+str(port),
            options = [
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024)
            ]
        )

        stub = service_pb2_grpc.WorkerStub(channel)
        stubs.append(stub)

    lambda_schedule = [
        np.array([0.5, 0.7]),
        np.array([0.4, 1.0]),
        np.array([0.2, 2.0])
    ]


    for epoch in range(100):
        print("Epoch", epoch)
        epoch_dir = os.path.join(args.out_dir, "epoch_"+str(epoch))

        # for stage in [0,1,2]:
        # stage = 1

        
        # stage 1 ti_lambdas
        stage_forward_futures = []
        stub_idx = 0

        # step 1. Prepare the jobs

        for stage in [0,1,2]:

            # if stage == 0:
            #     # we need to goto a larger lambda for the morse potential to decay to zero.
            #     ti_lambdas = np.linspace(7.0, 0.0, 32)

            #     ti_lambdas = [0.5, 0.7]
            # elif stage == 1:
            #     # lambda spans from [0, inf], is close enough to zero over [0, 1.2] cutoff
            #     ti_lambdas = np.concatenate([
            #         np.linspace(0.0, 0.5, 24, endpoint=False),
            #         np.linspace(0.5, 1.2, 8)
            #     ])

            #     ti_lambdas = [0.4, 1.0]
            # elif stage == 2:
            #     # we need to goto a larger lambda for the morse potential to decay to zero.
            #     ti_lambdas = np.linspace(0.0, 7.0, 32)

            #     ti_lambdas = [0.2, 2.0]
            # else:
            #     raise Exception("Unknown stage.")

            # print("---Starting stage", stage, '---')
            stage_dir = os.path.join(epoch_dir, "stage_"+str(stage))

            if not os.path.exists(stage_dir):
                os.makedirs(stage_dir)

            x0, combined_masses, final_gradients, final_vjp_fns = setup_system.create_system(
                mol_a,
                host_pdb,
                ff_handlers,
                stage,
                core_atoms,
                args.restr_force,
                args.restr_alpha,
                args.restr_count
            )

            ti_lambdas = lambda_schedule[stage]

            forward_futures = []

            for lamb_idx, lamb in enumerate(ti_lambdas):

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

                request = service_pb2.ForwardRequest(
                    inference=False,
                    system=pickle.dumps(system),
                    precision=args.precision
                )

                stub = stubs[stub_idx]
                stub_idx += 1

                # launch asynchronously
                response_future = stub.ForwardMode.future(request)
                forward_futures.append(response_future)

            stage_forward_futures.append(forward_futures)



        # step 2. Run forward mode on the jobs

        du_dl_cutoff = 4000

        all_du_dls = []
        for stage_idx, stage_futures in enumerate(stage_forward_futures):
            stage_du_dls = []
            for future in stage_futures:

                print("waiting on future")
                response = future.result()

                full_du_dls = pickle.loads(response.du_dls)
                full_energies = pickle.loads(response.energies)

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

                equil_du_dls = full_du_dls[:, du_dl_cutoff:]

                for f, du_dls in zip(final_gradients, equil_du_dls):
                    fname = f[0]
                    print("lambda:", "{:.3f}".format(lamb), "\t median {:8.2f}".format(np.median(du_dls)), "\t mean", "{:8.2f}".format(np.mean(du_dls)), "+-", "{:7.2f}".format(np.std(du_dls)), "\t <-", fname)

                total_equil_du_dls = np.sum(equil_du_dls, axis=0) # [1, T]
                print("lambda:", "{:.3f}".format(lamb), "\t mean", "{:8.2f}".format(np.mean(total_equil_du_dls)), "+-", "{:7.2f}".format(np.std(total_equil_du_dls)), "\t <- Total")

                stage_du_dls.append(full_du_dls)

            all_du_dls.append(stage_du_dls)

        expected_dG = 60.0
        loss = loss_fn(all_du_dls, lambda_schedule, expected_dG, du_dl_cutoff)
        all_adjoint_du_dls = loss_fn_grad(all_du_dls, lambda_schedule, expected_dG, du_dl_cutoff)[0]

        # step 3. run backward mode
        stage_backward_futures = []

        stub_idx = 0
        for stage_idx, adjoint_du_dls in enumerate(all_adjoint_du_dls):

            futures = []
            for lambda_du_dls in adjoint_du_dls:
                request = service_pb2.BackwardRequest(
                    adjoint_du_dls=pickle.dumps(np.asarray(lambda_du_dls)),
                )

                # futures.append(response_future)
                futures.append(stubs[stub_idx].BackwardMode.future(request))
                stub_idx += 1

            stage_backward_futures.append(futures)

        charge_derivatives = []
        gb_derivatives = []

        for stage_idx, stage_futures in enumerate(stage_backward_futures):
            print("stage_idx", stage_idx)
            for future in stage_futures:
                backward_response = future.result()
                dl_dps = pickle.loads(backward_response.dl_dps)

                for g, vjp_fn, dl_dp in zip(final_gradients, final_vjp_fns, dl_dps):

                    # train charges only
                    if g[0] == 'Nonbonded':
                        # 0 is for charges
                        # 1 is for lj terms
                        charge_derivatives.append(vjp_fn[0](dl_dp[0]))
                    elif g[0] == 'GBSA':
                        # 0 is for charges
                        # 1 is for gb terms
                        charge_derivatives.append(vjp_fn[0](dl_dp[0]))
                        gb_derivatives.append(vjp_fn[1](dl_dp[1]))


        charge_gradients = np.sum(charge_derivatives, axis=0) # reduce
        charge_lr = 1e-3

        print(charge_gradients)

        for h in ff_handlers:
            if isinstance(h, nonbonded.SimpleChargeHandler):
                h.params -= charge_gradients*charge_lr

        assert 0
