import matplotlib
matplotlib.use('Agg')
# import pickle
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

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from simtk.openmm.app import PDBFile
from fe import dataset

from fe import loss, bar
from fe.pdb_writer import PDBWriter


import grpc

from training import trainer
from training import service_pb2_grpc

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18

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

    data = []

    for guest_idx, mol in enumerate(suppl):
        mol_dG = -1*convert_uIC50_to_kJ_per_mole(float(mol.GetProp("IC50[uM](SPA)")))
        data.append((mol, mol_dG))

    full_dataset = dataset.Dataset(data)
    train_frac = 0.6
    train_dataset, test_dataset = full_dataset.split(0.6)

    # process the host first
    host_pdb_file = args.protein_pdb
    host_pdb = PDBFile(host_pdb_file)

    core_smarts = '[#6]1:[#6]:[#6]:[#6](:[#6](:[#6]:1-[#8]-[#6](:[#6]-[#1]):[#6])-[#1])-[#1]'

    stage_dGs = []

    ff_raw = open(args.forcefield, "r").read()
    ff_handlers = deserialize(ff_raw)

    ports = [
        50000,
        50001,
        50002,
        50003,
        50004,
        50005,
        50006,
        50007,
        50008,
        50009
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

    # lambda_schedule = [
    #     np.array([0.5, 0.7]),
    #     np.array([0.4, 1.0]),
    #     np.array([0.2, 2.0])
    # ]

    # lambda_schedule = [
    #     np.linspace(7.0, 0.0, 32),
    #     np.concatenate([
    #         np.linspace(0.0, 0.5, 24, endpoint=False),
    #         np.linspace(0.5, 1.2, 8)
    #     ]), 
    #     np.linspace(0.0, 7.0, 32)
    # ]

    lambda_schedule = [
        np.linspace(7.0, 0.0, 3),
        np.linspace(0.0, 1.2, 4),
        np.linspace(0.0, 7.0, 3)
    ]

    engine = trainer.Trainer(
        host_pdb, 
        stubs,
        ff_handlers,
        lambda_schedule,
        core_smarts,
        args.restr_force,
        args.restr_alpha,
        args.restr_count,
        args.steps,
        args.precision)

    for epoch in range(100):

        print("Starting Epoch", epoch)

        train_dataset.shuffle()
        epoch_dir = os.path.join(args.out_dir, "epoch_"+str(epoch))

        for mol, experiment_dG in test_dataset.data:
            print("test mol", mol.GetProp("_Name"), "Smiles:", Chem.MolToSmiles(mol))
            mol_dir = os.path.join(epoch_dir, "test_mol_"+mol.GetProp("_Name"))
            loss, dG = engine.run_mol(mol, inference=True, run_dir=mol_dir, experiment_dG=experiment_dG)
        
            print("loss", loss, "pred_dG", dG, "exp_dG", experiment_dG)

        for mol, experiment_dG in train_dataset.data:
            print("train mol", mol.GetProp("_Name"), "Smiles:", Chem.MolToSmiles(mol))
            mol_dir = os.path.join(epoch_dir, "train_mol_"+mol.GetProp("_Name"))
            loss, dG = engine.run_mol(mol, inference=False, run_dir=mol_dir, experiment_dG=experiment_dG)

            print("loss", loss, "pred_dG", dG, "exp_dG", experiment_dG)

        continue
