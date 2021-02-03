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

from ff.handlers.serialize import serialize_handlers
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

from fe import dataset

from fe import loss, bar
from fe.pdb_writer import PDBWriter

import configparser
import grpc

from training import trainer
from training import service_pb2_grpc

from fe.utils import convert_uIC50_to_kJ_per_mole

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Absolute Binding Free Energy Script')
    parser.add_argument('--config_file', type=str, required=True, help='Location of config file.')

    args = parser.parse_args()

    print("Launch Time:", datetime.datetime.now())

    config = configparser.ConfigParser()
    config.read(args.config_file)
    print("Config Settings:")
    config.write(sys.stdout)

    general_cfg = config['general']

    if not os.path.exists(general_cfg['out_dir']):
        os.makedirs(general_cfg['out_dir'])

    suppl = Chem.SDMolSupplier(general_cfg['ligand_sdf'], removeHs=False)

    all_guest_mols = []

    data = []

    for guest_idx, mol in enumerate(suppl):
        mol_dG = -1*convert_uIC50_to_kJ_per_mole(float(mol.GetProp(general_cfg['bind_prop'])))
        data.append((mol, mol_dG))

    full_dataset = dataset.Dataset(data)
    train_frac = float(general_cfg['train_frac'])
    train_dataset, test_dataset = full_dataset.split(train_frac)

    # process the host first
    host_pdbfile = general_cfg['protein_pdb']

    stage_dGs = []

    ff_raw = open(general_cfg['forcefield'], "r").read()
    ff_handlers = deserialize(ff_raw)

    worker_address_list = []
    for address in config['workers']['hosts'].split(','):
        worker_address_list.append(address)

    stubs = []

    for address in worker_address_list:
        print("connecting to", address)
        channel = grpc.insecure_channel(address,
            options = [
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024)
            ]
        )

        stub = service_pb2_grpc.WorkerStub(channel)
        stubs.append(stub)

    intg_cfg = config['integrator']
    lr_config = config['learning_rates']
    restr_config = config['restraints']

    lambda_schedule = {}

    for stage_str, v in config['lambda_schedule'].items():

        stage = int(stage_str)
        stage_schedule = np.array([float(x) for x in v.split(',')])

        assert stage not in lambda_schedule

        if stage == 0 or stage == 1:
            # stage 0 must be monotonically decreasing
            assert np.all(np.diff(stage_schedule) > 0)
        else:
            raise Exception("unknown stage")
            # stage 1 and 2 must be monotonically increasing
            # assert np.all(np.diff(stage_schedule) > 0)
        lambda_schedule[stage] = stage_schedule

    learning_rates = {}
    for k, v in config['learning_rates'].items():
        learning_rates[k] = np.array([float(x) for x in v.split(',')])

    engine = trainer.Trainer(
        host_pdbfile, 
        stubs,
        worker_address_list,
        ff_handlers,
        lambda_schedule,
        int(general_cfg['du_dl_cutoff']),
        float(restr_config['search_radius']),
        float(restr_config['force_constant']),
        int(general_cfg['n_frames']),
        int(intg_cfg['steps']),
        float(intg_cfg['dt']),
        float(intg_cfg['temperature']),
        float(intg_cfg['friction']),
        learning_rates,
        general_cfg['precision']
    )

    for epoch in range(100):

        print("Starting Epoch", epoch, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        epoch_dir = os.path.join(general_cfg['out_dir'], "epoch_"+str(epoch))

        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        epoch_params = serialize_handlers(ff_handlers)
        with open(os.path.join(epoch_dir, "start_epoch_params.py"), 'w') as fh:
            fh.write(epoch_params)

        for mol, experiment_dG in test_dataset.data:
            print("test mol", mol.GetProp("_Name"), "Smiles:", Chem.MolToSmiles(mol))
            mol_dir = os.path.join(epoch_dir, "test_mol_"+mol.GetProp("_Name"))
            start_time = time.time()
            dG, ci, loss = engine.run_mol(mol, inference=True, run_dir=mol_dir, experiment_dG=experiment_dG)
            print(mol.GetProp("_Name"), "test loss", loss, "pred_dG", dG, "exp_dG", experiment_dG, "time", time.time() - start_time, "ci 95% (mean, lower, upper)", ci.value, ci.lower_bound, ci.upper_bound)

        train_dataset.shuffle()

        for mol, experiment_dG in train_dataset.data:
            print("train mol", mol.GetProp("_Name"), "Smiles:", Chem.MolToSmiles(mol))
            mol_dir = os.path.join(epoch_dir, "train_mol_"+mol.GetProp("_Name"))
            start_time = time.time()
            dG, ci, loss = engine.run_mol(mol, inference=False, run_dir=mol_dir, experiment_dG=experiment_dG)
            print(mol.GetProp("_Name"), "train loss", loss, "pred_dG", dG, "exp_dG", experiment_dG, "time", time.time() - start_time, "ci 95% (mean, lower, upper)", ci.value, ci.lower_bound, ci.upper_bound)

        epoch_params = serialize_handlers(ff_handlers)
        with open(os.path.join(epoch_dir, "end_epoch_params.py"), 'w') as fh:
            fh.write(epoch_params)
