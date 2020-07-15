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

from simtk.openmm.app import PDBFile
from fe import dataset

from fe import loss, bar
from fe.pdb_writer import PDBWriter

import configparser
import grpc

from training import trainer
from training import service_pb2_grpc

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18

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

    assert os.path.isdir(general_cfg['out_dir'])

    suppl = Chem.SDMolSupplier(general_cfg['ligand_sdf'], removeHs=False)

    all_guest_mols = []

    data = []

    for guest_idx, mol in enumerate(suppl):
        mol_dG = -1*convert_uIC50_to_kJ_per_mole(float(mol.GetProp(general_cfg['bind_prop'])))
        data.append((mol, mol_dG))

    full_dataset = dataset.Dataset(data)
    train_frac = 0.6
    train_dataset, test_dataset = full_dataset.split(0.6)

    # process the host first
    host_pdb_file = general_cfg['protein_pdb']
    host_pdb = PDBFile(host_pdb_file)

    # (tbd): set to MCS if this is None
    stage_dGs = []

    ff_raw = open(general_cfg['forcefield'], "r").read()
    ff_handlers = deserialize(ff_raw)


    worker_address_list = []
    for address in config['workers']['hosts'].split(','):
        worker_address_list.append(address)

    stubs = []

    for address in worker_address_list:
        channel = grpc.insecure_channel(address,
            options = [
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024)
            ]
        )

        stub = service_pb2_grpc.WorkerStub(channel)
        stubs.append(stub)

    lambda_schedule = []
    for stage_idx, (_, v) in enumerate(config['lambda_schedule'].items()):
        stage_schedule = np.array([float(x) for x in v.split(',')])
        if stage_idx == 0:
            # stage 0 must be monotonically decreasing
            assert np.all(np.diff(stage_schedule) < 0)
        else:
            # stage 1 and 2 must be monotonically increasing
            assert np.all(np.diff(stage_schedule) > 0)
        lambda_schedule.append(stage_schedule)

    restr_cfg = config['restraints']
    intg_cfg = config['integrator']

    engine = trainer.Trainer(
        host_pdb, 
        stubs,
        ff_handlers,
        lambda_schedule,
        int(general_cfg['du_dl_cutoff']),
        restr_cfg['core_smarts'],
        float(restr_cfg['force']),
        float(restr_cfg['alpha']),
        int(restr_cfg['count']),
        int(intg_cfg['steps']),
        float(intg_cfg['dt']),
        float(intg_cfg['temperature']),
        float(intg_cfg['friction']),
        general_cfg['precision'])

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
            dG, loss = engine.run_mol(mol, inference=True, run_dir=mol_dir, experiment_dG=experiment_dG)
            print("test loss", loss, "pred_dG", dG, "exp_dG", experiment_dG, "time", time.time() - start_time)

        train_dataset.shuffle()

        for mol, experiment_dG in train_dataset.data:
            print("train mol", mol.GetProp("_Name"), "Smiles:", Chem.MolToSmiles(mol))
            mol_dir = os.path.join(epoch_dir, "train_mol_"+mol.GetProp("_Name"))
            start_time = time.time()
            dG, loss = engine.run_mol(mol, inference=False, run_dir=mol_dir, experiment_dG=experiment_dG)
            print("train loss", loss, "pred_dG", dG, "exp_dG", experiment_dG, "time", time.time() - start_time)

        epoch_params = serialize_handlers(ff_handlers)
        with open(os.path.join(epoch_dir, "end_epoch_params.py"), 'w') as fh:
            fh.write(epoch_params)
