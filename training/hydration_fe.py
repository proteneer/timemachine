import matplotlib
matplotlib.use('Agg')
import pickle

# import copy
import argparse
import time
import datetime
import numpy as np
# from io import StringIO
# import itertools
import os
import sys

from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize_handlers

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)
# import jax
# import jax.numpy as jnp

# import rdkit
from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import rdFMCS

from fe import dataset

# from fe import loss, bar
# from fe.pdb_writer import PDBWriter

import configparser
import grpc

from training import hydration_model, hydration_setup
# # from training import trainer
from training import service_pb2, service_pb2_grpc

# 


from training import water_box

# from matplotlib import pyplot as plt

# import pickle

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18


def concat_with_vjps(p_a, p_b, vjp_a, vjp_b):
    """
    Returns the combined parameters p_c, and a vjp_fn that can take in adjoint with shape
    of p_c and returns adjoints of primitives of p_a and p_b.

    i.e. 
       vjp_a            
    A' -----> A 
                \ vjp_c
                 +-----> C
       vjp_b    /
    B' -----> B

    """
    p_c, vjp_c = jax.vjp(jnp.concatenate, [p_a, p_b])
    adjoints = np.random.randn(*p_c.shape)

    def adjoint_fn(p_c):
        ad_a, ad_b = vjp_c(p_c)[0]
        if vjp_a is not None:
            ad_a = vjp_a(ad_a)
        else:
            ad_a = None

        if vjp_b is not None:
            ad_b = vjp_b(ad_b)
        else:
            ad_b = None

        return ad_b[0]

    return p_c, adjoint_fn

# used during visualization
def recenter(conf, box):

    new_coords = []

    periodicBoxSize = box

    for atom in conf:
        diff = np.array([0., 0., 0.])
        diff += periodicBoxSize[2]*np.floor(atom[2]/periodicBoxSize[2][2]);
        diff += periodicBoxSize[1]*np.floor((atom[1]-diff[1])/periodicBoxSize[1][1]);
        diff += periodicBoxSize[0]*np.floor((atom[0]-diff[0])/periodicBoxSize[0][0]);
        new_coords.append(atom - diff)

    return np.array(new_coords)


# from fe import math_utils, system


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Absolute Hydration Free Energy Script')
    parser.add_argument('--config_file', type=str, required=True, help='Location of config file.')
    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    print("Config Settings:")
    config.write(sys.stdout)

    general_cfg = config['general']



    suppl = Chem.SDMolSupplier(general_cfg['ligand_sdf'], removeHs=False)

    all_guest_mols = []
    data = []

    for guest_idx, mol in enumerate(suppl):
        true_dG = -1*float(mol.GetProp(general_cfg['dG']))
        true_dG_err = -1*float(mol.GetProp(general_cfg['dG_err']))
        data.append((mol, true_dG, true_dG_err))

    full_dataset = dataset.Dataset(data)
    train_frac = float(general_cfg['train_frac'])
    train_dataset, test_dataset = full_dataset.split(train_frac)

    forcefield = general_cfg['forcefield']

    stubs = []

    worker_address_list = []
    for address in config['workers']['hosts'].split(','):
        worker_address_list.append(address)

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

    ff_raw = open(forcefield, "r").read()

    ff_handlers = deserialize_handlers(ff_raw)

    box_width = 3.0
    host_system, host_coords, box, _ = water_box.prep_system(box_width)

    for epoch in range(100):

        print("Starting Epoch", epoch, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        epoch_dir = os.path.join(general_cfg["out_dir"], "epoch_"+str(epoch))

        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        epoch_params = serialize_handlers(ff_handlers)
        with open(os.path.join(epoch_dir, "start_epoch_params.py"), 'w') as fh:
            fh.write(epoch_params)

        # simulate(
        #     guest_mol,
        #     ff_handlers,
        #     stubs,
        #     general_cfg.n_frames,
        #     epoch_dir
        # )

        for mol, experiment_dG, experiment_error in test_dataset.data:
            print("test mol", mol.GetProp("_Name"), "Smiles:", Chem.MolToSmiles(mol))
            out_dir = os.path.join(epoch_dir, "test_mol_"+mol.GetProp("_Name"))

            hydration_setup.combine_potentials(
                ff_handlers,
                mol,
                host_system,
                precision=np.float32
            )


            assert 0

            start_time = time.time()
            dG, ci, loss = engine.run_mol(mol, inference=True, run_dir=out_dir, experiment_dG=experiment_dG)
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