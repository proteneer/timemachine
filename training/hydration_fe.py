import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pickle

from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)

# import copy
import argparse
import time
import datetime
import numpy as np
# from io import StringIO
# import itertools
import os
import sys

from ff import handlers
from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize_handlers



from rdkit import Chem

from fe import dataset

# from fe import loss, bar
# from fe.pdb_writer import PDBWriter

import configparser
import grpc

from training import hydration_model, hydration_setup
from training import simulation
from training import service_pb2_grpc

from timemachine.lib import LangevinIntegrator
from training import water_box

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Absolute Hydration Free Energy Script')
    parser.add_argument('--config_file', type=str, required=True, help='Location of config file.')
    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    print("Config Settings:")
    config.write(sys.stdout)

    general_cfg = config['general']

    # set up learning rates
    learning_rates = {}
    for k, v in config['learning_rates'].items():
        vals = [float(x) for x in v.split(',')]
        if k == 'am1ccc':
            learning_rates[handlers.AM1CCCHandler] = np.array(vals)
        elif k == 'lj':
            learning_rates[handlers.LennardJonesHandler] = np.array(vals)

    intg_cfg = config['integrator']

    suppl = Chem.SDMolSupplier(general_cfg['ligand_sdf'], removeHs=False)

    data = []

    for guest_idx, mol in enumerate(suppl):
        label_dG = -4.184*float(mol.GetProp(general_cfg['dG'])) # in kcal/mol
        label_err = 4.184*float(mol.GetProp(general_cfg['dG_err'])) # errs are positive!
        data.append((mol, label_dG, label_err))

    data = data[:10]

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

    print("Host Shape", host_coords.shape)

    lambda_schedule = np.array([float(x) for x in general_cfg['lambda_schedule'].split(',')])

    # lambda_schedule = np.concatenate([
    #     np.linspace(0.0, 0.6, 6, endpoint=False),
    #     np.linspace(0.6, 1.5, 2, endpoint=False),
    #     np.linspace(1.5, 5.5, 2, endpoint=True),
    #     # np.linspace(0.0, 1.5, 4, endpoint=True)
    # ])
    # for l in lambda_schedule:
    #     print("{:.4}".format(l),end=',')

    # assert 0

    num_steps = int(general_cfg['n_steps'])

    for epoch in range(100):

        print("Starting Epoch", epoch, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        epoch_dir = os.path.join(general_cfg["out_dir"], "epoch_"+str(epoch))

        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        epoch_params = serialize_handlers(ff_handlers)
        with open(os.path.join(epoch_dir, "start_epoch_params.py"), 'w') as fh:
            fh.write(epoch_params)

        all_data = []
        # test_items = [(x, True) for x in test_dataset.data]
        test_items = [(x, False) for x in test_dataset.data]
        train_dataset.shuffle()
        train_items = [(x, False) for x in train_dataset.data]

        all_data.extend(test_items)
        all_data.extend(train_items)

        for (mol, label_dG, label_err), inference in all_data:

            if inference:
                prefix = "test"
            else:
                prefix = "train"

            start_time = time.time()

            # out_dir = os.path.join(epoch_dir, "mol_"+mol.GetProp("_Name"))\
            # if not os.path.exists(out_dir):
                # os.makedirs(out_dir)

            try:

                potentials, masses, vjp_fns = hydration_setup.combine_potentials(
                    ff_handlers,
                    mol,
                    host_system,
                    precision=np.float32
                )

                coords = hydration_setup.combine_coordinates(
                    host_coords,
                    mol,
                )

                seed = np.random.randint(0, np.iinfo(np.int32).max)

                intg = LangevinIntegrator(
                    float(intg_cfg['temperature']),
                    float(intg_cfg['dt']),
                    float(intg_cfg['friction']),
                    masses,
                    seed
                )

                sim = simulation.Simulation(
                    coords,
                    np.zeros_like(coords),
                    box,
                    potentials,
                    intg
                )

                (pred_dG, pred_err), grad_dG, du_dls = hydration_model.simulate(
                    sim,
                    num_steps,
                    lambda_schedule,
                    stubs
                )

                plt.plot(lambda_schedule, du_dls)
                plt.ylabel("du_dlambda")
                plt.xlabel("lambda")
                plt.savefig(os.path.join(epoch_dir, "ti_mol_"+mol.GetProp("_Name")))
                plt.clf()

                loss = np.abs(pred_dG - label_dG)

                # error CIs are wrong "95% CI [{:.2f}, {:.2f}, {:.2f}]".format(pred_err.lower_bound, pred_err.value, pred_err.upper_bound),
                print(prefix, "mol", mol.GetProp("_Name"), "loss {:.2f}".format(loss), "pred_dG {:.2f}".format(pred_dG), "label_dG {:.2f}".format(label_dG), "label err {:.2f}".format(label_err), "time {:.2f}".format(time.time() - start_time), "smiles:", Chem.MolToSmiles(mol))

                # update ff parameters
                if not inference:

                    loss_grad = np.sign(pred_dG - label_dG)

                    assert len(grad_dG) == len(vjp_fns)

                    for grad, handle_and_vjp_fns in zip(grad_dG, vjp_fns):
                        for handle, vjp_fn in handle_and_vjp_fns:
                            if type(handle) in learning_rates:
                                bounds = learning_rates[type(handle)]
                                # deps_ij/(eps_i*eps_j) is unstable so we skip eps
                                if isinstance(handle, handlers.LennardJonesHandler):
                                    grad[:, 1] = 0

                                dL_dp = loss_grad*vjp_fn(grad)[0]
                                dL_dp = np.clip(dL_dp, -bounds, bounds)
                                handle.params -= dL_dp

                    epoch_params = serialize_handlers(ff_handlers)
                    with open(os.path.join(epoch_dir, "checkpoint_epoch_params.py"), 'w') as fh:
                        fh.write(epoch_params)

            except Exception as e:
                import traceback
                print("Exception in mol", mol.GetProp("_Name"), Chem.MolToSmiles(mol), e)
                traceback.print_exc()


        # epoch_params = serialize_handlers(ff_handlers)
        # with open(os.path.join(epoch_dir, "end_epoch_params.py"), 'w') as fh:
        #     fh.write(epoch_params)