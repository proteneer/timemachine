import matplotlib
matplotlib.use('Agg')
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

    all_guest_mols = []
    data = []

    for guest_idx, mol in enumerate(suppl):
        label_dG = -4.184*float(mol.GetProp(general_cfg['dG']))
        label_err = 4.184*float(mol.GetProp(general_cfg['dG_err'])) # errs are positive!
        data.append((mol, label_dG, label_err))

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

    lambda_schedule = np.concatenate([
        # np.linspace(0.0, 0.6, 40, endpoint=False),
        # np.linspace(0.6, 1.5, 20, endpoint=False),
        # np.linspace(1.5, 5.5, 20, endpoint=True),
        np.linspace(0.0, 1.5, 4, endpoint=True)
    ])

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
        test_items = [(x, True) for x in test_dataset.data]
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

            out_dir = os.path.join(epoch_dir, "mol_"+mol.GetProp("_Name"))

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
                intg,
            )


            (pred_dG, pred_err), grad_dG = hydration_model.simulate(
                sim,
                num_steps,
                lambda_schedule,
                stubs
            )

            loss = np.abs(pred_dG - label_dG)

            print(prefix, "mol", mol.GetProp("_Name"), "loss {:.2f}".format(loss), "pred_dG {:.2f}".format(pred_dG), "95% CI [{:.2f}, {:.2f}, {:.2f}]".format(pred_err.lower_bound, pred_err.value, pred_err.upper_bound), "label_dG {:.2f}".format(label_dG), "label err {:.2f}".format(label_err), "time {:.2f}".format(time.time() - start_time), "smiles:", Chem.MolToSmiles(mol))

            # update ff parameters
            if not inference:

                loss_grad = np.sign(pred_dG - label_dG)

                assert len(grad_dG) == len(vjp_fns)

                for grad, handle_and_vjp_fn in zip(grad_dG, vjp_fns):
                    if handle_and_vjp_fn:
                        handle, vjp_fn = handle_and_vjp_fn
                        if type(handle) in learning_rates:

                            bounds = learning_rates[type(handle)]

                            # deps_ij/(eps_i*eps_j) is unstable so we skip eps
                            if isinstance(handle, handlers.LennardJonesHandler):
                                grad[:, 1] = 0

                            dL_dp = loss_grad*vjp_fn(grad)[0]
                            dL_dp = np.clip(dL_dp, -bounds, bounds)
                            handle.params -= dL_dp
                        # else:
                            # print("skipping", type(handle))


        epoch_params = serialize_handlers(ff_handlers)
        with open(os.path.join(epoch_dir, "end_epoch_params.py"), 'w') as fh:
            fh.write(epoch_params)