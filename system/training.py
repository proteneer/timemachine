import os
import sys
import numpy as np
import datetime
import sklearn.metrics
import jax
import scipy
import json
import glob

from scipy import stats
from rdkit import Chem

from system import serialize
from system import forcefield
from system import simulation

from openforcefield.typing.engines.smirnoff import ForceField
from timemachine.lib import custom_ops
from jax.experimental import optimizers

import multiprocessing

def run_simulation(params):

    p = multiprocessing.current_process()
    combined_params, guest_sdf_file, label, idx = params
    if 'gpu_offset' in properties:
        gpu_offset = properties['gpu_offset']
    else:
        gpu_offset = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(idx % properties['batch_size'] + gpu_offset)

    host_potentials, host_conf, (dummy_host_params, host_param_groups), host_masses = serialize.deserialize_system(properties['host_path'])
    host_params = combined_params[:len(dummy_host_params)]
    guest_sdf = open(os.path.join(properties['guest_directory'], guest_sdf_file), "r").read()
    print("processing",guest_sdf_file)
    
    mol = Chem.MolFromMol2Block(guest_sdf, sanitize=True, removeHs=False, cleanupSubstructures=True)
        
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    guest_potentials, _, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)
    smirnoff_params = combined_params[len(dummy_host_params):]

    combined_potentials, _, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
        host_potentials, guest_potentials,
        host_params, smirnoff_params,
        host_param_groups, smirnoff_param_groups,
        host_conf, guest_conf,
        host_masses, guest_masses)

    def filter_groups(param_groups, groups):
        roll = np.zeros_like(param_groups)
        for g in groups:
            roll = np.logical_or(roll, param_groups == g)
        return roll
    
    dp_idxs = properties['dp_idxs']
    
    if len(dp_idxs) == 0:
        host_dp_idxs = np.array([0])
        guest_dp_idxs = np.array([0])
        combined_dp_idxs = np.array([0])
    else:
        host_dp_idxs = np.argwhere(filter_groups(host_param_groups, dp_idxs)).reshape(-1)
        guest_dp_idxs = np.argwhere(filter_groups(smirnoff_param_groups, dp_idxs)).reshape(-1)
        combined_dp_idxs = np.argwhere(filter_groups(combined_param_groups, dp_idxs)).reshape(-1)

    RH = simulation.run_simulation(
        host_potentials,
        host_params,
        host_param_groups,
        host_conf,
        host_masses,
        host_dp_idxs,
        1000
    )
    
    H_E, H_derivs, _ = simulation.average_E_and_derivatives(RH)

    RG = simulation.run_simulation(
        guest_potentials,
        smirnoff_params,
        smirnoff_param_groups,
        guest_conf,
        guest_masses,
        guest_dp_idxs,
        1000
    )

    G_E, G_derivs, _ = simulation.average_E_and_derivatives(RG)

    RHG = simulation.run_simulation(
        combined_potentials,
        combined_params,
        combined_param_groups,
        combined_conf,
        combined_masses,
        combined_dp_idxs,
        1000
    )
    
    HG_E, HG_derivs, _ = simulation.average_E_and_derivatives(RHG)
        
    pred_enthalpy = HG_E - (G_E + H_E)
    delta = pred_enthalpy - label
    
    combined_derivs = np.zeros_like(combined_params)
    combined_derivs[combined_dp_idxs] += HG_derivs
    combined_derivs[host_dp_idxs] -= H_derivs
    combined_derivs[guest_dp_idxs + len(host_params)] -= G_derivs

    if properties['loss_fn'] == 'L2':
        combined_derivs = 2*delta*combined_derivs
    elif properties['loss_fn'] == 'L1':
        combined_derivs = (delta/np.abs(delta))*combined_derivs
    elif properties['loss_fn'] == 'Huber':
        # Modified Huber Loss
        # if abs(delta) < cutoff, loss = (delta^2)/ (2 * cutoff)
        # if abs(delta) > cutoff, loss = abs(delta)
        huber_cutoff = np.where('huber_cutoff' in properties, properties['huber_cutoff'], 1)
        combined_derivs = np.where(abs(delta) < huber_cutoff, delta / huber_cutoff, delta/np.abs(delta)) * combined_derivs
    elif properties['loss_fn'] == 'log-cosh':
        combined_derivs = (1 / np.cosh(delta)) * np.sinh(delta) * combined_derivs

    return combined_derivs, pred_enthalpy, label

def initialize_parameters(host_path):

    _, _, (host_params, _), _ = serialize.deserialize_system(host_path)

    # setting general smirnoff parameters for guest
    structure_path = os.path.join(properties['guest_directory'], properties['guest_template'])
    if '.mol2' in properties['guest_template']:
        structure_file = open(structure_path,'r').read()
        mol = Chem.MolFromMol2Block(structure_file, sanitize=True, removeHs=False, cleanupSubstructures=True)
    else:
        raise Exception('only mol2 files currently supported for ligand training')

    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")
    
    _, smirnoff_params, _, _, _ = forcefield.parameterize(mol, smirnoff)
    
    epoch_combined_params = np.concatenate([host_params, smirnoff_params])
    
    return epoch_combined_params

def train(num_epochs, opt_init, opt_update, get_params, init_params):
    
    training_data = properties['training_data']
    
    batch_size = properties['batch_size']
    pool = multiprocessing.Pool(batch_size)
    num_data_points = len(training_data)
    num_batches = int(np.ceil(num_data_points/batch_size))
    
    opt_state = opt_init(init_params)
    count = 0
    
    for epoch in range(num_epochs):

        print('--- epoch:', epoch, "started at", datetime.datetime.now(), '----')

        np.random.shuffle(training_data)
        
        epoch_predictions = []
        epoch_labels = []
        epoch_filenames = []

        for fn in training_data:
            epoch_filenames.append(fn[0])

        for b_idx in range(num_batches):
            start_idx = b_idx*batch_size
            end_idx = min((b_idx+1)*batch_size, num_data_points)
            batch_data = training_data[start_idx:end_idx]

            args = []

            for b_idx, b in enumerate(batch_data):
                args.append([get_params(opt_state), b[0], b[1], b_idx])

            results = pool.map(run_simulation, args)

            batch_dp = np.zeros_like(init_params)

            for grads, preds, labels in results:
                batch_dp += grads
                epoch_predictions.append(preds)
                epoch_labels.append(labels)    

            count += 1
            opt_state = opt_update(count, batch_dp, opt_state)

        epoch_predictions = np.array(epoch_predictions)
        epoch_labels = np.array(epoch_labels)

        np.savez("run_{}.npz".format(epoch), preds=epoch_predictions, labels=epoch_labels, filenames=epoch_filenames, params=get_params(opt_state))
    
        mae = np.mean(np.abs(epoch_predictions-epoch_labels))
        mean = np.mean(epoch_predictions-epoch_labels)
        if len(epoch_predictions) > 1:
            pearson_r = stats.pearsonr(epoch_predictions, epoch_labels)
            r2_score = sklearn.metrics.r2_score(epoch_predictions, epoch_labels)
            print('''

Epoch: {}
-----------------
Pearson R: {}
R2 score: {}
MAE: {}
Mean: {}
-----------------

            '''.format(epoch,pearson_r[0], r2_score, mae, mean))
        else:
            print('''
            
Epoch: {}
-----------------
MAE: {}
Mean: {}
-----------------
            
            '''.format(epoch,mae, mean))
        
    return preds, labels, get_params(opt_state)

def initialize_optimizer(optimizer, lr):
    
    if optimizer == 'Adam':
        opt_init, opt_update, get_params = optimizers.adam(lr)
    elif optimizer == 'SGD':
        opt_init, opt_update, get_params = optimizers.sgd(lr)
    elif optimizer == 'RMSProp':
        opt_init, opt_update, get_params = optimizers.rmsprop(lr)
    elif optimizer == 'Adagrad':
        opt_init, opt_update, get_params = optimizers.adagrad(lr)
    elif optimizer == 'SM3':
        opt_init, opt_update, get_params = optimizers.sm3(lr)
        
    return opt_init, opt_update, get_params

if __name__ == "__main__":
    
    config_file = glob.glob('*.json')
    if len(config_file) == 0:
        raise Exception('config file not found')
    elif len(config_file) > 1:
        raise Exception('multiple config files found')
    with open(config_file[0], 'r') as file:
        config = json.load(file)
   
    globals()['properties'] = config

    init_params = initialize_parameters(properties['host_path'])
    opt_init, opt_update, get_params = initialize_optimizer(properties['optimizer'], properties['learning_rate'])
    
    preds, labels, final_params = train(properties['num_epochs'], opt_init, opt_update, get_params, init_params)
    
    np.savez('final_params.npz', params=final_params)
    
