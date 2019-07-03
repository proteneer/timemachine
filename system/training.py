import os
import sys
import numpy as np
import datetime
import sklearn.metrics
import jax
import scipy
import json
import glob
import csv
import itertools
import functools

from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem

from system import serialize
from system import forcefield
from system import simulation

from openforcefield.typing.engines.smirnoff import ForceField
from timemachine.lib import custom_ops
from jax.experimental import optimizers, stax

import multiprocessing 

def rescale_and_center(conf, scale_factor=1):
    mol_com = np.sum(conf, axis=0)/conf.shape[0]
    true_com = np.array([1.97698696, 1.90113478, 2.26042174]) # a-cd
#     true_com = np.array([5.4108882, 4.75821426, 9.33421262]) # london
    centered = conf - mol_com  # centered to origin
    return true_com + centered/scale_factor
    
def run_simulation(params):

    p = multiprocessing.current_process()
    combined_params, guest_sdf_file, label, idx = params
    label = float(label)

    if 'gpu_offset' in properties:
        gpu_offset = properties['gpu_offset']
    else:
        gpu_offset = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(idx % properties['batch_size'] + gpu_offset)

    host_potentials, host_conf, (dummy_host_params, host_param_groups), host_masses = serialize.deserialize_system(properties['host_path'])
    host_params = combined_params[:len(dummy_host_params)]
    guest_sdf = open(os.path.join(properties['guest_directory'], guest_sdf_file), "r").read()
    print("processing",guest_sdf_file)
        
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")
    
    mol = Chem.MolFromMol2Block(guest_sdf, sanitize=True, removeHs=False, cleanupSubstructures=True)
#     AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=1234, clearConfs=True)
    AllChem.EmbedMultipleConfs(mol, numConfs=5, clearConfs=True)

    guest_potentials, _, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)
    smirnoff_params = combined_params[len(dummy_host_params):]

    RH = []
    RG = []
    RHG = []
    
    for conf_idx in range(mol.GetNumConformers()):
        c = mol.GetConformer(conf_idx)
        conf = np.array(c.GetPositions(),dtype=np.float64)
        guest_conf = conf/10
        rot_matrix = stats.special_ortho_group.rvs(3).astype(dtype=np.float32)
        guest_conf = np.matmul(guest_conf, rot_matrix)
        guest_conf = rescale_and_center(guest_conf)
        combined_potentials, _, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
            host_potentials, guest_potentials,
            host_params, smirnoff_params,
            host_param_groups, smirnoff_param_groups,
            host_conf, guest_conf,
            host_masses, guest_masses)

        num_atoms = len(combined_masses)

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

        RH_i = None
        if properties['fit_method'] == 'absolute':
            RH_i = simulation.run_simulation(
                host_potentials,
                host_params,
                host_param_groups,
                host_conf,
                host_masses,
                host_dp_idxs,
                1000
            )

        RG_i = simulation.run_simulation(
            guest_potentials,
            smirnoff_params,
            smirnoff_param_groups,
            guest_conf,
            guest_masses,
            guest_dp_idxs,
            1000
        )

        RHG_i = simulation.run_simulation(
            combined_potentials,
            combined_params,
            combined_param_groups,
            combined_conf,
            combined_masses,
            combined_dp_idxs,
            1000
        )    
        if RH_i is not None:
            RH.append(RH_i[0])
            
        RG.append(RG_i[0])
        RHG.append(RHG_i[0])
    
    return RH, RG, RHG, label, host_dp_idxs, guest_dp_idxs, combined_dp_idxs, num_atoms

def boltzmann_derivatives(reservoir):
    n_reservoir = len(reservoir)
    num_atoms = len(reservoir[0][-1])

    E= []
    dE_dx_temp = np.zeros((n_reservoir,n_reservoir,num_atoms,3))
    dE_dx = []
    dx_dp = []
    dE_dp = []
    for E_i, dE_dx_i, dx_dp_i, dE_dp_i, _ in reservoir:
        E.append(E_i)
        dE_dx.append(dE_dx_i)
        dx_dp.append(dx_dp_i)
        dE_dp.append(dE_dp_i)
    
    E = np.array(E,dtype=np.float32)
    dE_dx = np.array(dE_dx,dtype=np.float64)
    dx_dp = np.array(dx_dp)
    dx_dp = np.transpose(dx_dp,(0,2,3,1))
    dE_dp = np.array(dE_dp) 
    
    for i in range(n_reservoir):
        dE_dx_temp[i][i] = np.array(dE_dx[i])
        
    dE_dx = dE_dx_temp
    
    ds_de_fn = jax.jacfwd(stax.softmax, argnums=(0,))
    ds_de = ds_de_fn(-E)

    # tensor contract [C,C,N,3] with [C,N,3,P] and dE_dp
    tot_dE_dp = np.einsum('ijkl,jklm->im', dE_dx, dx_dp) + dE_dp
    s_e = stax.softmax(-E)

    total_derivs = np.matmul(-ds_de[0], tot_dE_dp)*np.expand_dims(E, 1) + np.expand_dims(s_e, axis=-1) * tot_dE_dp

    return np.sum(stax.softmax(-E)*E), np.sum(total_derivs, axis=0)

def compute_derivatives(params1,
                        params2,
                        host_params,
                       combined_params):
    
    RH1, RG1, RHG1, label_1, host_dp_idxs, guest_dp_idxs_1, combined_dp_idxs_1, num_atoms = params1
    G_E, G_derivs = boltzmann_derivatives(RG1)
    HG_E, HG_derivs = boltzmann_derivatives(RHG1)
#     G_E_mean, _, _ = simulation.average_E_and_derivatives(RG1)
#     HG_E_mean, _, _ = simulation.average_E_and_derivatives(RHG1)
    
#     print('boltzmann',G_E,HG_E)
#     print('normal mean', G_E_mean,HG_E_mean)
        
    if properties['fit_method'] == 'absolute':
#         H_E, H_derivs, _ = simulation.average_E_and_derivatives(RH1)
        H_E, H_derivs = boltzmann_derivatives(RH1)
        pred_enthalpy = HG_E - (G_E + H_E)
        label = label_1
        delta = pred_enthalpy - label

        combined_derivs = np.zeros_like(combined_params)
        combined_derivs[combined_dp_idxs_1] += HG_derivs
        combined_derivs[host_dp_idxs] -= H_derivs
        combined_derivs[guest_dp_idxs_1 + len(host_params)] -= G_derivs
            
    elif properties['fit_method'] == 'relative':
        RH2, RG2, RHG2, label_2, host_dp_idxs, guest_dp_idxs_2, combined_dp_idxs_2, _ = params2
        G_E_2, G_derivs_2 = boltzmann_derivatives(RG2)
        HG_E_2, HG_derivs_2 = boltzmann_derivatives(RHG2)
#         G_E_2, G_derivs_2, _ = simulation.average_E_and_derivatives(RG2)
#         HG_E_2, HG_derivs_2, _ = simulation.average_E_and_derivatives(RHG2)
        
        pred_enthalpy = HG_E - HG_E_2 - G_E + G_E_2
        label = label_1 - label_2
        delta = pred_enthalpy - label
        
        combined_derivs = np.zeros_like(combined_params)
        combined_derivs[combined_dp_idxs_1] += HG_derivs
        combined_derivs[combined_dp_idxs_2] -= HG_derivs_2
        combined_derivs[guest_dp_idxs_1 + len(host_params)] -= G_derivs
        combined_derivs[guest_dp_idxs_2 + len(host_params)] += G_derivs_2
        
    if properties['loss_fn'] == 'L2':
        '''
        loss = (delta) ^ 2
        '''
        combined_derivs = 2*delta*combined_derivs
    elif properties['loss_fn'] == 'L1':
        '''
        loss = |delta|
        '''
        combined_derivs = (delta/np.abs(delta))*combined_derivs
    elif properties['loss_fn'] == 'Huber':
        '''
        if |delta| < cutoff, loss = (delta^2) / (2 * cutoff)
        if |delta| > cutoff, loss = |delta|
        '''
        huber_cutoff = np.where('huber_cutoff' in properties, properties['huber_cutoff'], 1)
        combined_derivs = np.where(abs(delta) < huber_cutoff, delta / huber_cutoff, delta/np.abs(delta)) * combined_derivs
    elif properties['loss_fn'] == 'log-cosh':
        '''
        loss = log(cosh(delta))
        '''
        combined_derivs = (1 / np.cosh(delta)) * np.sinh(delta) * combined_derivs
        
    if np.amax(abs(combined_derivs)) * properties['learning_rate'] > 1e-1:
        print("bad gradients")
        combined_derivs *= 0

    return combined_derivs, pred_enthalpy, label
    
def initialize_parameters(host_path):

    _, _, (host_params, _), _ = serialize.deserialize_system(host_path)

    # setting general smirnoff parameters for guest
    structure_path = os.path.join(properties['guest_directory'], properties['guest_template'])
    if '.mol2' in properties['guest_template']:
        structure_file = open(structure_path,'r').read()
        ref_mol = Chem.MolFromMol2Block(structure_file, sanitize=True, removeHs=False, cleanupSubstructures=True)
    else:
        raise Exception('only mol2 files currently supported for ligand training')

    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")
    
    _, smirnoff_params, _, _, _ = forcefield.parameterize(ref_mol, smirnoff)
    
    epoch_combined_params = np.concatenate([host_params, smirnoff_params])
    
    return epoch_combined_params

def train(num_epochs, 
          opt_init, 
          opt_update, 
          get_params, 
          init_params):
    
    data_file = open(properties['training_data'],'r')
    data_reader = csv.reader(data_file, delimiter=',')
    training_data = list(data_reader)
           
    batch_size = properties['batch_size']
    pool = multiprocessing.Pool(batch_size)
    if properties['fit_method'] == 'relative':
        batch_size -= 1
    num_data_points = len(training_data)
    num_batches = int(np.ceil(num_data_points/batch_size))

    opt_state = opt_init(init_params)
    count = 0
    
    _, _, (dummy_host_params, host_param_groups), _ = serialize.deserialize_system(properties['host_path'])
    
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

            if properties['fit_method'] == 'relative':
                args.append([get_params(opt_state), properties['relative_reference'][0], properties['relative_reference'][1], 0])
            
            for b_idx, b in enumerate(batch_data):
                if properties['fit_method'] == 'relative':
                    args.append([get_params(opt_state), b[0], b[1], b_idx + 1])
                else:
                    args.append([get_params(opt_state), b[0], b[1], b_idx])
            
            results = pool.map(run_simulation, args)
            
            final_results = []
            if properties['fit_method'] == 'absolute':
                for params in results:
                    final_results.append(compute_derivatives(params, None, dummy_host_params, get_params(opt_state)))
            if properties['fit_method'] == 'relative':
                params1 = results[0]
                for i in range(1,len(results)):
                    final_results.append(compute_derivatives(params1, results[i], dummy_host_params, get_params(opt_state)))
                    
            batch_dp = np.zeros_like(init_params)

            for grads, preds, labels in final_results:
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

def initialize_optimizer(optimizer, 
                         lr):
    
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
   
    properties = config

    init_params = initialize_parameters(properties['host_path'])
    np.savez('init_params.npz', params=init_params)
    
    opt_init, opt_update, get_params = initialize_optimizer(properties['optimizer'], properties['learning_rate'])
    
#     structure_path = os.path.join(properties['guest_directory'], properties['guest_template'])
#     structure_file = open(structure_path,'r').read()
#     mol = Chem.MolFromMol2Block(structure_file, sanitize=True, removeHs=False, cleanupSubstructures=True)
    
#     AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=1234, clearConfs=True)
    
    preds, labels, final_params = train(properties['num_epochs'], opt_init, opt_update, get_params, init_params)
    
    np.savez('final_params.npz', params=final_params)
