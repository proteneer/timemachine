import os
import sys
import numpy as np
import datetime
import sklearn.metrics

from scipy import stats
from rdkit import Chem

from system import serialize
from system import forcefield
from system import simulation

from openforcefield.typing.engines.smirnoff import ForceField
from timemachine.lib import custom_ops
from jax.experimental import optimizers

import multiprocessing

num_gpus = 2
base_dir = "/home/yutong/Code/benchmarksets/input_files/cd-set1/mol2"

def run_simulation(params):

    p = multiprocessing.current_process()

    combined_params, guest_sdf_file, label, idx = params

    os.environ['CUDA_VISIBLE_DEVICES'] = str(idx % num_gpus)

    print("processing", guest_sdf_file)
    host_potentials, host_conf, (dummy_host_params, host_param_groups), host_masses = serialize.deserialize_system('examples/host_acd.xml')
    host_params = combined_params[:len(dummy_host_params)]

    guest_sdf = open(os.path.join(base_dir, guest_sdf_file), "r").read()

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

    host_dp_idxs = np.argwhere(filter_groups(host_param_groups, [7,4,5,0,1,2,3])).reshape(-1)
    guest_dp_idxs = np.argwhere(filter_groups(smirnoff_param_groups, [7,4,5,0,1,2,3])).reshape(-1)
    combined_dp_idxs = np.argwhere(filter_groups(combined_param_groups, [7,4,5,0,1,2,3])).reshape(-1)

    RH = simulation.run_simulation(
        host_potentials,
        host_params,
        host_param_groups,
        host_conf,
        host_masses,
        host_dp_idxs,
        1000,
        None
    )

    H_E, H_derivs, _ = simulation.average_E_and_derivatives(RH) # [host_dp_idxs,]

    RG = simulation.run_simulation(
        guest_potentials,
        smirnoff_params,
        smirnoff_param_groups,
        guest_conf,
        guest_masses,
        guest_dp_idxs,
        1000,
        None
    )

    G_E, G_derivs, _ = simulation.average_E_and_derivatives(RG) # [guest_dp_idxs,]

    RHG = simulation.run_simulation(
        combined_potentials,
        combined_params,
        combined_param_groups,
        combined_conf,
        combined_masses,
        combined_dp_idxs,
        1000,
        None
    )

    HG_E, HG_derivs, _ = simulation.average_E_and_derivatives(RHG) # [combined_dp_idxs,]
    
    pred_enthalpy = HG_E - (G_E + H_E)
    delta = pred_enthalpy - label
    num_atoms = len(combined_masses)
    loss = delta**2
    
    # fancy index into the full derivative set
    combined_derivs = np.zeros_like(combined_params)
    # remember its HG - H - G
    combined_derivs[combined_dp_idxs] += HG_derivs
    combined_derivs[host_dp_idxs] -= H_derivs
    combined_derivs[guest_dp_idxs + len(host_params)] -= G_derivs
    # combined_derivs = 2*delta*combined_derivs # L2 derivative
    combined_derivs = (delta/np.abs(delta))*combined_derivs # L1 derivative

    return combined_derivs, pred_enthalpy, label

training_data = [
    ['guest-1.mol2', -2.17*4.184],
    ['guest-2.mol2', -4.19*4.184],
    ['guest-3.mol2', -5.46*4.184],
    ['guest-4.mol2', -2.74*4.184],
    ['guest-5.mol2', -2.99*4.184],
    ['guest-6.mol2', -2.53*4.184],
    ['guest-7.mol2', -3.4*4.184],
    ['guest-8.mol2', -4.89*4.184],
    ['guest-s9.mol2', -2.57*4.184],
    ['guest-s10.mol2', -2.68*4.184],
    ['guest-s11.mol2', -3.28*4.184],
    ['guest-s12.mol2', -4.2*4.184],
    ['guest-s13.mol2', -4.28*4.184],
    ['guest-s14.mol2', -4.66*4.184],
    ['guest-s15.mol2', -4.74*4.184],
    ['guest-s16.mol2', -2.75*4.184],
    ['guest-s17.mol2', -0.93*4.184],
    ['guest-s18.mol2', -2.75*4.184],
    ['guest-s19.mol2', -4.12*4.184],
    ['guest-s20.mol2', -3.36*4.184],
    ['guest-s21.mol2', -4.19*4.184],
    ['guest-s22.mol2', -4.48*4.184]
]


def initialize_parameters():

    _, _, (host_params, _), _ = serialize.deserialize_system('examples/host_acd.xml')

    # setting general smirnoff parameters for guest
    sdf_file = open(os.path.join(base_dir, 'guest-1.mol2'),'r').read()
    mol = Chem.MolFromMol2Block(sdf_file, sanitize=True, removeHs=False, cleanupSubstructures=True)
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    _, smirnoff_params, _, _, _ = forcefield.parameterize(mol, smirnoff)

    epoch_combined_params = np.concatenate([host_params, smirnoff_params])
    return epoch_combined_params

lr = .0005

# plot.write('epoch,guest,true enthalpy,computed enthalpy\n')
# loss_plot.write('epoch,r2,mae,loss\n')

prev_loss = None

init_params = initialize_parameters()
opt_init, opt_update, get_params = optimizers.adam(lr)
opt_state = opt_init(init_params)

num_data_points = len(training_data)
batch_size = num_gpus
num_batches = int(np.ceil(num_data_points/batch_size))

pool = multiprocessing.Pool(batch_size)

count = 0

for epoch in range(100):

    print('----epoch:', epoch, "started at", datetime.datetime.now(), '----')
    
    np.random.shuffle(training_data)

    epoch_predictions = []
    epoch_labels = []
    epoch_filenames = []

    for fn in training_data:
        epoch_filenames.append(fn)

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

    np.savez("run_"+str(epoch)+".npz", preds=epoch_predictions, labels=epoch_labels, filenames=fn)
    
    print("pearsonr", stats.pearsonr(epoch_predictions, epoch_labels), "r2_score:", sklearn.metrics.r2_score(epoch_predictions, epoch_labels), "mae:", np.mean(np.abs(epoch_predictions-epoch_labels)))
