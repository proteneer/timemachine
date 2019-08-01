import os
import sys
import numpy as np
import datetime
import sklearn.metrics

from scipy import stats
from rdkit import Chem

from simtk import openmm as mm
from simtk.openmm import app

from system import serialize
from system import forcefield
from system import simulation

from openforcefield.typing.engines.smirnoff import ForceField
from timemachine.lib import custom_ops
from jax.experimental import optimizers

import multiprocessing

num_gpus = 1
base_dir = "/home/yutong/Code/benchmarksets/input_files/cd-set1/mol2"

num_atom_types = 6

atom_type_map = {
    # 1: 0,
    # 12: 1,
    # 14: 2,
    # 16: 3,
    # 32: 4,
    # 35: 5,
    # 19: 6
    # 6: 7,
    12: 0,
    16: 1,
    1: 2,
    14: 3,
    32: 4,
    19: 5,
}



def atom_counts(all_masses):
    # A = []
    # for mol in all_masses:
    row = np.zeros(num_atom_types)
    for m in all_masses:
        anum = np.round(m)
        row[atom_type_map[anum]] += 1
    return row
    # A.append(row)

    # assert np.sum(row) == len(mol)

    # return A


def run_simulation(params):

    p = multiprocessing.current_process()
    combined_params, linear_params, guest_mol, label, idx = params
    os.environ['CUDA_VISIBLE_DEVICES'] = str(idx % num_gpus)


    filepath = 'examples/host_acd.xml'
    filename, file_extension = os.path.splitext(filepath)
    sys_xml = open(filepath, 'r').read()
    system = mm.XmlSerializer.deserialize(sys_xml)
    coords = np.loadtxt(filename + '.xyz').astype(np.float64)
    coords = coords/10


    host_potentials, host_conf, (dummy_host_params, host_param_groups), host_masses = serialize.deserialize_system(system, coords)
    host_params = combined_params[:len(dummy_host_params)]
    # guest_sdf = open(os.path.join(base_dir, guest_mol), "r").read()

    # print(pdb)

    mol = guest_mol
    # mol = Chem.MolFromMol2Block(guest_sdf, sanitize=True, removeHs=False, cleanupSubstructures=True)
    
    # all_mols = Chem.SDMolSupplier('/home/yutong/structures/anon/single_ligand.sdf', removeHs=False)
    # for mol in all_mols:
    #     if mol is None:
    #         assert 0
    #     break

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

    res = np.argwhere(filter_groups(host_param_groups, [7])).reshape(-1)

    # host_dp_idxs = np.argwhere(filter_groups(host_param_groups, [7])).reshape(-1)
    # guest_dp_idxs = np.argwhere(filter_groups(smirnoff_param_groups, [7])).reshape(-1)
    # combined_dp_idxs = np.argwhere(filter_groups(combined_param_groups, [7])).reshape(-1)

    host_dp_idxs = np.array([1])
    guest_dp_idxs = np.array([1])
    combined_dp_idxs = np.array([1])

    # guest or combined? we have much bigger sensitivity if we do guest only since the counts
    # have much more flexibility
    counts = atom_counts(guest_masses)
    atomization_energy = np.sum(counts * linear_params)

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

    # pred_enthalpy = -32870.99609375 - -31325.781005859375 - atomization_energy
    pred_enthalpy = HG_E - (G_E + H_E) - atomization_energy
    # pred_enthalpy = HG_E - (G_E + H_E)

    # print("HG_E, G_E+H_E, A_E", HG_E, (G_E + H_E), atomization_energy)

    delta = pred_enthalpy - label
    # print("pred_enthalpy, LABEL", pred_enthalpy, label)
    num_atoms = len(combined_masses)
    loss = delta**2

    print("--------------LOSS", np.abs(delta))
    # fancy index into the full derivative set
    combined_derivs = np.zeros_like(combined_params)
    # remember its HG - H - G

    # combined_derivs[combined_dp_idxs] += HG_derivs
    # combined_derivs[host_dp_idxs] -= H_derivs
    # combined_derivs[guest_dp_idxs + len(host_params)] -= G_derivs
    # # # combined_derivs = 2*delta*combined_derivs # L2 derivative
    # combined_derivs = (delta/np.abs(delta))*combined_derivs # L1 derivative
    linear_derivs = (delta/np.abs(delta))*counts*-1

    return combined_derivs, linear_derivs, pred_enthalpy, label

input_data = [
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


training_data = []

for path, v in input_data:
    guest_sdf = open(os.path.join(base_dir, path), "r").read()
    mol = Chem.MolFromMol2Block(guest_sdf, sanitize=True, removeHs=False, cleanupSubstructures=True)
    training_data.append((mol, v))


# all_mols = Chem.SDMolSupplier('/home/yutong/structures/anon/single_ligand.sdf', removeHs=False)
# for mol in all_mols:
#     if mol is None:
#         assert 0
#     training_data.append((mol, np.random.rand()*-10))

def initialize_parameters():

    filepath = 'examples/host_acd.xml'
    filename, file_extension = os.path.splitext(filepath)
    sys_xml = open(filepath, 'r').read()
    system = mm.XmlSerializer.deserialize(sys_xml)
    coords = np.loadtxt(filename + '.xyz').astype(np.float64)
    coords = coords/10

    _, _, (host_params, _), _ = serialize.deserialize_system(system, coords)

    # setting general smirnoff parameters for guest
    sdf_file = open(os.path.join(base_dir, 'guest-1.mol2'),'r').read()
    mol = Chem.MolFromMol2Block(sdf_file, sanitize=True, removeHs=False, cleanupSubstructures=True)
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    _, smirnoff_params, _, _, _ = forcefield.parameterize(mol, smirnoff)

    epoch_combined_params = np.concatenate([host_params, smirnoff_params])

    linear_params = np.random.rand(num_atom_types)

    # linear_params = np.array([-0.35451102, -0.1918766, -0.34991387, -0.7834076, -0.4676441, -0.6352691])

    return epoch_combined_params, linear_params

# lr = 0.0005
lr = 0.01

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
        # print("b_idx", b_idx)
        start_idx = b_idx*batch_size
        end_idx = min((b_idx+1)*batch_size, num_data_points)
        batch_data = training_data[start_idx:end_idx]

        args = []

        ff_params, linear_params = get_params(opt_state)
        print("LP", linear_params)
        for b_idx, b in enumerate(batch_data):
            args.append([ff_params, linear_params, b[0], b[1], b_idx])

        results = []
        for arg in args:
            results.append(run_simulation(arg))

        # assert 0

        batch_dp = np.zeros_like(ff_params)
        linear_dp = np.zeros_like(linear_params)

        # print("atomization parameters", linear_params)

        for grads, linear_grads, preds, labels in results:
            batch_dp += grads
            linear_dp += linear_grads
            epoch_predictions.append(preds)
            epoch_labels.append(labels)

        count += 1
        opt_state = opt_update(count, (batch_dp, linear_dp), opt_state)

    epoch_predictions = np.array(epoch_predictions)
    epoch_labels = np.array(epoch_labels)

    np.savez("run_"+str(epoch)+".npz", preds=epoch_predictions, labels=epoch_labels, filenames=fn, params=get_params(opt_state))
    
    print("\n\npearsonr", stats.pearsonr(epoch_predictions, epoch_labels), "r2_score:", sklearn.metrics.r2_score(epoch_predictions, epoch_labels), "mae:", np.mean(np.abs(epoch_predictions-epoch_labels)))
