import os
import sys

import time
import numpy as np
import argparse
import unittest
import ctypes
import random

import tensorflow as tf


import multiprocessing
from multiprocessing import Pool

from openeye import oechem
from openeye.oechem import OEMol, OEParseSmiles, OEAddExplicitHydrogens, OEGetIsotopicWeight, OEGetAverageWeight
from openeye import oeomega
from openeye.oechem import OEFloatArray

from openforcefield.utils import get_data_filename, generateTopologyFromOEMol
from openforcefield.typing.engines.smirnoff import get_molecule_parameterIDs, ForceField

from timemachine import observable
from timemachine import minimizer
from timemachine.constants import BOLTZ
from timemachine import system_builder
from timemachine.cpu_functionals import custom_ops

from simtk import openmm

from tensorflow.python.client import device_lib


ksize = 200 # reservoir size FIXME
batch_size = 8 # number of GPUs


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_abc_coefficents(
    masses,
    dt,
    friction,
    temperature):
    """
    Get a,b,c coefficients of the integrator.
    """
    vscale = np.exp(-dt*friction)
    if friction == 0:
        fscale = dt
    else:
        fscale = (1-vscale)/friction

    kT = BOLTZ * temperature
    nscale = np.sqrt(kT*(1-vscale*vscale)) # noise scale
    invMasses = (1.0/masses).reshape((-1, 1))
    sqrtInvMasses = np.sqrt(invMasses)

    coeff_a = vscale
    coeff_bs = fscale*invMasses
    coeff_cs = nscale*sqrtInvMasses

    return coeff_a, coeff_bs, coeff_cs

def estimate_buffer_size(epsilon, coeff_a):
    """
    Estimate optimal size of the buffer to achieve convergence
    """
    return np.int64(np.log(epsilon)/np.log(coeff_a)+1)


def get_masses(mol):
    masses = []
    for atom in mol.GetAtoms():
        elem = atom.GetAtomicNum()
        mass = atom.GetIsotope()
        masses.append(OEGetIsotopicWeight(elem, mass))

    return np.array(masses)

def mol_coords_to_numpy_array(mol):
    coords = OEFloatArray(mol.GetMaxAtomIdx() * 3)
    mol.GetCoords(coords)
    arr = np.ctypeslib.as_array(ctypes.cast(int(coords.PtrCast()), ctypes.POINTER(ctypes.c_float)), shape=(len(coords),))
    return np.array(arr.reshape((-1, 3)))

def write_xyz(ofs, mol, coords):
    mol.SetCoords(OEFloatArray(coords.reshape(-1)))
    oechem.OEWriteMolecule(ofs, mol)


def initialize_system(
    smiles,
    dt=0.001,
    temperature=100,
    forcefield_file='forcefield/smirnoff99Frosst.offxml',
    am1_charges=True):



    mol = OEMol()
    # OEParseSmiles(mol, 'CCOCCSCC')
    # OEParseSmiles(mol, 'c1ccccc1')
    OEParseSmiles(mol, smiles)
    # OEParseSmiles(mol, 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O')
    OEAddExplicitHydrogens(mol)
    masses = get_masses(mol)
    num_atoms = mol.NumAtoms()

    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.SetMaxConfs(1)
    omega = oeomega.OEOmega(omegaOpts)
    omega.SetStrictStereo(False)

    if not omega(mol):
        assert 0

    topology = generateTopologyFromOEMol(mol)

    ff = ForceField(get_data_filename(forcefield_file))

    nrgs, total_params, offsets, charge_idxs = system_builder.construct_energies(ff, mol, am1_charges)

    # dt = 0.0025
    # friction = 10.0
    # temperature = 300

    # gradient descent
    dt = dt
    friction = 10.0
    temperature = temperature

    a,b,c = get_abc_coefficents(masses, dt, friction, temperature)

    buf_size = estimate_buffer_size(1e-10, a)
    # print("BUFFER_SIZE", buf_size)
    x0 = mol_coords_to_numpy_array(mol)/10

    intg = custom_ops.Integrator_double(
        dt,
        buf_size,
        num_atoms,
        total_params,
        a,
        b,
        c
    )

    context = custom_ops.Context_double(
        nrgs,
        intg
    )

    x0 = minimizer.minimize_newton_cg(nrgs, x0, total_params)

    return nrgs, offsets, intg, context, x0, total_params, charge_idxs

def run_once(nrgs, context, intg, x0, n_steps, total_params, ksize, inference):
    x0 = minimizer.minimize_newton_cg(nrgs, x0, total_params)
    origin = np.sum(x0, axis=0)/x0.shape[0]
    num_atoms = x0.shape[0]
    num_steps = n_steps
    # ofs = oechem.oemolostream("new_frames.xyz")
    # ofs.SetFormat(oechem.OEFormat_XYZ)

    intg.reset()
    intg.set_coordinates(x0.reshape(-1).tolist())
    intg.set_velocities(np.zeros_like(x0).reshape(-1).tolist())

    start_time = time.time()

    k = ksize
    reservoir = []

    for step in range(n_steps):
        # coords = intg.get_coordinates()
        # dxdps = intg.get_dxdp()
        # reservoir.append((np.array(coords).reshape((num_atoms, 3)), np.array(dxdps).reshape((total_params, num_atoms, 3))))
        if step < k:
            coords = intg.get_coordinates()
            dxdps = intg.get_dxdp()
            reservoir.append((np.array(coords).reshape((num_atoms, 3)), np.array(dxdps).reshape((total_params, num_atoms, 3))))
        else:
            j = random.randint(0, step)
            if j < k:
                coords = intg.get_coordinates()
                dxdps = intg.get_dxdp()
                reservoir[j] = (np.array(coords).reshape((num_atoms, 3)), np.array(dxdps).reshape((total_params, num_atoms, 3)))
        context.step(inference)

    confs = []
    for r in reservoir:
        confs.append(r[0])
    confs = np.array(confs)

    dxdps = []
    for r in reservoir:
        dxdps.append(r[1])
    dxdps = np.array(dxdps)

    return confs, dxdps

def generate_observables(smiles):

    pid = multiprocessing.current_process().pid % batch_size
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pid)

    nrgs1, offsets1, intg1, context1, init_x_1, total_params_1, gci1 = initialize_system(
        smiles=smiles,
        dt=0.001,
        temperature=100,
        forcefield_file='forcefield/smirnoff99Frosst.offxml',
        am1_charges=True
    )

    # generate the observable
    print("generating observable for", smiles)
    # FIXME
    confs1, _ = run_once(nrgs1, context1, intg1, init_x_1, 40000, total_params_1, ksize, inference=True) 

    return confs1

def train_molecule(args):
    smiles = args[0]
    global_params = args[1]

    pid = multiprocessing.current_process().pid % batch_size
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pid)

    nrgs0, offsets0, intg0, context0, init_x_0, total_params_0, gci0 = initialize_system(
        smiles=smiles,
        dt=0.001,
        temperature=100,
        forcefield_file='forcefield/smirnoff99Frosst_perturbed.offxml',
        am1_charges=False
    )

    for nrg in nrgs0:
        if isinstance(nrg, custom_ops.ElectrostaticsGPU_double):
            new_params = []
            for p_idx in gci0:
                new_params.append(global_params[p_idx])
            nrg.set_params(new_params)


    # FIXME *100
    confs0, dxdp0 = run_once(nrgs0, context0, intg0, init_x_0, 10000, total_params_0, ksize, inference=False)

    return confs0, dxdp0, gci0, offsets0, nrgs0


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def train_charges(all_smiles):

    all_observables = []
    all_mutual_losses = []

    global_params = np.array([
        0.5,
        0.2,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.5,
        0.5,
        0.15,
        0.2,
        0.2,
        0.2,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5
    ])

    with Pool(batch_size) as p:
        label_confs = p.map(generate_observables, all_smiles)

    print("starting training...")
    es_learning_rate = np.array([[0.001]])

    for epoch in range(1000):
        print("starting epoch...", epoch, "global params", global_params)
        tf.reset_default_graph()
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session()


        for batch_idxs in batch(range(0, len(all_smiles)), batch_size):
            train_confs = []
            train_dxdps = []
            train_gcis = []
            train_offsets = []
            train_nrgs = []

            batch_smiles_train = []
            batch_label_confs = []

            with Pool(batch_size) as p:

                args = []
                for idx in batch_idxs:
                    args.append((all_smiles[idx], global_params))

                results = p.map(train_molecule, args) # need to update parameters from global parameter pool
                for r in results:
                    train_confs.append(r[0])
                    train_dxdps.append(r[1])
                    train_gcis.append(r[2])
                    train_offsets.append(r[3])
                    train_nrgs.append(r[4])

            batch_labels = []
            for idx in batch_idxs:
                batch_labels.append(label_confs[idx])

            grads = np.zeros_like(global_params)
            batch_loss = 0

            for conf, dxdp, gci, offset, nrgs, label_conf in zip(train_confs, train_dxdps, train_gcis, train_offsets, train_nrgs, batch_labels):
                # print("processing...")
                x0 = tf.convert_to_tensor(conf)
                obs0_rij = observable.sorted_squared_distances(x0)
                obs1_rij = observable.sorted_squared_distances(tf.convert_to_tensor(label_conf))
                loss = tf.sqrt(tf.reduce_sum(tf.pow(obs0_rij - obs1_rij, 2))/ksize) # RMSE
                x0_grads = tf.gradients(loss, x0)[0]
                loss_np, dLdx = sess.run([loss, x0_grads])
                batch_loss += loss_np
                dLdx = np.expand_dims(dLdx, 1) # [B, 1, N, 3]
                dLdp = np.multiply(dLdx, dxdp) # dL/dx * dx/dp [B, P, N, 3]
                dp = np.sum(dLdp, axis=(0,2,3))

                for dparams, nrg in zip(np.split(dp, offset)[1:], nrgs):
                    if isinstance(nrg, custom_ops.ElectrostaticsGPU_double):
                        dp = es_learning_rate * dparams.reshape((-1, 1))

                        for p_grad, p_idx in zip(dp, gci):
                            # print("adjusting", p_idx, "by", p_grad)
                            global_params[p_idx] -= p_grad

            print("-----------Batch loss", batch_loss, batch_idxs)

            # if epoch > 2:
                # assert 0
            sys.stdout.flush()


if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description='Stability testing.')
    # parser.add_argument('--smiles', dest='smiles', help='what temperature we should run at')
    # args = parser.parse_args()
    # smiles = [
    #     "CCCCCOCCCC",
    #     "CCOCCCCOCCC",
    #     "CCCC",
    #     "CCOCC(CCN)CC",
    #     "CCOCC"
    # ]

    smiles = [
        "C(C(C(O)O)O)O",
        "C(C(CO)O)C(O)O",
        "C(C(CO)O)O",
        "C(C(O)O)(O)O",
        "C(C(O)O)C(O)O",
        "C(C(O)O)C(O)OCO",
        "C(C(O)O)O",
        "C(C(O)O)OCO",
        "C(C(O)OC(O)O)O",
        "C(C(O)OCO)O",
        "C(CC(O)O)CO",
        "C(CCO)CC(O)O",
        "C(CCO)CCO",
        "C(CCO)CO",
        "C(CO)C(C(O)O)O",
        "C(CO)C(CC(O)O)O",
        "C(CO)C(CCO)O",
        "C(CO)C(CO)O",
        "C(CO)C(O)O",
        "C(CO)C(O)OC(O)O",
        "C(CO)C(O)OCO",
        "C(CO)CO",
        "C(CO)COCO",
        "C(CO)O",
        "C(COC(O)O)O",
        "C(COCC(O)O)O",
        "C(COCCO)O",
        "C(COCO)C(O)O",
        "C(COCO)O",
        "C(O)(O)O",
        "C(O)(O)OC(O)O",
        "C(O)O",
        "C(O)OC(C(O)O)O",
        "C(O)OC(O)O"
    ]

    train_charges(smiles)