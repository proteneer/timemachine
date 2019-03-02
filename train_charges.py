import os
import sys
import sklearn

import traceback
import math

import time
import numpy as np
import argparse
import unittest
import ctypes
import random

import datetime


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


num_train_samples = 60 # 75% of this will be used for train, 25% will be used for test
ksize = 100 # reservoir size
batch_size = 8 # number of GPUs
obs_steps = 400
train_steps = 100


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
    nrglist,
    total_params,
    masses,
    mol,
    dt=0.001,
    temperature=50,
    am1_charges=True):

    num_atoms = mol.NumAtoms()
    nrg_funcs = [
        custom_ops.HarmonicBondGPU_double,
        custom_ops.HarmonicAngleGPU_double,
        custom_ops.PeriodicTorsionGPU_double,
        custom_ops.LennardJonesGPU_double,
        custom_ops.ElectrostaticsGPU_double
    ]

    nrgs = []
    for func, group in zip(nrg_funcs, nrglist):
        nrgs.append(func(*group))

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

    return nrgs, intg, context, x0

def run_once(nrgs, context, intg, x0, n_steps, total_params, ksize, inference):
    x0 = minimizer.minimize_newton_cg(nrgs, x0, total_params)
    origin = np.sum(x0, axis=0)/x0.shape[0]
    num_atoms = x0.shape[0]
    num_steps = n_steps

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

def generate_observables(args):

    nrg_params = args[0]
    total_params = args[1]
    masses = args[2]
    mol = args[3]

    pid = multiprocessing.current_process().pid % batch_size
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(pid)

    nrgs, intg, context, x0 = initialize_system(nrg_params, total_params, masses, mol)

    confs, _ = run_once(nrgs, context, intg, x0, obs_steps, total_params, ksize, inference=True) 

    return confs

def test_molecule(args):

    try:
        global_params = args[0]
        nrg_params = args[1]
        total_params = args[2]
        masses = args[3]
        mol = args[4]
        charge_idxs = args[5]

        pid = multiprocessing.current_process().pid % batch_size
        os.environ["CUDA_VISIBLE_DEVICES"] = str(pid)

        nrgs, intg, context, x0 = initialize_system(nrg_params, total_params, masses, mol)

        for nrg in nrgs:
            if isinstance(nrg, custom_ops.ElectrostaticsGPU_double):
                new_params = []
                for p_idx in charge_idxs:
                    new_params.append(global_params[p_idx])
                nrg.set_params(new_params)

        confs, none = run_once(nrgs, context, intg, x0, train_steps, total_params, ksize, inference=True)


        return confs, none, nrgs

    except Exception as e:

        print("TRACEBACK")
        traceback.print_exc()
        print("EXCEPTION CAUGHT", e)
        raise e

def train_molecule(args):

    try:
        global_params = args[0]
        nrg_params = args[1]
        total_params = args[2]
        masses = args[3]
        mol = args[4]
        charge_idxs = args[5]

        pid = multiprocessing.current_process().pid % batch_size
        os.environ["CUDA_VISIBLE_DEVICES"] = str(pid)

        nrgs, intg, context, x0 = initialize_system(nrg_params, total_params, masses, mol)

        for nrg in nrgs:
            if isinstance(nrg, custom_ops.ElectrostaticsGPU_double):
                new_params = []
                for p_idx in charge_idxs:
                    new_params.append(global_params[p_idx])
                nrg.set_params(new_params)

        confs, dxdp = run_once(nrgs, context, intg, x0, train_steps, total_params, ksize, inference=False)


        return confs, dxdp, nrgs

    except Exception as e:

        print("TRACEBACK")
        traceback.print_exc()
        print("EXCEPTION CAUGHT", e)
        raise e



def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def train_charges(all_smiles):

    global_params = np.array([
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
        -0.1,
        -0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        -0.2,
        -0.2,
        -0.2,
        -0.1,
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
        0.1,
        0.1,
        0.1,
        0.1
    ])

    # step 1. reference system generated using am1 charges and test systems generated using atom-typed charges
    reference_args = []
    all_args = []
    all_offset_idxs = []
    all_charge_idxs = []

    for smi_idx, smiles in enumerate(all_smiles):
        print("setting up", smiles, smi_idx, "/", len(all_smiles))
        mol = OEMol()
        OEParseSmiles(mol, smiles)
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
        reference_forcefield_file = 'forcefield/smirnoff99Frosst_perturbed.offxml'
        ff = ForceField(get_data_filename(reference_forcefield_file))
        params = system_builder.construct_energies(ff, mol, True)
        reference_args.append((params[0], params[1], masses, mol))

        params = system_builder.construct_energies(ff, mol, False)
        all_args.append((global_params, params[0], params[1], masses, mol, params[3]))
        all_offset_idxs.append(params[2])
        all_charge_idxs.append(params[3])

    # step 2. generate a collection of conformations from each molecules to train against
    with Pool(batch_size) as p:
        label_confs = p.map(generate_observables, reference_args)

    print("starting training...")
    es_learning_rate = np.array([[0.001]])

    num_batches = math.ceil(len(all_smiles) / batch_size)

    train_batches = int(0.75*num_batches)
    test_batches = num_batches - train_batches

    num_train_samples = train_batches*batch_size
    num_test_samples = len(all_smiles) - num_train_samples
    print("num train batches:", train_batches, "num test batches:", test_batches)

    for epoch in range(1000):
        print('--------------------')

        date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

        print(date, "starting epoch...", epoch, "global params", global_params.tolist())
        train_epoch_loss = 0
        test_epoch_loss = 0

        all_perm = np.arange(len(all_smiles))
        all_perm[:num_train_samples] = np.random.permutation(num_train_samples)

        label_confs = [label_confs[i] for i in all_perm]
        all_args = [all_args[i] for i in all_perm]
        all_offset_idxs = [all_offset_idxs[i] for i in all_perm]
        all_charge_idxs = [all_charge_idxs[i] for i in all_perm]

        for bidx, batch_idxs in enumerate(batch(range(0, len(all_smiles)), batch_size)):
            train_confs = []
            train_dxdps = []
            train_gcis = []
            train_offsets = []
            train_nrgs = []

            batch_smiles_train = []
            batch_label_confs = []

            start_time = time.time()
            with Pool(batch_size) as p:
                args = []
                for idx in batch_idxs:
                    train_gcis.append(all_charge_idxs[idx])
                    train_offsets.append(all_offset_idxs[idx])
                    args.append(all_args[idx])
                    # args.append((all_smiles[idx], global_params))

                if bidx < train_batches:
                    results = p.map(train_molecule, args) # need to update parameters from global parameter pool
                else:
                    results = p.map(test_molecule, args)

                for r in results:
                    train_confs.append(r[0])
                    train_dxdps.append(r[1])
                    train_nrgs.append(r[2])

            train_time = time.time() - start_time
            start_time = time.time()

            batch_labels = []
            for idx in batch_idxs:
                batch_labels.append(label_confs[idx])

            grads = np.zeros_like(global_params)
            batch_train_loss = 0
            batch_test_loss = 0

            for conf, dxdp, gci, offset, nrgs, label_conf in zip(train_confs, train_dxdps, train_gcis, train_offsets, train_nrgs, batch_labels):
                # print("processing...")

                tf.reset_default_graph()
                sess = tf.Session()
                x0 = tf.convert_to_tensor(conf)
                obs0_rij = observable.sorted_squared_distances(x0)
                obs1_rij = observable.sorted_squared_distances(tf.convert_to_tensor(label_conf))
                loss = tf.sqrt(tf.reduce_sum(tf.pow(obs0_rij - obs1_rij, 2))/ksize) # RMSE

                if bidx < train_batches:
                    x0_grads = tf.gradients(loss, x0)[0]
                    loss_np, dLdx = sess.run([loss, x0_grads])
                    batch_train_loss += loss_np
                    train_epoch_loss += loss_np

                    if loss_np > 10:
                        print("giant_loss detected, skipping", loss_np)
                        continue

                    dLdx = np.expand_dims(dLdx, 1) # [B, 1, N, 3]
                    dLdp = np.multiply(dLdx, dxdp) # dL/dx * dx/dp [B, P, N, 3]
                    dp = np.sum(dLdp, axis=(0,2,3))

                    for dparams, nrg in zip(np.split(dp, offset)[1:], nrgs):
                        if isinstance(nrg, custom_ops.ElectrostaticsGPU_double):
                            dp = es_learning_rate * dparams.reshape((-1, 1))
                            if np.any(np.isnan(dp)):
                                print("nan grad:", clipped_dp)
                                continue
                            amax, amin = np.amax(dp), np.amin(dp)
                            if amax > 1e-2 or amin < -1e2:
                                print("excessively large gradient:", dp)
                                continue
                            for p_grad, p_idx in zip(dp, gci):
                                global_params[p_idx] -= p_grad
                else:
                    test_loss = sess.run(loss)
                    batch_test_loss += test_loss
                    test_epoch_loss += test_loss

            date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

            if bidx < train_batches:
                print(date, "avg train batch loss", batch_train_loss/len(batch_idxs), batch_idxs, "reduce_time:", time.time()-start_time, "train_time:", train_time)
            else:
                print(date, "avg test batch loss", batch_test_loss/len(batch_idxs), batch_idxs, "reduce_time:", time.time()-start_time, "train_time:", train_time)

            sys.stdout.flush()

        print(date, 'epoch', epoch, 'train loss', train_epoch_loss/num_train_samples, 'test loss', test_epoch_loss/num_test_samples)

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description='Stability testing.')
    # parser.add_argument('--smiles', dest='smiles', help='what temperature we should run at')
    # args = parser.parse_args()


    # smiles = [
    #     "C(C(C(O)O)O)O",
    #     "C(C(CO)O)C(O)O",
    #     "C(C(CO)O)O",
    #     "C(C(O)O)(O)O",
    #     "C(C(O)O)C(O)O",
    #     "C(C(O)O)C(O)OCO",
    #     "C(C(O)O)O",
    #     "C(C(O)O)OCO",
    #     "C(C(O)OC(O)O)O",
    #     "C(C(O)OCO)O",
    #     "C(CC(O)O)CO",
    #     "C(CCO)CC(O)O",
    #     "C(CCO)CCO",
    #     "C(CCO)CO",
    #     "C(CO)C(C(O)O)O",
    #     "C(CO)C(CC(O)O)O",
    #     "C(CO)C(CCO)O",
    #     "C(CO)C(CO)O",
    #     "C(CO)C(O)O",
    #     "C(CO)C(O)OC(O)O",
    #     "C(CO)C(O)OCO",
    #     "C(CO)CO",
    #     "C(CO)COCO",
    #     "C(CO)O",
    #     "C(COC(O)O)O",
    #     "C(COCC(O)O)O",
    #     "C(COCCO)O",
    #     "C(COCO)C(O)O",
    #     "C(COCO)O",
    #     "C(O)(O)O",
    #     "C(O)(O)OC(O)O",
    #     "C(O)O",
    #     "C(O)OC(C(O)O)O",
    #     "C(O)OC(O)O"
    # ]

    smiles_train = [
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
        "C(O)OC(O)O",
        "C(O)OC(O)OC(O)O",
        "C(O)OC(O)OCO",
        "C(O)OCO",
        "CC",
        "CC(C(C(C)(C)C)O)O",
        "CC(C(C(C)(C)O)O)O",
        "CC(C(C(C)(O)O)O)O",
        "CC(C(C(C)O)O)O",
        "CC(C(C(O)O)O)O",
        "CC(C(C)(C(C)(C)C)O)O",
        "CC(C(C)(C(C)(C)O)O)O",
        "CC(C(C)(C(C)(O)O)O)O",
        "CC(C(C)(C(C)O)O)O",
        "CC(C(C)(C(O)O)O)O",
        "CC(C(C)(C)C(C)(C)C)O",
        "CC(C(C)(C)C(C)(C)O)O",
        "CC(C(C)(C)C(C)(O)O)O",
        "CC(C(C)(C)C(C)O)O",
        "CC(C(C)(C)C(O)O)O",
        "CC(C(C)(C)C)O",
        "CC(C(C)(C)C)OC(C)O",
        "CC(C(C)(C)C)OCO",
        "CC(C(C)(C)CC(C)(C)C)O",
        "CC(C(C)(C)CC(C)(C)O)O",
        "CC(C(C)(C)CC(C)(O)O)O",
        "CC(C(C)(C)CC(O)O)O",
        "CC(C(C)(C)CCO)O",
        "CC(C(C)(C)CO)O",
        "CC(C(C)(C)COC)O",
        "CC(C(C)(C)O)O",
        "CC(C(C)(C)O)OC(C)O",
        "CC(C(C)(C)O)OCO",
        "CC(C(C)(C)OC(C)(C)C)O",
        "CC(C(C)(C)OC(C)(C)O)O",
        "CC(C(C)(C)OC(C)(O)O)O",
        "CC(C(C)(C)OC(C)O)O",
        "CC(C(C)(C)OC(O)O)O",
        "CC(C(C)(C)OC)O",
        "CC(C(C)(C)OCO)O",
        "CC(C(C)(CC(C)(C)C)O)O",
        "CC(C(C)(CC(C)(C)O)O)O",
        "CC(C(C)(CC(C)(O)O)O)O",
        "CC(C(C)(CC(O)O)O)O",
        "CC(C(C)(CCO)O)O",
        "CC(C(C)(CO)O)O",
        "CC(C(C)(COC)O)O",
        "CC(C(C)(O)O)(O)O",
        "CC(C(C)(O)O)O",
        "CC(C(C)(O)O)OC(C)O",
        "CC(C(C)(O)O)OCO",
        "CC(C(C)(O)OC(C)(C)C)O",
        "CC(C(C)(O)OC(C)(C)O)O",
        "CC(C(C)(O)OC(C)(O)O)O",
        "CC(C(C)(O)OC(C)O)O",
        "CC(C(C)(O)OC(O)O)O",
        "CC(C(C)(O)OC)O",
        "CC(C(C)(O)OCO)(O)O",
        "CC(C(C)(O)OCO)O",
        "CC(C(C)O)C(C)(C)C",
        "CC(C(C)O)C(C)(C)O",
        "CC(C(C)O)C(C)(O)O",
        "CC(C(C)O)C(C)O",
        "CC(C(C)O)C(O)O",
        "CC(C(C)O)O",
        "CC(C(C)OC(C)(C)C)O",
        "CC(C(C)OC(C)(C)O)O",
        "CC(C(C)OC(C)(O)O)O",
        "CC(C(C)OC(C)O)O",
        "CC(C(C)OC(O)O)O",
        "CC(C(C)OC)O",
        "CC(C(C)OCO)O",
        "CC(C(CC(C)(C)C)O)O",
        "CC(C(CC(C)(C)O)O)O",
        "CC(C(CC(C)(O)O)O)O",
        "CC(C(CC(O)O)O)O",
        "CC(C(CCO)O)(O)O",
        "CC(C(CCO)O)O",
        "CC(C(CO)O)(O)O",
        "CC(C(CO)O)O",
        "CC(C(COC)O)O",
        "CC(C(O)O)(O)O",
        "CC(C(O)O)(O)OCO",
        "CC(C(O)O)O",
        "CC(C(O)O)OC(C)O",
        "CC(C(O)O)OCO",
        "CC(C(O)OC(C)(C)C)O",
        "CC(C(O)OC(C)(C)O)O",
        "CC(C(O)OC(C)(O)O)O",
        "CC(C(O)OC(C)O)O",
        "CC(C(O)OC(O)O)O",
        "CC(C(O)OC)O",
        "CC(C(O)OCO)(O)O",
        "CC(C(O)OCO)O",
        "CC(C)(C(C)(C)O)O",
        "CC(C)(C(C)(C)OCO)O",
        "CC(C)(C(C)(CCO)O)O",
        "CC(C)(C(C)(CO)O)O",
        "CC(C)(C(C)(O)O)O",
        "CC(C)(C(C)(O)O)OCO",
        "CC(C)(C(C)(O)OCO)O",
        "CC(C)(C(CCO)O)O",
        "CC(C)(C(CO)O)O",
        "CC(C)(C(O)O)O",
        "CC(C)(C(O)O)OCO",
        "CC(C)(C(O)OCO)O",
        "CC(C)(C)C",
        "CC(C)(C)C(C)(C)C",
        "CC(C)(C)C(C)(C)CCO",
        "CC(C)(C)C(C)(C)CO",
        "CC(C)(C)C(C)(C)O",
        "CC(C)(C)C(C)(C)OCO",
        "CC(C)(C)C(C)(CCO)O",
        "CC(C)(C)C(C)(CO)O",
        "CC(C)(C)C(C)(O)O",
        "CC(C)(C)C(C)(O)OCO",
        "CC(C)(C)C(CCO)O",
        "CC(C)(C)C(CO)O",
        "CC(C)(C)C(O)O",
        "CC(C)(C)C(O)OCO",
        "CC(C)(C)CC(C)(C)C",
        "CC(C)(C)CC(C)(C)CCO",
        "CC(C)(C)CC(C)(C)CO",
        "CC(C)(C)CC(C)(C)O",
        "CC(C)(C)CC(C)(C)OCO",
        "CC(C)(C)CC(C)(CCO)O",
        "CC(C)(C)CC(C)(CO)O",
        "CC(C)(C)CC(C)(O)O",
        "CC(C)(C)CC(C)(O)OCO",
        "CC(C)(C)CC(CCO)O",
        "CC(C)(C)CC(CO)O",
        "CC(C)(C)CC(O)O",
        "CC(C)(C)CC(O)OCO",
        "CC(C)(C)CCCCO",
        "CC(C)(C)CCCO",
        "CC(C)(C)CCO",
        "CC(C)(C)CCOCO",
        "CC(C)(C)CO",
        "CC(C)(C)COC",
        "CC(C)(C)COCCO",
        "CC(C)(C)COCO",
        "CC(C)(C)O",
        "CC(C)(C)OC",
        "CC(C)(C)OC(C)(C)C",
        "CC(C)(C)OC(C)(C)CCO",
        "CC(C)(C)OC(C)(C)CO",
        "CC(C)(C)OC(C)(C)O",
        "CC(C)(C)OC(C)(C)OCO",
        "CC(C)(C)OC(C)(CCO)O",
        "CC(C)(C)OC(C)(CO)O",
        "CC(C)(C)OC(C)(O)O",
        "CC(C)(C)OC(C)(O)OCO",
        "CC(C)(C)OC(CCO)O",
        "CC(C)(C)OC(CO)O",
        "CC(C)(C)OC(O)O",
        "CC(C)(C)OC(O)OCO",
        "CC(C)(C)OCCO",
        "CC(C)(C)OCO",
        "CC(C)(CC(C)(C)O)CO",
        "CC(C)(CC(C)(C)O)O",
        "CC(C)(CC(C)(C)OCO)O",
        "CC(C)(CC(C)(CCO)O)O",
        "CC(C)(CC(C)(CO)O)O",
        "CC(C)(CC(C)(O)O)CO",
        "CC(C)(CC(C)(O)O)O",
        "CC(C)(CC(C)(O)O)OCO",
        "CC(C)(CC(C)(O)OCO)O",
        "CC(C)(CC(CCO)O)O",
        "CC(C)(CC(CO)O)O",
        "CC(C)(CC(O)O)CO",
        "CC(C)(CC(O)O)O",
        "CC(C)(CC(O)O)OCO",
        "CC(C)(CC(O)OCO)O",
        "CC(C)(CCCCO)O",
        "CC(C)(CCCO)O",
        "CC(C)(CCO)C(C)(C)O",
        "CC(C)(CCO)C(C)(O)O",
        "CC(C)(CCO)C(O)O",
        "CC(C)(CCO)CC(C)(C)O",
        "CC(C)(CCO)CC(C)(O)O",
        "CC(C)(CCO)CC(O)O",
        "CC(C)(CCO)CCO",
        "CC(C)(CCO)CO",
        "CC(C)(CCO)COC",
        "CC(C)(CCO)O",
        "CC(C)(CCO)OC",
        "CC(C)(CCO)OC(C)(C)O",
        "CC(C)(CCO)OC(C)(O)O",
        "CC(C)(CCO)OC(O)O",
        "CC(C)(CCO)OCO",
        "CC(C)(CCOCO)O",
        "CC(C)(CO)C(C)(C)O",
        "CC(C)(CO)C(C)(O)O",
        "CC(C)(CO)C(O)O",
        "CC(C)(CO)CO",
        "CC(C)(CO)COC",
        "CC(C)(CO)O",
        "CC(C)(CO)OC",
        "CC(C)(CO)OC(C)(C)O",
        "CC(C)(CO)OC(C)(O)O",
        "CC(C)(CO)OC(O)O",
        "CC(C)(CO)OCO",
        "CC(C)(COC)O",
        "CC(C)(COC)OCO",
        "CC(C)(COCCO)O",
        "CC(C)(COCO)O",
        "CC(C)(O)O",
        "CC(C)(O)OC",
        "CC(C)(O)OC(C)(C)O",
        "CC(C)(O)OC(C)(C)OCO",
        "CC(C)(O)OC(C)(CCO)O",
        "CC(C)(O)OC(C)(CO)O",
        "CC(C)(O)OC(C)(O)O",
        "CC(C)(O)OC(C)(O)OCO",
        "CC(C)(O)OC(CCO)O",
        "CC(C)(O)OC(CO)O",
        "CC(C)(O)OC(O)O",
        "CC(C)(O)OC(O)OCO",
        "CC(C)(O)OCCO",
        "CC(C)(O)OCO",
        "CC(C)(OC)OCO",
        "CC(C)(OCO)OC(C)(O)O",
        "CC(C)(OCO)OC(O)O",
        "CC(C)(OCO)OCO",
        "CC(C)C",
        "CC(C)C(C(C)(C)C)O",
        "CC(C)C(C(C)(C)O)O",
        "CC(C)C(C(C)(O)O)O",
        "CC(C)C(C(C)C)O",
        "CC(C)C(C(C)O)O",
        "CC(C)C(C(O)O)O",
        "CC(C)C(C)(C(C)(C)C)O",
        "CC(C)C(C)(C(C)(C)O)O",
        "CC(C)C(C)(C(C)(O)O)O",
        "CC(C)C(C)(C(C)C)O",
        "CC(C)C(C)(C(C)O)O",
        "CC(C)C(C)(C(O)O)O",
        "CC(C)C(C)(C)C",
        "CC(C)C(C)(C)C(C)(C)C",
        "CC(C)C(C)(C)C(C)(C)O",
        "CC(C)C(C)(C)C(C)(O)O",
        "CC(C)C(C)(C)C(C)C",
        "CC(C)C(C)(C)C(C)O",
        "CC(C)C(C)(C)C(O)O",
        "CC(C)C(C)(C)CC(C)(C)C",
        "CC(C)C(C)(C)CC(C)(C)O",
        "CC(C)C(C)(C)CC(C)(O)O",
        "CC(C)C(C)(C)CC(C)O",
        "CC(C)C(C)(C)CC(O)O",
        "CC(C)C(C)(C)CCO",
        "CC(C)C(C)(C)CO",
        "CC(C)C(C)(C)COC",
        "CC(C)C(C)(C)O",
        "CC(C)C(C)(C)OC",
        "CC(C)C(C)(C)OC(C)(C)C",
        "CC(C)C(C)(C)OC(C)(C)O",
        "CC(C)C(C)(C)OC(C)(O)O",
        "CC(C)C(C)(C)OC(C)C",
        "CC(C)C(C)(C)OC(C)O",
        "CC(C)C(C)(C)OC(O)O",
        "CC(C)C(C)(C)OCO",
        "CC(C)C(C)(CC(C)(C)C)O",
        "CC(C)C(C)(CC(C)(C)O)O",
        "CC(C)C(C)(CC(C)(O)O)O",
        "CC(C)C(C)(CC(C)O)O",
        "CC(C)C(C)(CC(O)O)O",
        "CC(C)C(C)(CCO)O",
        "CC(C)C(C)(CO)O",
        "CC(C)C(C)(COC)O",
        "CC(C)C(C)(O)O",
        "CC(C)C(C)(O)OC",
        "CC(C)C(C)(O)OC(C)(C)C",
        "CC(C)C(C)(O)OC(C)(C)O",
        "CC(C)C(C)(O)OC(C)(O)O",
        "CC(C)C(C)(O)OC(C)C",
        "CC(C)C(C)(O)OC(C)O",
        "CC(C)C(C)(O)OC(O)O",
        "CC(C)C(C)(O)OCO",
        "CC(C)C(C)C",
        "CC(C)C(C)C(C)(C)C",
        "CC(C)C(C)C(C)(C)O",
        "CC(C)C(C)C(C)(O)O",
        "CC(C)C(C)C(C)C",
        "CC(C)C(C)C(C)O",
        "CC(C)C(C)C(O)O",
        "CC(C)C(C)CC(C)(C)C",
        "CC(C)C(C)CC(C)(C)O",
        "CC(C)C(C)CC(C)(O)O",
        "CC(C)C(C)CC(C)O",
        "CC(C)C(C)CC(O)O",
        "CC(C)C(C)CCO",
        "CC(C)C(C)CO",
        "CC(C)C(C)COC",
        "CC(C)C(C)O",
        "CC(C)C(C)OC",
        "CC(C)C(C)OC(C)(C)C",
        "CC(C)C(C)OC(C)(C)O",
        "CC(C)C(C)OC(C)(O)O",
        "CC(C)C(C)OC(C)C",
        "CC(C)C(C)OC(C)O",
        "CC(C)C(C)OC(O)O",
        "CC(C)C(C)OCO",
        "CC(C)C(CC(C)(C)C)O",
        "CC(C)C(CC(C)(C)O)O",
        "CC(C)C(CC(C)(O)O)O",
        "CC(C)C(CC(C)O)O",
        "CC(C)C(CC(O)O)O",
        "CC(C)C(CCO)O",
        "CC(C)C(CO)O",
        "CC(C)C(COC)O",
        "CC(C)C(O)O",
        "CC(C)C(O)OC",
        "CC(C)C(O)OC(C)(C)C",
        "CC(C)C(O)OC(C)(C)O",
        "CC(C)C(O)OC(C)(O)O",
        "CC(C)C(O)OC(C)C",
        "CC(C)C(O)OC(C)O",
        "CC(C)C(O)OC(O)O",
        "CC(C)C(O)OCO",
        "CC(C)CC(C(C)(C)C)O",
        "CC(C)CC(C(C)(C)O)O",
        "CC(C)CC(C(C)(O)O)O",
        "CC(C)CC(C(C)C)O",
        "CC(C)CC(C(C)O)O",
        "CC(C)CC(C(O)O)O",
        "CC(C)CC(C)(C(C)(C)C)O",
        "CC(C)CC(C)(C(C)(C)O)O",
        "CC(C)CC(C)(C(C)(O)O)O",
        "CC(C)CC(C)(C(C)C)O",
        "CC(C)CC(C)(C(C)O)O",
        "CC(C)CC(C)(C(O)O)O",
        "CC(C)CC(C)(C)C",
        "CC(C)CC(C)(C)C(C)(C)C",
        "CC(C)CC(C)(C)C(C)(C)O",
        "CC(C)CC(C)(C)C(C)(O)O",
        "CC(C)CC(C)(C)C(C)C",
        "CC(C)CC(C)(C)C(C)O",
        "CC(C)CC(C)(C)C(O)O",
        "CC(C)CC(C)(C)CC(C)(C)C",
        "CC(C)CC(C)(C)CC(C)(C)O",
        "CC(C)CC(C)(C)CC(C)(O)O",
        "CC(C)CC(C)(C)CC(C)C",
        "CC(C)CC(C)(C)CC(C)O",
        "CC(C)CC(C)(C)CC(O)O",
        "CC(C)CC(C)(C)CCO",
        "CC(C)CC(C)(C)CO",
        "CC(C)CC(C)(C)COC",
        "CC(C)CC(C)(C)O",
        "CC(C)CC(C)(C)OC",
        "CC(C)CC(C)(C)OC(C)(C)C",
        "CC(C)CC(C)(C)OC(C)(C)O",
        "CC(C)CC(C)(C)OC(C)(O)O",
        "CC(C)CC(C)(C)OC(C)C",
        "CC(C)CC(C)(C)OC(C)O",
        "CC(C)CC(C)(C)OC(O)O",
        "CC(C)CC(C)(C)OCO",
        "CC(C)CC(C)(CC(C)(C)C)O",
        "CC(C)CC(C)(CC(C)(C)O)O",
        "CC(C)CC(C)(CC(C)(O)O)O",
        "CC(C)CC(C)(CC(C)C)O",
        "CC(C)CC(C)(CC(C)O)O",
        "CC(C)CC(C)(CC(O)O)O",
        "CC(C)CC(C)(CCO)O",
        "CC(C)CC(C)(CO)O",
        "CC(C)CC(C)(COC)O",
        "CC(C)CC(C)(O)O",
        "CC(C)CC(C)(O)OC",
        "CC(C)CC(C)(O)OC(C)(C)C",
        "CC(C)CC(C)(O)OC(C)(C)O",
        "CC(C)CC(C)(O)OC(C)(O)O",
        "CC(C)CC(C)(O)OC(C)C",
        "CC(C)CC(C)(O)OC(C)O",
        "CC(C)CC(C)(O)OC(O)O",
        "CC(C)CC(C)(O)OCO",
        "CC(C)CC(C)C",
        "CC(C)CC(C)C(C)(C)C",
        "CC(C)CC(C)C(C)(C)O",
        "CC(C)CC(C)C(C)(O)O",
        "CC(C)CC(C)C(C)C",
        "CC(C)CC(C)C(C)O",
        "CC(C)CC(C)C(O)O",
        "CC(C)CC(C)CC(C)(C)C",
        "CC(C)CC(C)CC(C)(C)O",
        "CC(C)CC(C)CC(C)(O)O",
        "CC(C)CC(C)CC(C)C",
        "CC(C)CC(C)CC(C)O",
        "CC(C)CC(C)CC(O)O",
        "CC(C)CC(C)CCO",
        "CC(C)CC(C)CO",
        "CC(C)CC(C)COC",
        "CC(C)CC(C)O",
        "CC(C)CC(C)OC",
        "CC(C)CC(C)OC(C)(C)C",
        "CC(C)CC(C)OC(C)(C)O",
        "CC(C)CC(C)OC(C)(O)O",
        "CC(C)CC(C)OC(C)C",
        "CC(C)CC(C)OC(C)O",
        "CC(C)CC(C)OC(O)O",
        "CC(C)CC(C)OCO",
        "CC(C)CC(CC(C)(C)C)O",
        "CC(C)CC(CC(C)(C)O)O",
        "CC(C)CC(CC(C)(O)O)O",
        "CC(C)CC(CC(C)C)O",
        "CC(C)CC(CC(C)O)O",
        "CC(C)CC(CC(O)O)O",
        "CC(C)CC(CCO)O",
        "CC(C)CC(CO)O",
        "CC(C)CC(COC)O",
        "CC(C)CC(O)O",
        "CC(C)CC(O)OC",
        "CC(C)CC(O)OC(C)(C)C",
        "CC(C)CC(O)OC(C)(C)O",
        "CC(C)CC(O)OC(C)(O)O",
        "CC(C)CC(O)OC(C)C",
        "CC(C)CC(O)OC(C)O",
        "CC(C)CC(O)OC(O)O",
        "CC(C)CC(O)OCO",
        "CC(C)CCC(C)(C)C",
        "CC(C)CCC(C)(C)O",
        "CC(C)CCC(C)(O)O",
        "CC(C)CCC(C)C",
        "CC(C)CCC(C)O",
        "CC(C)CCC(O)O",
        "CC(C)CCCC(C)(C)C",
        "CC(C)CCCC(C)(C)O",
        "CC(C)CCCC(C)(O)O",
        "CC(C)CCCC(C)C",
        "CC(C)CCCC(C)O",
        "CC(C)CCCC(O)O",
        "CC(C)CCCCO",
        "CC(C)CCCO",
        "CC(C)CCCOC",
        "CC(C)CCO",
        "CC(C)CCOC",
        "CC(C)CCOC(C)C",
        "CC(C)CCOC(C)O",
        "CC(C)CCOCO",
        "CC(C)CO",
        "CC(C)COC",
        "CC(C)COC(C)(C)C",
        "CC(C)COC(C)(C)O",
        "CC(C)COC(C)(O)O",
        "CC(C)COC(C)C",
        "CC(C)COC(C)O",
        "CC(C)COC(O)O",
        "CC(C)COCC(C)(C)C",
        "CC(C)COCC(C)(C)O",
        "CC(C)COCC(C)(O)O",
        "CC(C)COCC(C)C",
        "CC(C)COCC(C)O",
        "CC(C)COCC(O)O",
        "CC(C)COCCO",
        "CC(C)COCO",
        "CC(C)COCOC",
        "CC(C)O",
        "CC(C)OC",
        "CC(C)OC(C(C)(C)C)O",
        "CC(C)OC(C(C)(C)O)O",
        "CC(C)OC(C(C)(O)O)O",
        "CC(C)OC(C(C)O)O",
        "CC(C)OC(C(O)O)O",
        "CC(C)OC(C)(C(C)(C)C)O",
        "CC(C)OC(C)(C(C)(C)O)O",
        "CC(C)OC(C)(C(C)(O)O)O",
        "CC(C)OC(C)(C(C)O)O",
        "CC(C)OC(C)(C(O)O)O",
        "CC(C)OC(C)(C)C",
        "CC(C)OC(C)(C)C(C)(C)C",
        "CC(C)OC(C)(C)C(C)(C)O",
        "CC(C)OC(C)(C)C(C)(O)O",
        "CC(C)OC(C)(C)C(C)O",
        "CC(C)OC(C)(C)C(O)O",
        "CC(C)OC(C)(C)CC(C)(C)C",
        "CC(C)OC(C)(C)CC(C)(C)O",
        "CC(C)OC(C)(C)CC(C)(O)O",
        "CC(C)OC(C)(C)CC(C)O",
        "CC(C)OC(C)(C)CC(O)O",
        "CC(C)OC(C)(C)CCO",
        "CC(C)OC(C)(C)CO",
        "CC(C)OC(C)(C)COC",
        "CC(C)OC(C)(C)O",
        "CC(C)OC(C)(C)OC",
        "CC(C)OC(C)(C)OC(C)(C)C",
        "CC(C)OC(C)(C)OC(C)(C)O",
        "CC(C)OC(C)(C)OC(C)(O)O",
        "CC(C)OC(C)(C)OC(C)C",
        "CC(C)OC(C)(C)OC(C)O",
        "CC(C)OC(C)(C)OC(O)O",
        "CC(C)OC(C)(C)OCO",
        "CC(C)OC(C)(CC(C)(C)C)O",
        "CC(C)OC(C)(CC(C)(C)O)O",
        "CC(C)OC(C)(CC(C)(O)O)O",
        "CC(C)OC(C)(CC(C)O)O",
        "CC(C)OC(C)(CC(O)O)O",
        "CC(C)OC(C)(CCO)O",
        "CC(C)OC(C)(CO)O",
        "CC(C)OC(C)(COC)O",
        "CC(C)OC(C)(O)O",
        "CC(C)OC(C)(O)OC",
        "CC(C)OC(C)(O)OC(C)(C)C",
        "CC(C)OC(C)(O)OC(C)(C)O",
        "CC(C)OC(C)(O)OC(C)(O)O",
        "CC(C)OC(C)(O)OC(C)C",
        "CC(C)OC(C)(O)OC(C)O",
        "CC(C)OC(C)(O)OC(O)O",
        "CC(C)OC(C)(O)OCO",
        "CC(C)OC(C)C",
        "CC(C)OC(C)C(C)(C)C",
        "CC(C)OC(C)C(C)(C)O",
        "CC(C)OC(C)C(C)(O)O",
        "CC(C)OC(C)C(C)O",
        "CC(C)OC(C)C(O)O",
        "CC(C)OC(C)CC(C)(C)C",
        "CC(C)OC(C)CC(C)(C)O",
        "CC(C)OC(C)CC(C)(O)O",
        "CC(C)OC(C)CC(C)O",
        "CC(C)OC(C)CC(O)O",
        "CC(C)OC(C)CCO",
        "CC(C)OC(C)CO",
        "CC(C)OC(C)COC",
        "CC(C)OC(C)O",
        "CC(C)OC(C)OC",
        "CC(C)OC(C)OC(C)(C)C",
        "CC(C)OC(C)OC(C)(C)O",
        "CC(C)OC(C)OC(C)(O)O",
        "CC(C)OC(C)OC(C)C",
        "CC(C)OC(C)OC(C)O",
        "CC(C)OC(C)OC(O)O",
        "CC(C)OC(C)OCO",
        "CC(C)OC(CC(C)(C)C)O",
        "CC(C)OC(CC(C)(C)O)O",
        "CC(C)OC(CC(C)(O)O)O",
        "CC(C)OC(CC(C)O)O",
        "CC(C)OC(CC(O)O)O",
        "CC(C)OC(CCO)O",
        "CC(C)OC(CO)O",
        "CC(C)OC(COC)O",
        "CC(C)OC(O)O",
        "CC(C)OC(O)OC",
        "CC(C)OC(O)OC(C)(C)C",
        "CC(C)OC(O)OC(C)(C)O",
        "CC(C)OC(O)OC(C)(O)O",
        "CC(C)OC(O)OC(C)C",
        "CC(C)OC(O)OC(C)O",
        "CC(C)OC(O)OC(O)O",
        "CC(C)OC(O)OCO",
        "CC(C)OCC(C)(C)C",
        "CC(C)OCC(C)(C)O",
        "CC(C)OCC(C)(O)O",
        "CC(C)OCC(C)O",
        "CC(C)OCC(O)O",
        "CC(C)OCCC(C)(C)C",
        "CC(C)OCCC(C)(C)O",
        "CC(C)OCCC(C)(O)O",
        "CC(C)OCCC(C)O",
        "CC(C)OCCC(O)O",
        "CC(C)OCCCO",
        "CC(C)OCCO",
        "CC(C)OCCOC",
        "CC(C)OCO",
        "CC(C)OCOC",
        "CC(CC(C(C)(C)C)O)O",
        "CC(CC(C(C)(C)O)O)O",
        "CC(CC(C(C)(O)O)O)O",
        "CC(CC(C(C)O)O)O",
        "CC(CC(C(O)O)O)O",
        "CC(CC(C)(C(C)(C)C)O)O",
        "CC(CC(C)(C(C)(C)O)O)O",
        "CC(CC(C)(C(C)(O)O)O)O",
        "CC(CC(C)(C(C)O)O)O",
        "CC(CC(C)(C(O)O)O)O",
        "CC(CC(C)(C)C(C)(C)C)O",
        "CC(CC(C)(C)C(C)(C)O)O",
        "CC(CC(C)(C)C(C)(O)O)O",
        "CC(CC(C)(C)C(C)O)O",
        "CC(CC(C)(C)C(O)O)O",
        "CC(CC(C)(C)C)C(C)O",
        "CC(CC(C)(C)C)CO",
        "CC(CC(C)(C)C)O",
        "CC(CC(C)(C)C)OC(C)O",
        "CC(CC(C)(C)C)OCO",
        "CC(CC(C)(C)CC(C)(C)C)O",
        "CC(CC(C)(C)CC(C)(C)O)O",
        "CC(CC(C)(C)CC(C)(O)O)O",
        "CC(CC(C)(C)CC(C)O)O",
        "CC(CC(C)(C)CC(O)O)O",
        "CC(CC(C)(C)CCO)O",
        "CC(CC(C)(C)CO)O",
        "CC(CC(C)(C)COC)O",
        "CC(CC(C)(C)O)C(C)O",
        "CC(CC(C)(C)O)CO",
        "CC(CC(C)(C)O)O",
        "CC(CC(C)(C)O)OC(C)O",
        "CC(CC(C)(C)O)OCO",
        "CC(CC(C)(C)OC(C)(C)C)O",
        "CC(CC(C)(C)OC(C)(C)O)O",
        "CC(CC(C)(C)OC(C)(O)O)O",
        "CC(CC(C)(C)OC(C)O)O",
        "CC(CC(C)(C)OC(O)O)O",
        "CC(CC(C)(C)OC)O",
        "CC(CC(C)(C)OCO)O",
        "CC(CC(C)(CC(C)(C)C)O)O",
        "CC(CC(C)(CC(C)(C)O)O)O",
        "CC(CC(C)(CC(C)(O)O)O)O",
        "CC(CC(C)(CC(C)O)O)O",
        "CC(CC(C)(CC(O)O)O)O",
        "CC(CC(C)(CCO)O)O",
        "CC(CC(C)(CO)O)O",
        "CC(CC(C)(COC)O)O",
        "CC(CC(C)(O)O)(CO)O",
        "CC(CC(C)(O)O)(O)O",
        "CC(CC(C)(O)O)C(C)O",
        "CC(CC(C)(O)O)CO",
        "CC(CC(C)(O)O)O",
        "CC(CC(C)(O)O)OC(C)O",
        "CC(CC(C)(O)O)OCO",
        "CC(CC(C)(O)OC(C)(C)C)O",
        "CC(CC(C)(O)OC(C)(C)O)O",
        "CC(CC(C)(O)OC(C)(O)O)O",
        "CC(CC(C)(O)OC(C)O)O",
        "CC(CC(C)(O)OC(O)O)O",
        "CC(CC(C)(O)OC)O",
        "CC(CC(C)(O)OCO)(O)O",
        "CC(CC(C)(O)OCO)O",
        "CC(CC(C)O)C(C)(C)C",
        "CC(CC(C)O)C(C)(C)O",
        "CC(CC(C)O)C(C)(O)O",
        "CC(CC(C)O)C(C)O",
        "CC(CC(C)O)C(O)O",
        "CC(CC(C)O)CC(C)(C)C",
        "CC(CC(C)O)CC(C)(C)O",
        "CC(CC(C)O)CC(C)(O)O",
        "CC(CC(C)O)CC(C)O",
        "CC(CC(C)O)CC(O)O",
        "CC(CC(C)O)CO",
        "CC(CC(C)O)COC",
        "CC(CC(C)O)O",
        "CC(CC(C)OC(C)(C)C)O",
        "CC(CC(C)OC(C)(C)O)O",
        "CC(CC(C)OC(C)(O)O)O",
        "CC(CC(C)OC(C)O)O",
        "CC(CC(C)OC(O)O)O",
        "CC(CC(C)OC)O",
        "CC(CC(C)OCO)O",
        "CC(CC(CC(C)(C)C)O)O",
        "CC(CC(CC(C)(C)O)O)O",
        "CC(CC(CC(C)(O)O)O)O",
        "CC(CC(CC(C)O)O)O",
        "CC(CC(CC(O)O)O)O",
        "CC(CC(CCO)O)(O)O",
        "CC(CC(CCO)O)O",
        "CC(CC(CO)O)(O)O",
        "CC(CC(CO)O)O",
        "CC(CC(COC)O)O",
        "CC(CC(O)O)(CO)O",
        "CC(CC(O)O)(O)O",
        "CC(CC(O)O)(O)OCO",
        "CC(CC(O)O)C(C)O",
        "CC(CC(O)O)CO",
        "CC(CC(O)O)O",
        "CC(CC(O)O)OC(C)O",
        "CC(CC(O)O)OCO",
        "CC(CC(O)OC(C)(C)C)O",
        "CC(CC(O)OC(C)(C)O)O",
        "CC(CC(O)OC(C)(O)O)O",
        "CC(CC(O)OC(C)O)O",
        "CC(CC(O)OC(O)O)O",
        "CC(CC(O)OC)O",
        "CC(CC(O)OCO)(O)O",
        "CC(CC(O)OCO)O",
        "CC(CCC(C)(C)C)O",
        "CC(CCC(C)(C)O)O",
        "CC(CCC(C)(O)O)O",
        "CC(CCC(C)O)O",
        "CC(CCC(O)O)O",
        "CC(CCCC(C)(C)C)O",
        "CC(CCCC(C)(C)O)O",
        "CC(CCCC(C)(O)O)O",
        "CC(CCCC(C)O)O",
        "CC(CCCC(O)O)O",
        "CC(CCCCO)(O)O",
        "CC(CCCCO)O",
        "CC(CCCO)(O)O",
        "CC(CCCO)O",
        "CC(CCCOC)O",
        "CC(CCO)(C(C)(O)O)O",
        "CC(CCO)(C(O)O)O",
        "CC(CCO)(CC(C)(O)O)O",
        "CC(CCO)(CC(O)O)O",
        "CC(CCO)(CCO)O",
        "CC(CCO)(CO)O",
        "CC(CCO)(COC)O",
        "CC(CCO)(O)O",
        "CC(CCO)(O)OC",
        "CC(CCO)(O)OC(C)(O)O",
        "CC(CCO)(O)OC(O)O",
        "CC(CCO)(O)OCO",
        "CC(CCO)C(C)(C)C",
        "CC(CCO)C(C)(C)O",
        "CC(CCO)C(C)(O)O",
        "CC(CCO)C(C)O",
        "CC(CCO)C(O)O",
        "CC(CCO)CC(C)(C)C",
        "CC(CCO)CC(C)(C)O",
        "CC(CCO)CC(C)(O)O",
        "CC(CCO)CC(C)O",
        "CC(CCO)CC(O)O",
        "CC(CCO)CCO",
        "CC(CCO)CO",
        "CC(CCO)COC",
        "CC(CCO)O",
        "CC(CCO)OC",
        "CC(CCO)OC(C)(C)C",
        "CC(CCO)OC(C)(C)O",
        "CC(CCO)OC(C)(O)O",
        "CC(CCO)OC(C)O",
        "CC(CCO)OC(O)O",
        "CC(CCO)OCO",
        "CC(CCOC(C)O)O",
        "CC(CCOC)O",
        "CC(CCOCO)(O)O",
        "CC(CCOCO)O",
        "CC(CO)(C(C)(O)O)O",
        "CC(CO)(C(O)O)O",
        "CC(CO)(CO)O",
        "CC(CO)(COC)O",
        "CC(CO)(O)O",
        "CC(CO)(O)OC",
        "CC(CO)(O)OC(C)(O)O",
        "CC(CO)(O)OC(O)O",
        "CC(CO)(O)OCO",
        "CC(CO)C(C)(C)C",
        "CC(CO)C(C)(C)O",
        "CC(CO)C(C)(O)O",
        "CC(CO)C(C)O",
        "CC(CO)C(O)O",
        "CC(CO)CO",
        "CC(CO)COC",
        "CC(CO)O",
        "CC(CO)OC",
        "CC(CO)OC(C)(C)C",
        "CC(CO)OC(C)(C)O",
        "CC(CO)OC(C)(O)O",
        "CC(CO)OC(C)O",
        "CC(CO)OC(O)O",
        "CC(CO)OCO",
        "CC(COC(C)(C)C)O",
        "CC(COC(C)(C)O)O",
        "CC(COC(C)(O)O)O",
        "CC(COC(C)O)O",
        "CC(COC(O)O)O",
        "CC(COC)(O)O",
        "CC(COC)(O)OCO",
        "CC(COC)C(C)O",
        "CC(COC)O",
        "CC(COC)OC(C)O",
        "CC(COC)OCO",
        "CC(COCC(C)(C)C)O",
        "CC(COCC(C)(C)O)O",
        "CC(COCC(C)(O)O)O",
        "CC(COCC(C)O)O",
        "CC(COCC(O)O)O",
        "CC(COCCO)(O)O",
        "CC(COCCO)O",
        "CC(COCO)(O)O",
        "CC(COCO)O",
        "CC(COCOC)O",
        "CC(O)(O)O",
        "CC(O)(O)OC",
        "CC(O)(O)OC(C)(O)O",
        "CC(O)(O)OC(C)(O)OCO",
        "CC(O)(O)OC(CCO)O",
        "CC(O)(O)OC(CO)O",
        "CC(O)(O)OC(O)O",
        "CC(O)(O)OC(O)OCO",
        "CC(O)(O)OCCO",
        "CC(O)(O)OCO",
        "CC(O)(OC)OCO",
        "CC(O)(OCO)OC(O)O",
        "CC(O)(OCO)OCO",
        "CC(O)O",
        "CC(O)OC",
        "CC(O)OC(C(C)(C)C)O",
        "CC(O)OC(C(C)(C)O)O",
        "CC(O)OC(C(C)(O)O)O",
        "CC(O)OC(C(O)O)O",
        "CC(O)OC(C)(C(C)(C)C)O",
        "CC(O)OC(C)(C(C)(C)O)O",
        "CC(O)OC(C)(C(C)(O)O)O",
        "CC(O)OC(C)(C(O)O)O",
        "CC(O)OC(C)(C)C",
        "CC(O)OC(C)(C)C(C)(C)C",
        "CC(O)OC(C)(C)C(C)(C)O",
        "CC(O)OC(C)(C)C(C)(O)O",
        "CC(O)OC(C)(C)C(O)O",
        "CC(O)OC(C)(C)CC(C)(C)C",
        "CC(O)OC(C)(C)CC(C)(C)O",
        "CC(O)OC(C)(C)CC(C)(O)O",
        "CC(O)OC(C)(C)CC(O)O",
        "CC(O)OC(C)(C)CCO",
        "CC(O)OC(C)(C)CO",
        "CC(O)OC(C)(C)COC",
        "CC(O)OC(C)(C)O",
        "CC(O)OC(C)(C)OC",
        "CC(O)OC(C)(C)OC(C)(C)C",
        "CC(O)OC(C)(C)OC(C)(C)O",
        "CC(O)OC(C)(C)OC(C)(O)O",
        "CC(O)OC(C)(C)OC(C)O",
        "CC(O)OC(C)(C)OC(O)O",
        "CC(O)OC(C)(C)OCO",
        "CC(O)OC(C)(CC(C)(C)C)O",
        "CC(O)OC(C)(CC(C)(C)O)O",
        "CC(O)OC(C)(CC(C)(O)O)O",
        "CC(O)OC(C)(CC(O)O)O",
        "CC(O)OC(C)(CCO)O",
        "CC(O)OC(C)(CO)O",
        "CC(O)OC(C)(COC)O",
        "CC(O)OC(C)(O)O",
        "CC(O)OC(C)(O)OC",
        "CC(O)OC(C)(O)OC(C)(C)C",
        "CC(O)OC(C)(O)OC(C)(C)O",
        "CC(O)OC(C)(O)OC(C)(O)O",
        "CC(O)OC(C)(O)OC(C)O",
        "CC(O)OC(C)(O)OC(O)O",
        "CC(O)OC(C)(O)OCO",
        "CC(O)OC(C)O",
        "CC(O)OC(C)OC",
        "CC(O)OC(C)OC(C)(C)C",
        "CC(O)OC(C)OC(C)(C)O",
        "CC(O)OC(C)OC(C)(O)O",
        "CC(O)OC(C)OC(C)O",
        "CC(O)OC(C)OC(O)O",
        "CC(O)OC(C)OCO",
        "CC(O)OC(CC(C)(C)C)O",
        "CC(O)OC(CC(C)(C)O)O",
        "CC(O)OC(CC(C)(O)O)O",
        "CC(O)OC(CC(O)O)O",
        "CC(O)OC(CCO)O",
        "CC(O)OC(CO)O",
        "CC(O)OC(COC)O",
        "CC(O)OC(O)O",
        "CC(O)OC(O)OC",
        "CC(O)OC(O)OC(C)(C)C",
        "CC(O)OC(O)OC(C)(C)O",
        "CC(O)OC(O)OC(C)(O)O",
        "CC(O)OC(O)OC(C)O",
        "CC(O)OC(O)OC(O)O",
        "CC(O)OC(O)OCO",
        "CC(O)OCC(C)(C)C",
        "CC(O)OCC(C)(C)O",
        "CC(O)OCC(C)(O)O",
        "CC(O)OCC(O)O",
        "CC(O)OCCC(C)(C)C",
        "CC(O)OCCC(C)(C)O",
        "CC(O)OCCC(C)(O)O",
        "CC(O)OCCC(O)O",
        "CC(O)OCCCO",
        "CC(O)OCCO",
        "CC(O)OCCOC",
        "CC(O)OCO",
        "CC(O)OCOC",
        "CC(OC)OCO",
        "CC(OCO)OC(C)(C)C",
        "CC(OCO)OC(C)(C)O",
        "CC(OCO)OC(C)(O)O",
        "CC(OCO)OC(O)O",
        "CC(OCO)OCO",
        "CCC",
        "CCC(C(C)(C)C)O",
        "CCC(C(C)(C)O)O",
        "CCC(C(C)(O)O)O",
        "CCC(C(C)C)O",
        "CCC(C(C)O)O",
        "CCC(C(O)O)O",
        "CCC(C)(C(C)(C)C)O",
        "CCC(C)(C(C)(C)O)O",
        "CCC(C)(C(C)(O)O)O",
        "CCC(C)(C(C)C)O",
        "CCC(C)(C(C)O)O",
        "CCC(C)(C(O)O)O",
        "CCC(C)(C)C",
        "CCC(C)(C)C(C)(C)C",
        "CCC(C)(C)C(C)(C)O",
        "CCC(C)(C)C(C)(O)O",
        "CCC(C)(C)C(C)C",
        "CCC(C)(C)C(C)O",
        "CCC(C)(C)C(O)O",
        "CCC(C)(C)CC",
        "CCC(C)(C)CC(C)(C)C",
        "CCC(C)(C)CC(C)(C)O",
        "CCC(C)(C)CC(C)(O)O",
        "CCC(C)(C)CC(C)C",
        "CCC(C)(C)CC(C)O",
        "CCC(C)(C)CC(O)O",
        "CCC(C)(C)CCO",
        "CCC(C)(C)CO",
        "CCC(C)(C)COC",
        "CCC(C)(C)O",
        "CCC(C)(C)OC",
        "CCC(C)(C)OC(C)(C)C",
        "CCC(C)(C)OC(C)(C)O",
        "CCC(C)(C)OC(C)(O)O",
        "CCC(C)(C)OC(C)C",
        "CCC(C)(C)OC(C)O",
        "CCC(C)(C)OC(O)O",
        "CCC(C)(C)OCC",
        "CCC(C)(C)OCO",
        "CCC(C)(CC(C)(C)C)O",
        "CCC(C)(CC(C)(C)O)O",
        "CCC(C)(CC(C)(O)O)O",
        "CCC(C)(CC(C)C)O",
        "CCC(C)(CC(C)O)O",
        "CCC(C)(CC(O)O)O",
        "CCC(C)(CC)O",
        "CCC(C)(CCO)O",
        "CCC(C)(CO)O",
        "CCC(C)(COC)O",
        "CCC(C)(O)O",
        "CCC(C)(O)OC",
        "CCC(C)(O)OC(C)(C)C",
        "CCC(C)(O)OC(C)(C)O",
        "CCC(C)(O)OC(C)(O)O",
        "CCC(C)(O)OC(C)C",
        "CCC(C)(O)OC(C)O",
        "CCC(C)(O)OC(O)O",
        "CCC(C)(O)OCC",
        "CCC(C)(O)OCO",
        "CCC(C)C",
        "CCC(C)C(C)(C)C",
        "CCC(C)C(C)(C)O",
        "CCC(C)C(C)(O)O",
        "CCC(C)C(C)C",
        "CCC(C)C(C)O",
        "CCC(C)C(O)O",
        "CCC(C)CC",
        "CCC(C)CC(C)(C)C",
        "CCC(C)CC(C)(C)O",
        "CCC(C)CC(C)(O)O",
        "CCC(C)CC(C)C",
        "CCC(C)CC(C)O",
        "CCC(C)CC(O)O",
        "CCC(C)CCO",
        "CCC(C)CO",
        "CCC(C)COC",
        "CCC(C)O",
        "CCC(C)OC",
        "CCC(C)OC(C)(C)C",
        "CCC(C)OC(C)(C)O",
        "CCC(C)OC(C)(O)O",
        "CCC(C)OC(C)C",
        "CCC(C)OC(C)O",
        "CCC(C)OC(O)O",
        "CCC(C)OCC",
        "CCC(C)OCO",
        "CCC(CC(C)(C)C)O",
        "CCC(CC(C)(C)O)O",
        "CCC(CC(C)(O)O)O",
        "CCC(CC(C)C)O",
        "CCC(CC(C)O)O",
        "CCC(CC(O)O)O",
        "CCC(CC)O",
        "CCC(CCO)O",
        "CCC(CO)O",
        "CCC(COC)O",
        "CCC(O)O",
        "CCC(O)OC",
        "CCC(O)OC(C)(C)C",
        "CCC(O)OC(C)(C)O",
        "CCC(O)OC(C)(O)O",
        "CCC(O)OC(C)C",
        "CCC(O)OC(C)O",
        "CCC(O)OC(O)O",
        "CCC(O)OCC",
        "CCC(O)OCO",
        "CCCC",
        "CCCC(C(C)(C)C)O",
        "CCCC(C(C)(C)O)O",
        "CCCC(C(C)(O)O)O",
        "CCCC(C(C)C)O",
        "CCCC(C(C)O)O",
        "CCCC(C(O)O)O",
        "CCCC(C)(C(C)(C)C)O",
        "CCCC(C)(C(C)(C)O)O",
        "CCCC(C)(C(C)(O)O)O",
        "CCCC(C)(C(C)C)O",
        "CCCC(C)(C(C)O)O",
        "CCCC(C)(C(O)O)O",
        "CCCC(C)(C)C",
        "CCCC(C)(C)C(C)(C)C",
        "CCCC(C)(C)C(C)(C)O",
        "CCCC(C)(C)C(C)(O)O",
        "CCCC(C)(C)C(C)C",
        "CCCC(C)(C)C(C)O",
        "CCCC(C)(C)C(O)O",
        "CCCC(C)(C)CC",
        "CCCC(C)(C)CC(C)(C)C",
        "CCCC(C)(C)CC(C)(C)O",
        "CCCC(C)(C)CC(C)(O)O",
        "CCCC(C)(C)CC(C)C",
        "CCCC(C)(C)CC(C)O",
        "CCCC(C)(C)CC(O)O",
        "CCCC(C)(C)CCC",
        "CCCC(C)(C)CCO",
        "CCCC(C)(C)CO",
        "CCCC(C)(C)COC",
        "CCCC(C)(C)O",
        "CCCC(C)(C)OC",
        "CCCC(C)(C)OC(C)(C)C",
        "CCCC(C)(C)OC(C)(C)O",
        "CCCC(C)(C)OC(C)(O)O",
        "CCCC(C)(C)OC(C)C",
        "CCCC(C)(C)OC(C)O",
        "CCCC(C)(C)OC(O)O",
        "CCCC(C)(C)OCC",
        "CCCC(C)(C)OCO",
        "CCCC(C)(CC(C)(C)C)O",
        "CCCC(C)(CC(C)(C)O)O",
        "CCCC(C)(CC(C)(O)O)O",
        "CCCC(C)(CC(C)C)O",
        "CCCC(C)(CC(C)O)O",
        "CCCC(C)(CC(O)O)O",
        "CCCC(C)(CC)O",
        "CCCC(C)(CCC)O",
        "CCCC(C)(CCO)O",
        "CCCC(C)(CO)O",
        "CCCC(C)(COC)O",
        "CCCC(C)(O)O",
        "CCCC(C)(O)OC",
        "CCCC(C)(O)OC(C)(C)C",
        "CCCC(C)(O)OC(C)(C)O",
        "CCCC(C)(O)OC(C)(O)O",
        "CCCC(C)(O)OC(C)C",
        "CCCC(C)(O)OC(C)O",
        "CCCC(C)(O)OC(O)O",
        "CCCC(C)(O)OCC",
        "CCCC(C)(O)OCO",
        "CCCC(C)C",
        "CCCC(C)C(C)(C)C",
        "CCCC(C)C(C)(C)O",
        "CCCC(C)C(C)(O)O",
        "CCCC(C)C(C)C",
        "CCCC(C)C(C)O",
        "CCCC(C)C(O)O",
        "CCCC(C)CC",
        "CCCC(C)CC(C)(C)C",
        "CCCC(C)CC(C)(C)O",
        "CCCC(C)CC(C)(O)O",
        "CCCC(C)CC(C)C",
        "CCCC(C)CC(C)O",
        "CCCC(C)CC(O)O",
        "CCCC(C)CCC",
        "CCCC(C)CCO",
        "CCCC(C)CO",
        "CCCC(C)COC",
        "CCCC(C)O",
        "CCCC(C)OC",
        "CCCC(C)OC(C)(C)C",
        "CCCC(C)OC(C)(C)O",
        "CCCC(C)OC(C)(O)O",
        "CCCC(C)OC(C)C",
        "CCCC(C)OC(C)O",
        "CCCC(C)OC(O)O",
        "CCCC(C)OCC",
        "CCCC(C)OCO",
        "CCCC(CC(C)(C)C)O",
        "CCCC(CC(C)(C)O)O",
        "CCCC(CC(C)(O)O)O",
        "CCCC(CC(C)C)O",
        "CCCC(CC(C)O)O",
        "CCCC(CC(O)O)O",
        "CCCC(CC)O",
        "CCCC(CCC)O",
        "CCCC(CCO)O",
        "CCCC(CO)O",
        "CCCC(COC)O",
        "CCCC(O)O",
        "CCCC(O)OC",
        "CCCC(O)OC(C)(C)C",
        "CCCC(O)OC(C)(C)O",
        "CCCC(O)OC(C)(O)O",
        "CCCC(O)OC(C)C",
        "CCCC(O)OC(C)O",
        "CCCC(O)OC(O)O",
        "CCCC(O)OCC",
        "CCCC(O)OCO",
        "CCCCC",
        "CCCCC(C)(C)C",
        "CCCCC(C)(C)O",
        "CCCCC(C)(O)O",
        "CCCCC(C)C",
        "CCCCC(C)O",
        "CCCCC(O)O",
        "CCCCCC",
        "CCCCCC(C)(C)C",
        "CCCCCC(C)(C)O",
        "CCCCCC(C)(O)O",
        "CCCCCC(C)C",
        "CCCCCC(C)O",
        "CCCCCC(O)O",
        "CCCCCCC",
        "CCCCCCO",
        "CCCCCO",
        "CCCCCOC",
        "CCCCO",
        "CCCCOC",
        "CCCCOC(C)C",
        "CCCCOC(C)O",
        "CCCCOCC",
        "CCCCOCO",
        "CCCO",
        "CCCOC",
        "CCCOC(C)(C)C",
        "CCCOC(C)(C)O",
        "CCCOC(C)(O)O",
        "CCCOC(C)C",
        "CCCOC(C)O",
        "CCCOC(O)O",
        "CCCOCC"
        "CCCOCC(C)(C)C",
        "CCCOCC(C)(C)O",
        "CCCOCC(C)(O)O",
        "CCCOCC(C)C",
        "CCCOCC(C)O",
        "CCCOCC(O)O",
        "CCCOCCC",
        "CCCOCCO",
        "CCCOCO",
        "CCCOCOC",
        "CCO",
        "CCOC",
        "CCOC(C(C)(C)C)O",
        "CCOC(C(C)(C)O)O",
        "CCOC(C(C)(O)O)O",
        "CCOC(C(C)C)O",
        "CCOC(C(C)O)O",
        "CCOC(C(O)O)O",
        "CCOC(C)(C(C)(C)C)O",
        "CCOC(C)(C(C)(C)O)O",
        "CCOC(C)(C(C)(O)O)O",
        "CCOC(C)(C(C)C)O",
        "CCOC(C)(C(C)O)O",
        "CCOC(C)(C(O)O)O",
        "CCOC(C)(C)C",
        "CCOC(C)(C)C(C)(C)C",
        "CCOC(C)(C)C(C)(C)O",
        "CCOC(C)(C)C(C)(O)O",
        "CCOC(C)(C)C(C)C",
        "CCOC(C)(C)C(C)O",
        "CCOC(C)(C)C(O)O",
        "CCOC(C)(C)CC(C)(C)C",
        "CCOC(C)(C)CC(C)(C)O",
        "CCOC(C)(C)CC(C)(O)O",
        "CCOC(C)(C)CC(C)C",
        "CCOC(C)(C)CC(C)O",
        "CCOC(C)(C)CC(O)O",
        "CCOC(C)(C)CCO",
        "CCOC(C)(C)CO",
        "CCOC(C)(C)COC",
        "CCOC(C)(C)O",
        "CCOC(C)(C)OC",
        "CCOC(C)(C)OC(C)(C)C",
        "CCOC(C)(C)OC(C)(C)O",
        "CCOC(C)(C)OC(C)(O)O",
        "CCOC(C)(C)OC(C)C",
        "CCOC(C)(C)OC(C)O",
        "CCOC(C)(C)OC(O)O",
        "CCOC(C)(C)OCC",
        "CCOC(C)(C)OCO",
        "CCOC(C)(CC(C)(C)C)O",
        "CCOC(C)(CC(C)(C)O)O",
        "CCOC(C)(CC(C)(O)O)O",
        "CCOC(C)(CC(C)C)O",
        "CCOC(C)(CC(C)O)O",
        "CCOC(C)(CC(O)O)O",
        "CCOC(C)(CCO)O",
        "CCOC(C)(CO)O",
        "CCOC(C)(COC)O",
        "CCOC(C)(O)O",
        "CCOC(C)(O)OC",
        "CCOC(C)(O)OC(C)(C)C",
        "CCOC(C)(O)OC(C)(C)O",
        "CCOC(C)(O)OC(C)(O)O",
        "CCOC(C)(O)OC(C)C",
        "CCOC(C)(O)OC(C)O",
        "CCOC(C)(O)OC(O)O",
        "CCOC(C)(O)OCC",
        "CCOC(C)(O)OCO",
        "CCOC(C)C",
        "CCOC(C)C(C)(C)C",
        "CCOC(C)C(C)(C)O",
        "CCOC(C)C(C)(O)O",
        "CCOC(C)C(C)C",
        "CCOC(C)C(C)O",
        "CCOC(C)C(O)O",
        "CCOC(C)CC(C)(C)C",
        "CCOC(C)CC(C)(C)O",
        "CCOC(C)CC(C)(O)O",
        "CCOC(C)CC(C)C",
        "CCOC(C)CC(C)O",
        "CCOC(C)CC(O)O",
        "CCOC(C)CCO",
        "CCOC(C)CO",
        "CCOC(C)COC",
        "CCOC(C)O",
        "CCOC(C)OC",
        "CCOC(C)OC(C)(C)C",
        "CCOC(C)OC(C)(C)O",
        "CCOC(C)OC(C)(O)O",
        "CCOC(C)OC(C)C",
        "CCOC(C)OC(C)O",
        "CCOC(C)OC(O)O",
        "CCOC(C)OCC",
        "CCOC(C)OCO",
        "CCOC(CC(C)(C)C)O",
        "CCOC(CC(C)(C)O)O",
        "CCOC(CC(C)(O)O)O",
        "CCOC(CC(C)C)O",
        "CCOC(CC(C)O)O",
        "CCOC(CC(O)O)O",
        "CCOC(CCO)O",
        "CCOC(CO)O",
        "CCOC(COC)O",
        "CCOC(O)O",
        "CCOC(O)OC",
        "CCOC(O)OC(C)(C)C",
        "CCOC(O)OC(C)(C)O",
        "CCOC(O)OC(C)(O)O",
        "CCOC(O)OC(C)C",
        "CCOC(O)OC(C)O",
        "CCOC(O)OC(O)O",
        "CCOC(O)OCC",
        "CCOC(O)OCO",
        "CCOCC",
        "CCOCC(C)(C)C",
        "CCOCC(C)(C)O",
        "CCOCC(C)(O)O",
        "CCOCC(C)C",
        "CCOCC(C)O",
        "CCOCC(O)O",
        "CCOCCC(C)(C)C",
        "CCOCCC(C)(C)O",
        "CCOCCC(C)(O)O",
        "CCOCCC(C)C",
        "CCOCCC(C)O",
        "CCOCCC(O)O",
        "CCOCCCO",
        "CCOCCO",
        "CCOCCOC",
        "CCOCO",
        "CCOCOC",
        "CO",
        "COC",
        "COC(CCO)O",
        "COC(CO)O",
        "COC(O)O",
        "COC(O)OCO",
        "COCC(CCO)O",
        "COCC(CO)O",
        "COCC(O)O",
        "COCC(O)OCO",
        "COCCCCO",
        "COCCCO",
        "COCCO",
        "COCCOCO",
        "COCO",
        "COCOC",
        "COCOCCO",
        "COCOCO"
    ]

    # smiles = [
    #     "CCCCCOCCCC",
    #     "CCOCCCCOCCC",
    #     "CCCC",
    #     "CCOCC(CCN)CC",
    #     "CCOCC"
    # ]

    random.shuffle(smiles_train)

    train_charges(
        smiles_train[:num_train_samples]
    )