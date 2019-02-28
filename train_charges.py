import time
import numpy as np
import argparse
import unittest
import ctypes
import random

import tensorflow as tf

# tf.enable_eager_execution()


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

    nrgs, total_params, offsets = system_builder.construct_energies(ff, mol, am1_charges)

    # dt = 0.0025
    # friction = 10.0
    # temperature = 300

    # gradient descent
    dt = dt
    friction = 10.0
    temperature = temperature

    a,b,c = get_abc_coefficents(masses, dt, friction, temperature)

    buf_size = estimate_buffer_size(1e-10, a)

    print("BUFFER SIZE", buf_size)


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

    return nrgs, offsets, intg, context, x0, total_params

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

def test_mol(smiles):

    nrgs1, offsets1, intg1, context1, init_x_1, total_params_1 = initialize_system(
        smiles=smiles,
        dt=0.001,
        temperature=100,
        forcefield_file='forcefield/smirnoff99Frosst.offxml',
        am1_charges=True
    )


    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session()

    fh = open("charge_training_"+str(smiles)+".log", "w")

    # generate the observable
    print("generating observable", file=fh)
    ksize = 1000
    confs1, _ = run_once(nrgs1, context1, intg1, init_x_1, 80000, total_params_1, ksize, inference=True) 

    for nrg in nrgs1:
        print("Reference parameters:", nrg, nrg.get_params())

    x1 = tf.convert_to_tensor(confs1)
    obs1_rij = observable.sorted_squared_distances(x1)

    intg1.reset()

    confs2, _ = run_once(nrgs1, context1, intg1, init_x_1, 10000, total_params_1, ksize, inference=True) 
    x2 = tf.convert_to_tensor(confs2)
    obs2_rij = observable.sorted_squared_distances(x2)

    mutual_loss = tf.sqrt(tf.reduce_sum(tf.pow(obs2_rij - obs1_rij, 2))/ksize) # RMSE

    print("Mutual loss", sess.run(mutual_loss))

    # assert 0

    # train this secondary system
    print("starting training...", file=fh)
    bond_learning_rate = np.array([[0.1, 0.0001]])
    angle_learning_rate = np.array([[0.01, 0.001]])
    torsion_learning_rate = np.array([[0.01, 0.001, 0.0]])
    lj_learning_rate = np.array([[0.000, 0.000]])
    es_learning_rate = np.array([[0.001]])

    nrgs0, offsets0, intg0, context0, init_x_0, total_params_0 = initialize_system(
        smiles=smiles,
        dt=0.001,
        temperature=100,
        forcefield_file='forcefield/smirnoff99Frosst_perturbed.offxml',
        am1_charges=False
    )

    for epoch in range(1000): 

        print("starting epoch", epoch)
        confs0, dxdp0 = run_once(nrgs0, context0, intg0, init_x_0, 10000, total_params_0, ksize, inference=False)
        x0 = tf.convert_to_tensor(confs0)
        obs0_rij = observable.sorted_squared_distances(x0)
        loss = tf.sqrt(tf.reduce_sum(tf.pow(obs0_rij - obs1_rij, 2))/ksize) # RMSE
        x0_grads = tf.gradients(loss, x0)[0]

        np_loss, x0g = sess.run([loss, x0_grads])

        print("------------------LOSS", np_loss)
        x0g = np.expand_dims(x0g, 1) # [B, 1, N, 3]
        res = np.multiply(x0g, dxdp0) # dL/dx * dx/dp [B, P, N, 3]

        dLdp = np.sum(res, axis=(0,2,3))

        for dparams, nrg in zip(np.split(dLdp, offsets0)[1:], nrgs0):
            if isinstance(nrg, custom_ops.HarmonicBondGPU_double):
                cp = nrg.get_params()
                dp = bond_learning_rate * dparams.reshape((-1, 2))
                print("BOND PARAMS", cp)
                print("BOND CONSTANTS, LENGTHS", dp)
                # nrg.set_params(cp - dp.reshape(-1))
            elif isinstance(nrg, custom_ops.HarmonicAngleGPU_double):
                dp = angle_learning_rate * dparams.reshape((-1, 2))
                print("ANGLE CONSTANTS, ANGLES", dp)
            elif isinstance(nrg, custom_ops.PeriodicTorsionGPU_double):
                dp = torsion_learning_rate * dparams.reshape((-1, 3))
                print("TORSION CONSTANTS, PERIODS, PHASES", dp)
            elif isinstance(nrg, custom_ops.LennardJonesGPU_double):
                dp = lj_learning_rate * dparams.reshape((-1, 2))
                print("LJ SIG, EPS", dp)
            elif isinstance(nrg, custom_ops.ElectrostaticsGPU_double):
                dp = es_learning_rate * dparams.reshape((-1, 1))
                cp = nrg.get_params()
                print("ES PARAMS", cp)
                print("ES", dp)
                nrg.set_params(cp - dp.reshape(-1))
            else:
                assert 0

        fh.flush()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Stability testing.')
    parser.add_argument('--smiles', dest='smiles', help='what temperature we should run at')
    args = parser.parse_args()
    test_mol(args.smiles)