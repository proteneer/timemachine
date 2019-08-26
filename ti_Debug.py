import numpy as np
import py3Dmol
import jax
import jax.numpy as jnp
import functools
import os
import sys
import time

from rdkit.Chem import AllChem
from system import serialize
from system import forcefield
from system import simulation
import scipy.integrate
from rdkit import Chem
from openforcefield.typing.engines.smirnoff import ForceField
from simtk import openmm as mm
from simtk.openmm import app

from scipy.stats import special_ortho_group
# from timemachine.lib import custom_ops
from jax.experimental import optimizers


import jax.numpy as jnp
import numpy as np
import random

# from system import forcefield
from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients

from timemachine import constants

from simtk.openmm.app import PDBFile

import scipy

from simtk import openmm as mm
from simtk.openmm import app
from simtk.openmm.app import PDBFile
from simtk.openmm.app import forcefield as ff
from simtk import unit

import py3Dmol
import sklearn.decomposition

from matplotlib import pyplot as plt
# %matplotlib inline
import multiprocessing

plt.rcParams['figure.dpi'] = 200

num_gpus = 8

def value(quantity):
    return quantity.value_in_unit_system(unit.md_unit_system)

def write(xyz, masses):

    xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
    buf = str(len(masses)) + '\n'
    buf += 'timemachine\n'
    for m, (x,y,z) in zip(masses, xyz):
        if int(round(m)) == 12:
            symbol = 'C'
        elif int(round(m)) == 14:
            symbol = 'N'
        elif int(round(m)) == 16:
            symbol = 'O'
        elif int(round(m)) == 32:
            symbol = 'S'
        elif int(round(m)) == 35:
            symbol = 'Cl'
        elif int(round(m)) == 1:
            symbol = 'H'
        elif int(round(m)) == 31:
            symbol = 'P'
        elif int(round(m)) == 19:
            symbol = 'F'
        else:
            raise Exception("Unknown mass:" + str(m))

        buf += symbol + ' ' + str(round(x,5)) + ' ' + str(round(y,5)) + ' ' +str(round(z,5)) + '\n'
    return buf

def set_velocities_to_temperature(n_atoms, temperature, masses):
    assert 0 # don't call this yet
    v_t = np.random.normal(size=(n_atoms, 3))
    velocity_scale = np.sqrt(constants.BOLTZ*temperature/np.expand_dims(masses, -1))
    return v_t*velocity_scale

class ReferenceLangevin():

    def __init__(self, ca, cb, cc):
        self.coeff_a = ca
        self.coeff_bs = cb
        self.coeff_cs = cc

    def step(self, dt, x_t, v_t, dE_dx):
        noise = np.random.normal(size=(x_t.shape[0], x_t.shape[1]))
        v_t_1 = self.coeff_a*v_t - np.expand_dims(self.coeff_bs, axis=-1)*dE_dx + np.expand_dims(self.coeff_cs, axis=-1)*noise
        x_t_1 = x_t + v_t_1*dt
        final_X = jnp.concatenate([x_t_1[:, :3], x_t[:, 3:]], axis=1)
        final_V = jnp.concatenate([v_t_1[:, :3], v_t[:, 3:]], axis=1)
        return final_X, final_V

@jax.jit
def com_motion_remover(v_t, masses):
    com = jnp.sum(v_t * jnp.expand_dims(masses, -1), axis=0)
    momentum = com / jnp.sum(masses)
    return v_t - jnp.expand_dims(momentum, axis=0)


def compute_d2u_dldp(energies, params, xs, dx_dps, dp_idxs, num_host_atoms):

    assert len(xs.shape) == 2
    assert len(dx_dps.shape) == 3

    mixed_partials = []
    hessians = []
    # we need to compute this separately since the context's sgemm call overwrites
    # the values of d2u_dxdp
    # batched call
    for p in energies:
        _, _, ph, _, pmp  = p.derivatives(np.expand_dims(xs, axis=0), params, dp_idxs)
        mixed_partials.append(pmp)
        hessians.append(ph)
    
    hessians = np.sum(hessians, axis=0)[0]

    # print(np.triu(hessi))

    mixed_part = np.sum(mixed_partials, axis=0)[0]

    hess_idxs = jax.ops.index[num_host_atoms:, 3:, :, :3]
    dx_dp_idxs = jax.ops.index[:, :, :3]
    mp_idxs = jax.ops.index[:, num_host_atoms:, 3:]
    lhs = np.einsum('ijkl,mkl->mij', hessians[hess_idxs], dx_dps[dx_dp_idxs]) # correct only up to main hessian
    rhs = mixed_part[mp_idxs]
    # lhs + rhs has shape [P, num_atoms-num_host_atoms, 1] 
    d2u_dldp = np.sum(lhs+rhs, axis=(1,2)) # P N 4 -> P
    return d2u_dldp

def minimize(
    num_host_atoms,
    potentials,
    params,
    param_groups,
    conf,
    masses,
    dp_idxs,
    n_samples,
    pdb,
    starting_dimension,
    lamb):

    # print("running lambda", lamb)

    num_atoms = len(masses)
    num_guest_atoms = num_atoms - num_host_atoms
    
    potentials = forcefield.merge_potentials(potentials)

    def dU_dlambda(dE_dx):
        # this is only correct if it's sum, don't be tempted by the mean
        return np.sum(dE_dx[num_host_atoms:, 3:]) 

    dt = 1e-3
    ca, cb, cc = langevin_coefficients(
        temperature=300,
        # temperature=0,
        dt=dt,
        friction=91, # (ytz) probably need to double this?
        # friction=100, # (ytz) probably need to double this?
        masses=masses
    )

    m_dt, m_ca, m_cb, m_cc = dt, 0.5, np.ones_like(cb)/10000, np.zeros_like(masses)

    np.random.seed()

    opt = custom_ops.LangevinOptimizer_f64(
        dt,
        4,
        m_ca,
        m_cb.astype(np.float64),
        m_cc.astype(np.float64)
    )

    dp_idxs = dp_idxs.astype(np.int32)

    # tolerance = 1

    # def mean_norm(conf):
    #     norm_x = np.dot(conf.reshape(-1), conf.reshape(-1))/num_atoms
    #     if norm_x < 1:
    #         norm_x = 1
    #         # raise ValueError("Starting norm is less than one")
    #     return np.sqrt(norm_x)

    # epsilon = tolerance/mean_norm(conf)    
    count = 0
    max_iter = 15000

    num_atoms = conf.shape[0]
    num_dimensions = starting_dimension
    
    d4_t = np.zeros((num_atoms, num_dimensions), dtype=np.float64)
    d4_t_lambdas = np.zeros((num_atoms, num_dimensions), dtype=np.float64) + lamb

    # set coordinates
    d4_t[:num_host_atoms, :3] = conf[:num_host_atoms, :3]
    d4_t[num_host_atoms:, :3] = conf[num_host_atoms:, :3]
    d4_t[num_host_atoms:, 3:] = d4_t_lambdas[num_host_atoms:, 3:]

    x_t = d4_t
    v_t = np.zeros_like(x_t)

    # print("starting x_t", x_t)

    cur_dim = num_dimensions

    ctxt = custom_ops.Context_f64(
        potentials,
        opt,
        params.astype(np.float64),
        x_t.astype(np.float64),
        v_t.astype(np.float64), # n
        dp_idxs.astype(np.int32)
    )

    xyz_buffer = []
    dt = 1e-7
    for i in range(max_iter):
        dt *= 1.003
        dt = min(dt, 0.02)

        # dE_dx = grad_fn(x_t, params)[0]
        opt.set_dt(dt)
        # x_t, v_t = ctxt.step(x_t, v_t, dE_dx)
        ctxt.step()

        if i % 500 == 0:
            E = ctxt.get_E()
            xi = ctxt.get_x()
            dE_dx = ctxt.get_dE_dx()
            dUdL = dU_dlambda(dE_dx)

            # xyz = write(np.asarray(xi[:, :3]*10), masses)
            # xyz_buffer.append(xyz)

        cur_dim -= 1

    # sys.exit(0)

    # assert 0
    print("Lambda", lamb, ": minimized in ", i, "steps to", E, 'dU_dl', dUdL, 'dx_dp', np.amin(ctxt.get_dx_dp()), np.amax(ctxt.get_dx_dp()))

    # dynamics loop production
    cutoff = 50000
    sampling_interval = 10000
    if lamb == 0.0:
        max_iter = 100000
    elif lamb < 0.4:
        max_iter = 5000000*2
    else:
        max_iter = 100000

    # testing cycle
    cutoff = 100000
    sampling_interval = 2000
    if lamb == 0.0:
        max_iter = 120000
    elif lamb < 0.4:
        max_iter = 1000000*2
    else:
        max_iter = 120000

    dt = 1e-3
    
    md_dudls = []

    # swap
    opt.set_dt(dt)
    opt.set_coeff_a(np.float64(ca))
    opt.set_coeff_b(cb.astype(np.float64))
    opt.set_coeff_c(cc.astype(np.float64))

    def compute_ke(vt):
        return np.sum(np.expand_dims(masses, axis=-1)*vt*vt)/2

    def ns_per_day(delta, timestep_in_fs):
        return (timestep_in_fs/delta)*(86400)*1e-6

    all_dudls = []
    all_d2u_dldps = []

    # all_dxdps = []
    # all_d2u_dldps = []
    all_kes = []

    start = time.time()
    for i in range(max_iter):
        ctxt.step()
        if i % sampling_interval == 0 and i >= cutoff:
            
            E = ctxt.get_E()
            xi = ctxt.get_x()
            dxi_dp = ctxt.get_dx_dp()
            all_d2u_dldps.append(compute_d2u_dldp(
                potentials,
                params.astype(np.float64),
                xi,
                dxi_dp,
                dp_idxs,
                num_host_atoms)
            )

            dE_dx = ctxt.get_dE_dx()
            dUdL = dU_dlambda(dE_dx)
            all_dudls.append(dUdL)

            dxdp = ctxt.get_dx_dp()
            ke = compute_ke(ctxt.get_v())
            all_kes.append(ke)
            speed = ns_per_day(time.time()-start, i)

            print(f"{lamb} \t {i} \t {E:9.4f} \t {dUdL:9.4f} \t | dxdp max/min {np.amax(dxdp):9.4f} \t {np.amin(dxdp):9.4f} \t max mean/median deriv: {np.amax(np.mean(all_d2u_dldps, axis=0)):9.4f} \t {np.amax(np.median(all_d2u_dldps, axis=0)):9.4f} \t mean/median dudl {np.mean(all_dudls):9.4f} \t {np.median(all_dudls):9.4f} \t @ {speed:9.4f} ns/day")
            # print(lamb, "\t", i, "\t", E, "\t", dUdL, "\t", "| dxdp max/min", np.amax(dxdp), "\t", np.amin(dxdp), "\t | max mean/median deriv: ", np.amax(np.mean(all_d2u_dldps, axis=0)), "\t", np.amax(np.median(all_d2u_dldps, axis=0)), "\t mean/median dudl: ", np.mean(all_dudls), "\t", np.median(all_dudls), "+-", np.std(all_dudls), "\t @ ", speed, "ns/day")
#            if np.amax(dxdp) > 100:
#                raise ValueError("DXDP IS TOO LARGE")
            xyz = write(np.asarray(xi[:, :3]*10), masses)
            xyz_buffer.append(xyz)


    # print("FINAL", lamb, "\t", i, "\t", E, "\t", dUdL, "\t", np.mean(all_dudls), "+-", np.std(all_dudls))

    return all_dudls, all_d2u_dldps
    # return np.mean(all_dudls), np.mean(all_d2u_dldps, axis=0)
    # assert 0)
#     view = py3Dmol.view(width=600, height=600).addModelsAsFrames(jb, 'xyz')
#     view.animate({"loop": "forward","reps":15});
#     view.setStyle({'stick':{}})
#     view.zoomTo()
#     view.show()

#     assert 0
            
def rescale_and_center(conf, scale_factor=1):
    mol_com = np.sum(conf, axis=0)/conf.shape[0]
    true_com = np.array([1.97698696, 1.90113478, 2.26042174]) # a-cd
    centered = conf - mol_com  # centered to origin
    return true_com + centered/scale_factor 


def initialize_parameters(host_path=None):
    '''
    Initializes parameters for training.
    
    host_path (string): path to host if training binding energies (default = None)
    
    returns: (combined host and ligand parameters, smirnoff ligand parameters)
    '''
    # setting general smirnoff parameters for guest using random smiles string
    ref_mol = Chem.MolFromSmiles('CCCC')
    ref_mol = Chem.AddHs(ref_mol)
    AllChem.EmbedMolecule(ref_mol)

    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    # smirnoff params are all encompassing for small molecules
    _, smirnoff_params, _, _, _ = forcefield.parameterize(ref_mol, smirnoff)
    sys_xml = open(host_path, 'r').read()
    system = mm.XmlSerializer.deserialize(sys_xml)
    _, (host_params, _), _ = serialize.deserialize_system(system)       
    epoch_combined_params = np.concatenate([host_params, smirnoff_params])

    return epoch_combined_params


def run_simulation(params):
    mol, lamb, lambda_idx, combined_params = params
    p = multiprocessing.current_process()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(lambda_idx % num_gpus)

    filepath = 'examples/host_acd.xml'
    filename, file_extension = os.path.splitext(filepath)
    sys_xml = open(filepath, 'r').read()
    system = mm.XmlSerializer.deserialize(sys_xml)
    coords = np.loadtxt(filename + '.xyz').astype(np.float64)
    coords = coords/10
    
    host_conf = coords

    host_potentials, (host_params, host_param_groups), host_masses = serialize.deserialize_system(system)
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    guest_potentials, smirnoff_params, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)
    guest_conf = rescale_and_center(guest_conf)

    combined_potentials, _, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
        host_potentials, guest_potentials,
        host_params, smirnoff_params,
        host_param_groups, smirnoff_param_groups,
        host_conf, guest_conf,
        host_masses, guest_masses)

    num_host_atoms = host_conf.shape[0]

    def filter_groups(param_groups, groups):
        roll = np.zeros_like(param_groups)
        for g in groups:
            roll = np.logical_or(roll, param_groups == g)
        return roll

    # host_dp_idxs = np.argwhere(filter_groups(host_param_groups, [7])).reshape(-1)
    # guest_dp_idxs = np.argwhere(filter_groups(smirnoff_param_groups, [7])).reshape(-1)
    combined_dp_idxs = np.argwhere(filter_groups(combined_param_groups, [7])).reshape(-1)

    # print("combined_dp_idxs", combined_dp_idxs)

    # print("Number of parameter derivatives", combined_dp_idxs.shape)

    du_dls, du_dl_grads = minimize(
        num_host_atoms,
        combined_potentials,
        combined_params,
        combined_param_groups,
        combined_conf,
        combined_masses,
        combined_dp_idxs,
        1000,
        None,
        starting_dimension=4,
        lamb=lamb
    )

    # fname = "test_du_dl_grads_lambda_low_temp_charges"+str(lambda_idx)
    # print("Saving")
    # np.savez(fname, lamb=lamb, du_dls=du_dls, du_dl_grads=du_dl_grads)

    return lamb, du_dls, du_dl_grads, combined_dp_idxs

def train(true_dG):
    fname = "/home/ubuntu/Relay/Code/benchmarksets/input_files/cd-set1/mol2/guest-"+str(1)+".mol2"
    guest_mol2 = open(fname, "r").read()
    # guest_mol2 = Chem.MolFromSmiles("O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)[C@H](O)[C@@H]3O")
    # mol = Chem.AddHs(guest_mol2)
    mol = Chem.MolFromMol2Block(guest_mol2, sanitize=True, removeHs=False, cleanupSubstructures=True)

    pool = multiprocessing.Pool(num_gpus)

    AllChem.EmbedMolecule(mol, randomSeed=1337)
    # AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer(0)
    coords = conf.GetPositions()
    # rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
    # coords = np.matmul(coords, rot_matrix)
    # for idx, (x,y,z) in enumerate(coords):
        # conf.SetAtomPosition(idx, (x,y,z))

    starting_params = initialize_parameters('examples/host_acd.xml')
    lr=3e-3
    opt_init, opt_update, get_params = optimizers.adam(lr)
    # opt_init, opt_update, get_params = optimizers.sgd(lr)

    opt_state = opt_init(starting_params)

    num_epochs = 50
    for epoch in range(num_epochs):

        print("===============Epoch "+str(epoch)+"=============")

        lr = 1e-4
        all_params = []
        all_lambdas = []
        lambda_schedule = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5]
        #lambda_schedule = [0.0, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 0.9, 1.0, 1.5]
        # lambda_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        epoch_params = get_params(opt_state)

        for lamb_idx, lamb in enumerate(lambda_schedule):
            params = (mol, lamb, lamb_idx, epoch_params)
            all_params.append(params)

        results = pool.map(run_simulation, all_params)

        all_lambdas = []
        all_mean_du_dls = []
        all_median_du_dls = []
        all_mean_du_dl_grads = []
        all_median_du_dl_grads = []
        for lamb, du_dls, du_dl_grads, combined_dp_idxs in results:
            all_lambdas.append(lamb)
            all_mean_du_dls.append(np.mean(du_dls))
            all_median_du_dls.append(np.median(du_dls))
            all_mean_du_dl_grads.append(np.mean(du_dl_grads, axis=0))
            all_median_du_dl_grads.append(np.median(du_dl_grads, axis=0))

        all_lambdas = np.array(all_lambdas)
        all_mean_du_dls = np.array(all_mean_du_dls)
        all_median_du_dls = np.array(all_median_du_dls)
        all_mean_du_dl_grads = np.array(all_mean_du_dl_grads)
        all_median_du_dl_grads = np.array(all_median_du_dl_grads)

        num_params = all_mean_du_dl_grads.shape[-1]

        np.set_printoptions(linewidth=np.inf)
    
        print("MEAN DU_DL", all_mean_du_dls)
        print("MEDIAN DU_DL", all_median_du_dls)

        for p_idx in range(num_params):
            print("MEAN DU_DL_GRAD_"+str(p_idx), all_mean_du_dl_grads[:, p_idx])

        for p_idx in range(num_params):
            print("MEDIAN DU_DL_GRAD_"+str(p_idx), all_median_du_dl_grads[:, p_idx])

        pred_dG = np.trapz(all_lambdas, all_mean_du_dls)
        pred_dG_median = np.trapz(all_lambdas, all_median_du_dls)

        dG_grads = []
        dG_grads_median = []
        for p_idx in range(num_params):
            dG_grads.append(np.trapz(all_lambdas, all_mean_du_dl_grads[:, p_idx]))
            dG_grads_median.append(np.trapz(all_lambdas, all_median_du_dl_grads[:, p_idx]))
            # dG_grads.append(grad)
            # print("MEAN DU_DL_GRAD_"+str(p_idx), grad)

        dG_grads = np.array(dG_grads)
        dG_grads_median = np.array(dG_grads_median)

        dparams = np.zeros_like(epoch_params)
        dparams[combined_dp_idxs] = dG_grads
        # dparams[combined_dp_idxs] = dG_grads_median

        L2_loss = (pred_dG-true_dG)**2
        L1_loss = np.abs(pred_dG-true_dG)
        full_L2_grad = 2*(pred_dG-true_dG)*dparams

        print("L1_loss", L1_loss, "true vs pred", true_dG, pred_dG, pred_dG_median)

        # fix me when going to multiple molecules
        # print("full_L2_grad", full_L2_grad)
        opt_state = opt_update(epoch, full_L2_grad, opt_state)

    pool.close()

train(1.575*4.18)
