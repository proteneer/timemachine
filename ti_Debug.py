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
from timemachine.potentials import restraints

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

##### 
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

    print("running lambda", lamb)

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
        friction=91*2, # (ytz) probably need to double this?
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

    # dynamics loop
    if lamb == 0.0:
        max_iter = 15000
    elif lamb < 0.4:
        max_iter = 500000*2
    else:
        max_iter = 50000
    dt = 1e-3
    
    md_dudls = []
    
    # v_t = np.zeros((num_atoms, 4))
    # this would zero out the velocity derivatives dv_dp
    # not sure if this is necessary
    # v_t[:, :3] = set_velocities_to_temperature(num_atoms, 300, masses) 

    # ctxt_md = custom_ops.Context_f64(
    #     potentials,
    #     opt_md,
    #     params.astype(np.float64),
    #     ctxt.get_x(),
    #     v_t.astype(np.float64),
    #     dp_idxs.astype(np.int32)
    # )

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
        if i % 10000 == 0 and i >= 30000:
            
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

            print(lamb, "\t", i, "\t", E, "\t", dUdL, "\t", "| dxdp max/min", np.amax(dxdp), "\t", np.amin(dxdp), "| mean deriv: ", np.mean(all_d2u_dldps, axis=0), "\t dudl: ", np.mean(all_dudls), "+-", np.std(all_dudls), "\t @ ", speed, "ns/day")
            xyz = write(np.asarray(xi[:, :3]*10), masses)
            xyz_buffer.append(xyz)


            # jb = "".join(xyz_buffer)
            # open("animation.xyz", "w").write(jb)

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
            
    print("Lambda", lamb, "energy:", E, 'dUdl_min', dUdL_min, 'mean_dudl', np.mean(md_dudls))

    return np.mean(md_dudls)
    # return x_t, xyz_buffer, E, np.mean(md_dudls)

def rescale_and_center(conf, scale_factor=1):
    mol_com = np.sum(conf, axis=0)/conf.shape[0]
    true_com = np.array([1.97698696, 1.90113478, 2.26042174]) # a-cd
    centered = conf - mol_com  # centered to origin
    return true_com + centered/scale_factor 

def run_simulation(params):
    mol, lamb, idx = params
    p = multiprocessing.current_process()

    # os.environ['CUDA_VISIBLE_DEVICES'] = " "

    filepath = 'examples/host_acd.xml'
    filename, file_extension = os.path.splitext(filepath)
    sys_xml = open(filepath, 'r').read()
    system = mm.XmlSerializer.deserialize(sys_xml)
    coords = np.loadtxt(filename + '.xyz').astype(np.float64)
    coords = coords/10
    
    host_potentials, host_conf, (host_params, host_param_groups), host_masses = serialize.deserialize_system(system, coords)
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    guest_potentials, smirnoff_params, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)
    guest_conf = rescale_and_center(guest_conf)

    combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
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

    host_dp_idxs = np.argwhere(filter_groups(host_param_groups, [9])).reshape(-1)
    guest_dp_idxs = np.argwhere(filter_groups(smirnoff_param_groups, [9])).reshape(-1)
    combined_dp_idxs = np .argwhere(filter_groups(combined_param_groups, [9])).reshape(-1)

    print("Number of parameter derivatives", combined_dp_idxs.shape)

    # lambda_schedule = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5]
    lambda_schedule = [0.25]

    for lambda_idx, lamb in enumerate(lambda_schedule):
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

        fname = "test_du_dl_grads_lambda_everything_minima"+str(lambda_idx)
        print("Saving")
        np.savez(fname, lamb=lamb, du_dls=du_dls, du_dl_grads=du_dl_grads)

    assert 0


    # host_dp_idxs = np.array([1])
    # guest_dp_idxs = np.array([1])
    # combined_dp_idxs = np.array([1])

    # integrate_me = functools.partial(minimize,
    #     num_host_atoms,
    #     combined_potentials,
    #     combined_params,
    #     combined_param_groups,
    #     combined_conf,
    #     combined_masses,
    #     combined_dp_idxs,
    #     1000,
    #     None,
    #     4
    # )

    # res = scipy.integrate.quad(integrate_me, 0, 0.5, epsabs=1, epsrel=1.49e-03)



    # buf = minimize(
    #     num_host_atoms,
    #     combined_potentials,
    #     combined_params,
    #     combined_param_groups,
    #     combined_conf,
    #     combined_masses,
    #     combined_dp_idxs,
    #     1000,
    #     None,
    #     starting_dimension=4,
    #     lamb=lamb
    # )

    # return buf, num_host_atoms

replicas = 10

for x in range(15):
    print("STARTING NEW SET", x)
    complete_dudls = []
    all_dGs = []

    for rr in range(1, replicas):

        fname = "/home/yutong/Code/benchmarksets/input_files/cd-set1/mol2/guest-"+str(rr)+".mol2"
        guest_mol2 = open(fname, "r").read()
        # guest_mol2 = Chem.MolFromSmiles("O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)[C@H](O)[C@@H]3O")
        # mol = Chem.AddHs(guest_mol2)
        mol = Chem.MolFromMol2Block(guest_mol2, sanitize=True, removeHs=False, cleanupSubstructures=True)

        def pmf(nrgs, dudls, temperature=300):
            kT = constants.BOLTZ * temperature
            weights = scipy.special.softmax(-np.array(nrgs)/kT)
            print(weights)
            return np.sum(weights*dudls)/np.sum(weights)

        batch_size = 4
        pool = multiprocessing.Pool(batch_size)
        fh = open("4d_ti_results.txt", "w")

        free_nrgs = []

        def exp_weight(dgs):
            T = 300
            kT = constants.BOLTZ*T # in kJ
            dgs = np.asarray(dgs)
            return -kT*np.log(np.sum(np.exp(-dgs/kT)))

        AllChem.EmbedMolecule(mol, randomSeed=1337)
        # AllChem.EmbedMolecule(mol)
        conf = mol.GetConformer(0)
        coords = conf.GetPositions()
        # rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
        # coords = np.matmul(coords, rot_matrix)
        # for idx, (x,y,z) in enumerate(coords):
            # conf.SetAtomPosition(idx, (x,y,z))

        all_params = []
        all_lambdas = []
        for lamb_idx, lamb in enumerate(range(20, 100)):
            dlambda = float(lamb)/100
            all_lambdas.append(dlambda)
            # for r in range(batch_size):
        #     new_mol = Chem.Mol(mol)
        #     AllChem.EmbedMolecule(new_mol)
            params = (mol, dlambda, lamb_idx)
            all_params.append(params)


        run_simulation(all_params[0])
        # assert 0
    #     continue
        # results = pool.map(run_simulation, all_params)

        batch_dudls = []
        batch_nrgs = []

        for (_, _, energy, dudl), nha in results:
            batch_dudls.append(dudl)
            batch_nrgs.append(energy)

        dx = all_lambdas[1] - all_lambdas[0]    
        dG = np.trapz(batch_dudls, all_lambdas)
        free_nrgs.append(dG)
        print("free energy", dG)
        complete_dudls.append(batch_dudls)
        for dd in complete_dudls:
            plt.plot(all_lambdas, dd)
        plt.show()

    #     print("mean/std", np.mean(free_nrgs), np.std(free_nrgs))
    #     fh.write("mean/std: " + str(np.mean(free_nrgs)) + " " + str(np.std(free_nrgs)) + "\n")
    #     fh.flush()

        all_dGs.append(dG)
        print("dgs", np.array(all_dGs)/4.18)
        print("replica", rr, exp_weight(np.array(all_dGs))/4.18, "kcal/mol")

pool.close()