import jax
import jax.numpy as jnp
import numpy as np
import functools
import os
import sys
import time

import simtk.unit

from rdkit import Chem
from rdkit.Chem import AllChem
from system import serialize
from system import forcefield
from system import simulation

from openforcefield.typing.engines.smirnoff import ForceField
from simtk import openmm as mm
from simtk.openmm import app
from simtk.openmm.app import forcefield as ff

from scipy.stats import special_ortho_group
from jax.experimental import optimizers

import jax.numpy as jnp
import random

# from system import forcefield
from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients
from timemachine import constants

import multiprocessing

from matplotlib import pyplot as plt
import py3Dmol

plt.rcParams['figure.dpi'] = 200

num_gpus = 1

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
        elif int(round(m)) == 80:
            symbol = 'Br'
        else:
            raise Exception("Unknown mass:" + str(m))

        buf += symbol + ' ' + str(round(x,5)) + ' ' + str(round(y,5)) + ' ' +str(round(z,5)) + '\n'
    return buf

def set_velocities_to_temperature(n_atoms, temperature, masses):
    assert 0 # don't call this yet
    v_t = np.random.normal(size=(n_atoms, 3))
    velocity_scale = np.sqrt(constants.BOLTZ*temperature/np.expand_dims(masses, -1))
    return v_t*velocity_scale

def compute_d2e_dxdp(energies, params, xs, dp_idxs):
    mixed_partials = []
    # hessians = []
    for p in energies:
        _, _, ph, _, pmp  = p.derivatives(np.expand_dims(xs, axis=0), params, dp_idxs)
        mixed_partials.append(pmp)

        # print("max/min hessians of",p,'\t',np.amax(ph),'\t',np.amin(ph))
        # hessians.append(ph)

    mixed_part = np.sum(mixed_partials, axis=0)[0]
    # hessians = np.sum(mixed_partials, axis=0)[0]
    return mixed_part


def to_md_units(q):
    return q.value_in_unit_system(simtk.unit.md_unit_system)


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
    lamb,
    lamb_idx):

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
        friction=91*6, # (ytz) probably need to double this?
        # friction=100, # (ytz) probably need to double this?
        masses=masses
    )

    # print("LGV CF", ca, cb, cc)

    m_dt, m_ca, m_cb, m_cc = dt, 0.5, np.ones_like(cb)/10000, np.zeros_like(masses)
    m_dt, m_ca, m_cb, m_cc = dt, 0.0, np.ones_like(cb)/10000, np.zeros_like(masses)

    np.random.seed()

    opt = custom_ops.LangevinOptimizer_f64(
        dt,
        4,
        m_ca,
        m_cb.astype(np.float64),
        m_cc.astype(np.float64)
    )

    dp_idxs = dp_idxs.astype(np.int32)

    count = 0
    max_iter = 25000 # of steps for minimization

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
    dt = 1e-10
    fh = open("water_test"+str(lamb_idx)+".xyz", "w")
    xyz = write(np.asarray(x_t[:, :3]*10), masses)
    fh.write(xyz)
    # assert 0

    for i in range(max_iter):
        dt *= 1.0013
        dt = min(dt, 0.001)

        # dE_dx = grad_fn(x_t, params)[0]
        opt.set_dt(dt)
        # x_t, v_t = ctxt.step(x_t, v_t, dE_dx)
        ctxt.step()

        # print(dt, ctxt.get_E())    
        # print(i, ctxt.get_E())
        # if i % 50 == 0:
            # xi = ctxt.get_x()


        if i % 500 == 0:
            E = ctxt.get_E()
            print(i, dt, E, np.amin(ctxt.get_dx_dp()), np.amax(ctxt.get_dx_dp()))
            xi = ctxt.get_x()
            dE_dx = ctxt.get_dE_dx()
            dUdL = dU_dlambda(dE_dx)

            if np.isnan(E):
                assert 0

            xyz = write(np.asarray(xi[:, :3]*10), masses)
            fh.write(xyz)
            # xyz = write(np.asarray(xi[:, :3]*10), masses)
            # fh.write(xyz)

            # xyz = write(np.asarray(xi[:, :3]*10), masses)
            # xyz_buffer.append(xyz)

        cur_dim -= 1


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
    cutoff = 500
    sampling_interval = 500
    if lamb == 0.0:
        max_iter = 200000
    elif lamb < 0.4:
        # max_iter = 1000000*2
        max_iter = 500000
    else:
        max_iter = 20000

    dt = 1e-3
    
    md_dudls = []

    # swap out integrator parameters
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
    all_kes = []
    all_es = []

    start = time.time()

    # old_ctxt = ctxt

    # ctxt = custom_ops.Context_f64(
    #     potentials,
    #     opt,
    #     params.astype(np.float64),
    #     old_ctxt.get_x(),
    #     v_t.astype(np.float64), # n
    #     dp_idxs.astype(np.int32)
    # )


    for i in range(max_iter):
        ctxt.step()
        if i % sampling_interval == 0 and i >= cutoff:
            
            E = ctxt.get_E()
            all_es.append(E)
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
            hess = ctxt.get_d2E_dx2()
            d2E_dxdp = compute_d2e_dxdp(potentials, params.astype(np.float64), xi, dp_idxs)
            dUdL = dU_dlambda(dE_dx)
            all_dudls.append(dUdL)
            dx_dp = ctxt.get_dx_dp()
            dv_dp = ctxt.get_dv_dp()

            dxdp = ctxt.get_dx_dp()
            ke = compute_ke(ctxt.get_v())
            all_kes.append(ke)
            speed = ns_per_day(time.time()-start, i)

            print(f"{lamb} \t {i} \t avg E: {np.mean(all_es):9.4f} \t {dUdL:9.4f} \t | dxdp max/min {np.amax(dxdp):9.4f} \t {np.amin(dxdp):9.4f} \t max mean/median deriv: {np.amax(np.mean(all_d2u_dldps, axis=0)):9.4f} \t {np.amax(np.median(all_d2u_dldps, axis=0)):9.4f} \t mean/median dudl {np.mean(all_dudls):9.4f} \t {np.median(all_dudls):9.4f} \t @ {speed:9.4f} ns/day \t  hess max/abs mean/min {np.amax(hess):9.4f}  {np.mean(np.abs(hess)):9.4f}  {np.amin(hess):9.4f} \t mp max/min {np.amax(d2E_dxdp):10.4f}  {np.amin(d2E_dxdp):10.4f} | dv_dp max/min {np.amax(dv_dp):10.4f} {np.amin(dv_dp):10.4f}")

            # print(lamb, "\t", i, "\t", E, "\t", dUdL, "\t", "| dxdp max/min", np.amax(dxdp), "\t", np.amin(dxdp), "\t | max mean/median deriv: ", np.amax(np.mean(all_d2u_dldps, axis=0)), "\t", np.amax(np.median(all_d2u_dldps, axis=0)), "\t mean/median dudl: ", np.mean(all_dudls), "\t", np.median(all_dudls), "+-", np.std(all_dudls), "\t @ ", speed, "ns/day")
#            if np.amax(dxdp) > 100:
#                raise ValueError("DXDP IS TOO LARGE")
            xyz = write(np.asarray(xi[:, :3]*10), masses)
            fh.write(xyz)
            xyz_buffer.append(xyz)


    # print("FINAL", lamb, "\t", i, "\t", E, "\t", dUdL, "\t", np.mean(all_dudls), "+-", np.std(all_dudls))

    return all_dudls, all_d2u_dldps, all_es
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
    # true_com = np.array([1.97698696, 1.90113478, 2.26042174]) # a-cd
    true_com = np.array([0, 0, 0]) # water
    centered = conf - mol_com  # centered to origin
    return true_com + centered/scale_factor 


def initialize_parameters(host_path=None, host_sys=None):
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
    if host_path is not None:
        sys_xml = open(host_path, 'r').read()
        system = mm.XmlSerializer.deserialize(sys_xml)
    else:
        system = host_sys
    _, (host_params, _), _ = serialize.deserialize_system(system)       
    epoch_combined_params = np.concatenate([host_params, smirnoff_params])

    return epoch_combined_params


def run_simulation(params):
    mol, lamb, lambda_idx, combined_params = params

    # conf = mol.GetConformer(0)
    # coords = conf.GetPositions()
    # np.random.seed(int(time.time()+float(lambda_idx)))
    # rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
    # # print("ROT_MATRIX", rot_matrix)
    # coords = np.matmul(coords, rot_matrix)
    # for idx, (x,y,z) in enumerate(coords):
    #     conf.SetAtomPosition(idx, (x,y,z))

    p = multiprocessing.current_process()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(lambda_idx % num_gpus)


    fname = "/home/yutong/Code/benchmarksets/input_files/BRD4/pdb/water.pdb"
    # omm_forcefield = app.ForceField('amber96.xml', 'amber99_obc.xml')
    omm_forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
    pdb = app.PDBFile(fname)
    system = omm_forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)
    


    coords = []
    for x,y,z in pdb.positions:
        coords.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    coords = np.array(coords)
    # print(coords)


    # filepath = 'examples/host_acd.xml'
    # filename, file_extension = os.path.splitext(filepath)
    # sys_xml = open(filepath, 'r').read()
    # system = mm.XmlSerializer.deserialize(sys_xml)
    # coords = np.loadtxt(filename + '.xyz').astype(np.float64)
    # coords = coords/10
    
    host_conf = coords

    host_potentials, (host_params, host_param_groups), host_masses = serialize.deserialize_system(system)
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    guest_potentials, smirnoff_params, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)
    guest_conf = rescale_and_center(guest_conf)

    # print("GUEST CONF", guest_conf)

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
    # print("filtering")
    combined_dp_idxs = np.argwhere(filter_groups(combined_param_groups, [7])).reshape(-1)
    combined_dp_idxs = combined_dp_idxs[0:2]
    # combined_dp_idxs = np.array([0])

    # print("combined_dp_idxs", combined_dp_idxs)

    # print("Number of parameter derivatives", combined_dp_idxs.shape)

    du_dls, du_dl_grads, all_es = minimize(
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
        lamb=lamb,
        lamb_idx=lambda_idx
    )

    # fname = "test_du_dl_grads_lambda_low_temp_charges"+str(lambda_idx)
    # print("Saving")
    # np.savez(fname, lamb=lamb, du_dls=du_dls, du_dl_grads=du_dl_grads)

    return lamb, du_dls, du_dl_grads, all_es, combined_dp_idxs

def train(true_dG):
    # fname = "/home/ubuntu/Relay/Code/benchmarksets/input_files/cd-set1/mol2/guest-"+str(1)+".mol2"
    fname = "/home/yutong/Code/benchmarksets/input_files/BRD4/mol2/ligand-4.mol2"
    guest_mol2 = open(fname, "r").read()
    # guest_mol2 = Chem.MolFromSmiles("O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)[C@H](O)[C@@H]3O")
    # mol = Chem.AddHs(guest_mol2)
    mol = Chem.MolFromMol2Block(guest_mol2, sanitize=True, removeHs=False, cleanupSubstructures=True)

    pool = multiprocessing.Pool(num_gpus)

    # AllChem.EmbedMolecule(mol, randomSeed=1337)
    # AllChem.EmbedMolecule(mol)


    fname = "/home/yutong/Code/benchmarksets/input_files/BRD4/pdb/water.pdb"
    # omm_forcefield = app.ForceField('amber96.xml', 'amber99_obc.xml') # for proteins
    omm_forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml') # for proteins
    pdb = app.PDBFile(fname)
    system = omm_forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)

    # pdb = PDBFile('input.pdb')
    omm_forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
    # fname = "/home/yutong/Code/benchmarksets/input_files/BRD4/pdb/ligand-4.pdb"
    # pdb = app.PDBFile(fname)



    # pdb = app.PDBFile('dummy.pdb')
    # top = app.Topology()
    # pos = simtk.unit.Quantity((), simtk.unit.angstroms)
    # modeller = app.Modeller(top, pos)
    # modeller.addSolvent(omm_forcefield, boxSize=mm.Vec3(2.5, 2.5, 2.5)*simtk.unit.nanometers, neutralize=False)
    # app.PDBFile.writeHeader(modeller.topology)
    # app.PDBFile.writeModel(modeller.topology, modeller.positions)
    # app.PDBFile.writeFooter(modeller.topology)
    # system = omm_forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)
    # assert 0

    starting_params = initialize_parameters(host_sys=system)
    # assert 0
    lr=3e-4
    opt_init, opt_update, get_params = optimizers.adam(lr)
    # opt_init, opt_update, get_params = optimizers.sgd(lr)

    opt_state = opt_init(starting_params)

    num_epochs = 50
    for epoch in range(num_epochs):

        print("turning off special ortho")
        # conf = mol.GetConformer(0)
        # coords = conf.GetPositions()
        # rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
        # coords = np.matmul(coords, rot_matrix)
        # for idx, (x,y,z) in enumerate(coords):
        #     conf.SetAtomPosition(idx, (x,y,z))

        print("===============Epoch "+str(epoch)+"=============")

        all_params = []
        all_lambdas = []
        # lambda_schedule = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 4.0, 6.0, 8.0, 10.0]
        lambda_schedule = [0.0, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.5,2.5,3.5,5.0,10.0,250.0]
        # lambda_schedule = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 5.0, 10.0]
        # lambda_schedule = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
        # lambda_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # lambda_schedule = [0.0, 25.0, 250.0, 2500.0, 100000.0]
        # lambda_schedule = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # lambda_schedule = [0.0]
        # lambda_schedule = [0.15]

        epoch_params = get_params(opt_state)
        # print("Loading parameters epoch_38.npz")
        # epoch_params = np.load("epoch_38.npz")['params']

        # print("saving ff params")
        # np.savez("epoch_run_2"+str(epoch), params=epoch_params)

        for lamb_idx, lamb in enumerate(lambda_schedule):
            params = (mol, lamb, lamb_idx, epoch_params)
            all_params.append(params)

        results = pool.map(run_simulation, all_params)

        all_lambdas = []
        all_mean_du_dls = []
        all_median_du_dls = []
        all_mean_du_dl_grads = []
        all_median_du_dl_grads = []
        all_energies = []
        for lamb, du_dls, du_dl_grads, nrgs, combined_dp_idxs in results:
            all_lambdas.append(lamb)
            all_mean_du_dls.append(np.mean(du_dls))
            all_median_du_dls.append(np.median(du_dls))
            all_mean_du_dl_grads.append(np.mean(du_dl_grads, axis=0))
            all_median_du_dl_grads.append(np.median(du_dl_grads, axis=0))
            all_energies.append(nrgs)

        all_energies = np.array(all_energies)
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

        pred_dG = np.trapz(all_mean_du_dls, all_lambdas)
        pred_dG_median = np.trapz(all_median_du_dls, all_lambdas)

        dG_grads = []
        dG_grads_median = []

        avg_enthalpy = np.mean(all_energies[-1]) - np.mean(all_energies[0])

        for p_idx in range(num_params):
            dG_grads.append(np.trapz(all_mean_du_dl_grads[:, p_idx], all_lambdas))
            dG_grads_median.append(np.trapz(all_median_du_dl_grads[:, p_idx], all_lambdas))
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


        print("L1_loss", L1_loss, "true vs pred", true_dG, pred_dG, pred_dG_median, avg_enthalpy)

        # fix me when going to multiple molecules
        # print("full_L2_grad", full_L2_grad)
        # opt_state = opt_update(epoch, full_L2_grad, opt_state)

    pool.close()

train(1.575*4.18)
